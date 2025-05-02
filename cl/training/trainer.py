# -------------  cl/training/trainer.py  --------------------
import torch._dynamo
# completely disable Dynamo/inductor so no libcuda.so check ever happens
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True
# disable Dynamo/inductor checks
import os
# tell torch.compile to use AOT‐eager by default, never inductor
os.environ["TORCH_COMPILE_BACKEND"] = "aot_eager"
import datetime, time, json, types, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch._dynamo
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm.auto import tqdm

import wandb, psutil, threading

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler

from cl.data.latent_webdataset import get_dataloader
from cl.training.lora        import apply_lora
from cl.model.architecture   import CubeDiffModel
from cl.training.seam_loss   import seam_loss

# -----------------------------------------------------------

class CubeDiffTrainer:
    def __init__(self, config,
                 pretrained_model_name="runwayml/stable-diffusion-v1-5",
                 output_dir="./outputs",
                 mixed_precision="bf16",
                 gradient_accumulation_steps=1):
        self.config  = config
        self.output_dir = Path(output_dir); self.output_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir = self.output_dir / "samples"; self.images_dir.mkdir(exist_ok=True)
        self.logs_dir   = self.output_dir / "logs";    self.logs_dir.mkdir(exist_ok=True)
        self.mixed_precision = mixed_precision
        self.model_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

        # Accelerator
        self.accelerator = Accelerator(mixed_precision=mixed_precision,
                                       gradient_accumulation_steps=gradient_accumulation_steps)

        # optional offline-wandb
        if "use_wandb" not in self.config:
            wandb.init(dir=str(self.logs_dir/"wandb"),
                       project=self.config.get("wandb_project","cubediff"),
                       name   =self.config.get("wandb_run_name", "cubediff_"+datetime.datetime.now().strftime("%H%M%S")),
                       mode="offline",
                       config=dict(self.config))

        # build model / VAE / text-enc once
        self.setup_model(pretrained_model_name)

        # ── PERCEPTUAL & BOUNDARY LOSS SETUP ──
        # self.perceptual_net = models.vgg16(pretrained=True).features.eval().to(self.accelerator.device)
        self.perceptual_net = models.vgg16(pretrained=True).features[:16].eval().to(dtype=torch.float16)
        for p in self.perceptual_net.parameters():
            p.requires_grad = False
        self.l1 = torch.nn.L1Loss()

        
    # --------------------------------------------------
    #  Model & tokenizer (unchanged except for LoRA hook)
    # --------------------------------------------------
    
    def setup_model(self, pretrained_model_name: str):
        # Note: This import should be done only once
        # Import the patching module - this applies the patches automatically
        # filter out unneeded arguments for forward() of peft
        # Note: This import should be done only once
        try:
            from cl.training import peft_patch  # This is the revised_peft_patch.py you created
        except ImportError:
            print("⚠️ peft_patch module not found, continuing without patching")
            
        # 1) Load stable-diffusion pipeline just to grab VAE & text-encoder
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.bfloat16,
            # revision="fp16", # 2025-5-1 Invalid rev id: fp16.
            use_safetensors=True,
        )
        # freeze VAE + text encoder
        self.vae          = pipe.vae.eval().requires_grad_(False)
        self.text_encoder = pipe.text_encoder.eval().requires_grad_(False)
        self.tokenizer    = pipe.tokenizer

        # 2) Noise scheduler (bfloat16)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler"
        )
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(torch.bfloat16))

        # 3) Instantiate CubeDiffModel (UNet wrapper)
        print("trainer.py - setup_model - before self.model = CubeDiffModel(pretrained_model_name)\n")
        self.model = CubeDiffModel(pretrained_model_name)
        print("trainer.py - setup_model - after self.model = CubeDiffModel(pretrained_model_name)\n")

        # 4) Inject LoRA *only* into the UNet backbone
        self.model.base_unet = apply_lora(
            self.model.base_unet,
            r=self.config["lora_r"],
            alpha=self.config["lora_alpha"],
        )

        # Ensure all model components use the same precision
        # model_dtype = torch.bfloat16  # Choose ONE precision type

        # Cast all model components to the same dtype
        self.model.base_unet = self.model.base_unet.to(dtype=self.model_dtype)
        self.text_encoder = self.text_encoder.to(dtype=self.model_dtype)
        self.vae = self.vae.to(dtype=self.model_dtype)

        # Cast scheduler tensors
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(self.model_dtype))

        # Add debug info
        print(f"Model components cast to {self.model_dtype}")

        # 5) Add manual parameter filtering to the base_unet forward method
        # This is a backup in case the module-level patching didn't work
        try:
            original_forward = self.model.base_unet.forward
            
            def filtered_forward(self_model, *args, **kwargs):
                # Filter out problematic parameters
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                if k not in ['input_ids', 'attention_mask', 'inputs_embeds', 
                                            'output_hidden_states', 'output_attentions', 
                                            'return_dict']}
                # Uncomment for debugging
                removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
                if removed:
                    print(f"Filtered out: {removed}")
                
                return original_forward(*args, **filtered_kwargs)
            
            # Replace the forward method
            self.model.base_unet.forward = types.MethodType(filtered_forward, self.model.base_unet)
            print("Added parameter filtering to LoRA U-Net")
        except Exception as e:
            print(f"⚠️ Could not add parameter filtering: {e}")

        # cast the adapted UNet to bfloat16 for memory/speed
        self.model.base_unet = self.model.base_unet.to(torch.bfloat16)

        # 5) Enable gradient checkpoints and circular padding
        #    Enable gradient checkpointing on the U-Net backbone only
        #    (saves ~30–40% memory at the cost of ~10–20% extra compute)
        # self.model.enable_gradient_checkpointing()
        self.model.base_unet.enable_gradient_checkpointing()
        for m in self.model.base_unet.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.padding_mode = "circular"

        # report
        tot   = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params {tot/1e6:.1f}M — LoRA trainable {train/1e6:.2f}M")

    # --------------------------------------------------
    # ✱✱✱  New dataloader creator  (latents, no JPEG)  ✱✱✱
    # --------------------------------------------------
    def build_dataloaders(self):
        self.train_dataloader = get_dataloader(
            self.config["dataset"],
            batch_size    = self.config["batch_size"],
            num_workers   = self.config.get("num_workers",4)
            )
        if "val_dataset" in self.config:
            self.val_dataloader = get_dataloader(
                self.config["val_dataset"],
                batch_size    = self.config["batch_size"],
                num_workers   = max(1, self.config["num_workers"]//2))
        else:
            self.val_dataloader = None

        # wrap only model + train loader
        print(f"trainer.py - build_dataloaders - before self.accelerator.prepare\n")
        self.model, self.train_dataloader = self.accelerator.prepare(
            self.model, self.train_dataloader
        )
        # Use the mixed_precision attribute to determine the dtype
        self.perceptual_net = self.perceptual_net.to(self.accelerator.device, dtype=self.model_dtype)
        self.text_encoder.to(self.accelerator.device)
        self.vae.to(self.accelerator.device)

    def boundary_loss(self, x):
        """
        x: either
        [B, F, C, H, W]   (5-D)
        or
        [B*F, C, H, W]    (4-D)
        Returns a scalar: average L1 across all seams.
        """
        # detect & reshape into [B, F, C, H, W]
        if x.dim() == 5:
            B, F, C, H, W = x.shape
            x5 = x
        elif x.dim() == 4:
            Bf, C, H, W = x.shape
            F = self.config.get("num_faces", 6)
            B = Bf // F
            x5 = x.view(B, F, C, H, W)
        else:
            raise ValueError(f"boundary_loss: unexpected tensor dim {x.dim()}")

        losses = []
        for i in range(F):
            # right edge of face i vs left edge of face (i+1)%F
            r = x5[:, i, :, :, -1]      # [B, C, H]
            l = x5[:, (i+1) % F, :, :,  0]  # [B, C, H]
            losses.append(self.l1(r, l))
        return sum(losses) / F
    
    # --------------------------------------------------
    #  Training loop  (shortened & adapted to latent input)
    # --------------------------------------------------
    def train(self):
        self.build_dataloaders()
        optim = AdamW((p for p in self.model.parameters() if p.requires_grad),
                      lr=self.config["learning_rate"], betas=(0.9,0.95), weight_decay=1e-2)
        sched = CosineAnnealingLR(optim, T_max=self.config["max_steps"])
        optim, sched = self.accelerator.prepare(optim, sched)

        sample_prompts        = ["A beautiful mountain lake at sunset with snow-capped peaks"]
        eval_every_n_samples  = self.config.get("eval_every_n_samples", 100)

        processed_samples = 0
        next_eval_at      = eval_every_n_samples
        gstep             = 0
        train_losses, val_losses = [], []
        g_start_tm = time.time()

        for step, batch in enumerate(self.train_dataloader):
            start_tm = time.time()
            with self.accelerator.accumulate(self.model):
                lat = batch["latent"].to(self.accelerator.device)              # [B,6,4,64,64]
                ids = batch["input_ids"].to(self.accelerator.device)
                mask = batch["attention_mask"].to(self.accelerator.device)

                noise     = torch.randn_like(lat)
                timesteps = torch.randint(0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (lat.shape[0],), device=lat.device)
                noisy_lat = self.noise_scheduler.add_noise(lat, noise, timesteps)

                with torch.no_grad():
                    txt_emb = self.text_encoder(ids, attention_mask=mask).last_hidden_state

                # print(f"trainer.py - train() - before pred = self.model, noisy_lat shape is {noisy_lat.shape} type is {noisy_lat.dtype}, timesteps is {timesteps}, txt_emb shape is {txt_emb.shape}, type is {txt_emb.dtype}\n")

                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    pred = self.model(
                        latents=noisy_lat,
                        timesteps=timesteps,
                        encoder_hidden_states=txt_emb
                    )
                # print(f"trainer.py - train() - after pred = self.model")

                # Add at the start of the first training iteration
                if step==0:
                    test_result = self.model(
                            latents=noisy_lat,
                            timesteps=timesteps,
                            encoder_hidden_states=txt_emb,
                            input_ids=ids,  # This should be filtered out
                            attention_mask=mask  # This should be filtered out
                        )
                    print("Parameter filtering through model path is working!")

                # ── NEW LOSS ──
                mse_loss = F.mse_loss(pred.float(), noise.float())

                # boundary loss
                # def boundary_loss(x):
                #     B6,C,H,W = x.shape
                #     B = B6//6
                #     x = x.view(B,6,C,H,W)
                #     losses=[]
                #     for i in range(6):
                #         r = x[:,i,:,:,-1]
                #         l = x[:,(i+1)%6,:,:,0]
                #         losses.append(self.l1(r, l))
                #     return sum(losses)/6
                
                # bdy = boundary_loss(pred.float()) * 0.1
                bdy = self.boundary_loss(pred) * self.config.get("boundary_weight", 0.1)

                # Convert perceptual_net to use the same precision as the model
                self.perceptual_net = self.perceptual_net.to(dtype=torch.bfloat16)

                # Then in the training loop:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Use the same dtype as your model
                    # perceptual loss
                    pred_rgb = self.decode_latents_to_rgb(pred)
                    true_rgb = self.decode_latents_to_rgb(noise)
                    fp = self.perceptual_net(pred_rgb)
                    ft = self.perceptual_net(true_rgb)
                    # perc = self.l1(fp, ft) * 0.01
                    perc = self.l1(fp, ft) * self.config.get("perceptual_weight", 0.01)

                loss = mse_loss + bdy + perc
                # loss = mse_loss

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    optim.step(); optim.zero_grad(); sched.step()
                    print(f"self.accelerator.sync_gradients is {self.accelerator.sync_gradients} - done\n")

            end_tm = time.time()
            print(f"in the training loop - step {step} cost {end_tm - start_tm:.2f} seconds\n")

            # ---- logging & sampling ----------------
            if self.accelerator.is_main_process:
                # 1) update sample count
                batch_size = batch["latent"].size(0)
                processed_samples += batch_size

                # 2) log train loss every N steps (unchanged)
                if gstep % self.config["log_every_n_steps"] == 0:
                    train_losses.append((processed_samples, loss.item()))
                    print(f"logging - batch_size is {batch_size}, samples {processed_samples:>4}  train-loss {loss.item():.4f}")

                # 3) every `eval_every_n_samples`, eval + sample‐gen
                if eval_every_n_samples and processed_samples >= next_eval_at:
                    val_loss = self.evaluate()
                    val_losses.append((processed_samples, val_loss))
                    print(f"→ val‐loss @ {processed_samples} samples: {val_loss:.4f}")

                    # generate panorama from current LoRA checkpoint - why the generated panorama is black ?!
                    self.generate_samples(sample_prompts, processed_samples)
                    next_eval_at += eval_every_n_samples
                    print(f"in the training loop - logging and sampling - self.accelerator.is_main_process is {self.accelerator.is_main_process} - after self.generate_samples - val_loss is {val_loss}\n")

                    # Replace with this more robust version:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    # Check LoRA weights using a more robust approach
                    try:
                        # Try different ways to access LoRA parameters
                        lora_params = None
                        
                        # Option 1: Modern PEFT structure
                        if hasattr(unwrapped.base_unet, "modules_to_save"):
                            for name, module in unwrapped.base_unet.named_modules():
                                if 'lora' in name.lower() and hasattr(module, 'weight') and module.weight.requires_grad:
                                    lora_params = module.weight
                                    print(f"Found LoRA weight in module: {name}")
                                    break
                        
                        # Option 2: Original approach (fallback)
                        elif hasattr(unwrapped.base_unet, "lora_layers") and len(unwrapped.base_unet.lora_layers) > 0:
                            lora_params = unwrapped.base_unet.lora_layers[0].weight
                        
                        # Option 3: Search for any LoRA-like parameters
                        else:
                            for name, param in unwrapped.base_unet.named_parameters():
                                if 'lora' in name.lower() and param.requires_grad:
                                    lora_params = param
                                    print(f"Found LoRA parameter: {name}")
                                    break
                        
                        if lora_params is not None:
                            w = lora_params.detach().cpu().view(-1)
                            print(f"LoRA weight stats - mean: {w.mean():.6f}  std: {w.std():.6f}")
                        else:
                            print("Warning: Could not find LoRA parameters for logging")
                            
                    except Exception as e:
                        print(f"Error accessing LoRA parameters: {e}")
                        print("Continuing training anyway...")

            gstep += 1
        g_end_tm = time.time()
        print(f"out of the training loop - all steps done, gstep is {gstep}, cost {g_end_tm - g_start_tm:.2f} seconda\n")

        # ----------------- save final LoRA ----------------------
        if self.accelerator.is_main_process:
            path = self.output_dir / "adapter_model.bin"
            # pull the real underlying model out of the Accelerator wrapper
            unwrapped = self.accelerator.unwrap_model(self.model)
            # grab just the U-Net’s weights
            unet_sd = unwrapped.base_unet.state_dict()
            torch.save(unet_sd, path)
            print(f"\n✔ saved U-Net adapter to {path}")

        # ───────────────────────────────────────────────────────────────
        # after all training, plot train & val curves
        try:
            steps_tr, loss_tr = zip(*train_losses) if train_losses else ([],[])
            steps_va, loss_va = zip(*val_losses)   if val_losses   else ([],[])

            plt.figure(figsize=(6,4))
            plt.plot(steps_tr, loss_tr, label="train")
            if steps_va:
                plt.plot(steps_va, loss_va, label="val")
            plt.xlabel("step")
            plt.ylabel("MSE loss")
            plt.legend()
            plt.tight_layout()
            out = self.output_dir / "loss_curve.png"
            plt.savefig(out)
            print(f"✔ loss curves saved to {out}")
        except Exception as e:
            print(f"⚠ could not plot loss curves: {e}")
        # ───────────────────────────────────────────────────────────────

    def decode_latents_to_rgb(self, lat: torch.Tensor) -> torch.Tensor:
        """
        Decode latents → RGB images via the frozen VAE.
        Accepts:
        - [B*6, C, H, W]  or
        - [B, 6, C, H, W]
        Returns:
        - [B*6, 3, H*8, W*8] float32 RGB
        """
        # flatten if needed
        if lat.ndim == 5:
            B, F, C, H, W = lat.shape
            lat = lat.reshape(B * F, C, H, W)

        # cast into VAE’s dtype (bfloat16)
        lat_bf16 = lat.to(self.vae.dtype)
        with torch.no_grad():
            out = self.vae.decode(lat_bf16)
            img = getattr(out, "sample", out)

        # return float32 for perceptual / plotting
        return img.float()
        
    # --------------------------------------------------
    # ✱✱✱  generate panorama after N steps for progress  ✱✱✱
    # --------------------------------------------------
    def generate_samples(self, prompts, step):
        from cl.inference.pipeline import CubeDiffPipeline
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process: 
            return

        tmp_ckpt = self.output_dir / f"tmp_{step}.bin"

        # pull the *underlying* model off of the accelerator
        unwrapped = self.accelerator.unwrap_model(self.model)
        # get just the UNet weights
        unet_sd   = unwrapped.base_unet.state_dict()
        torch.save(unet_sd, tmp_ckpt)
        print(f"in generate_samples -  unet_sd tmp_ckpt saved at {tmp_ckpt}\n")

        # 1) instantiate pipeline normally
        pipe = CubeDiffPipeline(
            pretrained_model_name="runwayml/stable-diffusion-v1-5"
        )

        # 2) read your U-Net weights back
        adapter_state = torch.load(tmp_ckpt, map_location="cuda")
        
        # 3) cast them to the same dtype as the pipeline’s U-Net
        unet_dtype = next(pipe.model.base_unet.parameters()).dtype
        for k, v in adapter_state.items():
            adapter_state[k] = v.to(unet_dtype)
        
        # 4) load *only* into the U-Net
        missing, unexpected = pipe.model.base_unet.load_state_dict(adapter_state, strict=False)
        print("in generate_samples - loaded UNet adapter, missing size: ", len(missing))

        print(f"in generate_samples -  create inference pipe and generate the pnoaram based on the given prompts\n")
        for i,p in enumerate(prompts):
            # why the generated panorama is black ?!
            pano = pipe.generate(p, num_inference_steps=30, guidance_scale=7.5)
            temp = self.images_dir / f"step{step}_{i}.jpg"
            pano.save(self.images_dir / f"step{step}_{i}.jpg")
            print(f"in generate_samples - prompt is {p}, saved pano at {temp}\n")

        tmp_ckpt.unlink()
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
        print(f"in generate_samples - prompts is {prompts}, step is {step}, pano saved at {temp}, unet tmp_ckpt {tmp_ckpt} was removed\n")

    # ---------------------------------------------------------------------
    # ✱✱✱  New: run a full pass over val_dataloader and return avg loss. ✱✱✱
    # reuse boundary_loss helper inside train()
    # and decode_latents_to_rgb + self.perceptual_net   
    
    def evaluate(self):
        """Compute avg. (MSE + boundary + perceptual) loss on the validation set."""
        if self.val_dataloader is None:
            return float("nan")

        self.model.eval()
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                lat = batch["latent"].to(self.accelerator.device)
                ids = batch["input_ids"].to(self.accelerator.device)
                mask = batch["attention_mask"].to(self.accelerator.device)

                txt_emb = self.text_encoder(ids, attention_mask=mask).last_hidden_state

                noise     = torch.randn_like(lat)
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (lat.shape[0],),
                    device=lat.device
                )
                noisy_lat = self.noise_scheduler.add_noise(lat, noise, timesteps)
                # print(f"trainer.py - evaluate() - before pred = self.model\n")
                pred = self.model(
                    latents=noisy_lat,
                    timesteps=timesteps,
                    encoder_hidden_states=txt_emb
                )

                # 1) MSE
                mse = F.mse_loss(pred.float(), noise.float(), reduction="mean")

                # 2) boundary (exact same boundary_loss as in train())
                # def boundary_loss(x):
                #     B6,C,H,W = x.shape
                #     B = B6//6
                #     x = x.view(B,6,C,H,W)
                #     losses=[]
                #     for i in range(6):
                #         r = x[:,i,:,:,-1]
                #         l = x[:,(i+1)%6,:,:,0]
                #         losses.append(self.l1(r, l))
                #     return sum(losses)/6
                bdy = self.boundary_loss(pred.float()) * self.config.get("boundary_weight", 0.1)

                # 3) perceptual

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Use the same dtype as your model
                    # perceptual loss
                    pred_rgb = self.decode_latents_to_rgb(pred)
                    true_rgb = self.decode_latents_to_rgb(noise)
                    fp = self.perceptual_net(pred_rgb)
                    ft = self.perceptual_net(true_rgb)
                    # perc = self.l1(fp, ft) * 0.01
                    perc = self.l1(fp, ft) * self.config.get("perceptual_weight", 0.01)

                loss = mse + bdy + perc

                total_loss   += loss.item() * lat.size(0)
                total_samples+= lat.size(0)

        self.model.train()
        return total_loss / total_samples if total_samples > 0 else float("nan")

# EOF -------------------------------------------------------------------------
