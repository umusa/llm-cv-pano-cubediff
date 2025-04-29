import torch._dynamo
# completely disable Dynamo/inductor so no libcuda.so check ever happens
torch._dynamo.disable()
# -------------  cl/training/trainer.py  --------------------

import os
# tell torch.compile to use AOT‐eager by default, never inductor
os.environ["TORCH_COMPILE_BACKEND"] = "aot_eager"
import torch
import torch._dynamo
# suppress any Dynamo/inductor errors so we always fall back cleanly
torch._dynamo.config.suppress_errors = True

import datetime, time, json, types, numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import wandb, psutil, threading

from cl.data.latent_webdataset import get_dataloader
from cl.training.lora        import apply_lora
from cl.model.architecture   import CubeDiffModel
# -----------------------------------------------------------

class CubeDiffTrainer:
    def __init__(self, config,
                 pretrained_model_name="runwayml/stable-diffusion-v1-5",
                 output_dir="./outputs",
                 mixed_precision="bf16",                 # <-- default bf16
                 gradient_accumulation_steps=1):
        self.config  = config
        self.output_dir = Path(output_dir); self.output_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir = self.output_dir / "samples"; self.images_dir.mkdir(exist_ok=True)
        self.logs_dir   = self.output_dir / "logs";    self.logs_dir.mkdir(exist_ok=True)
        self.mixed_precision = mixed_precision

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

    # --------------------------------------------------
    #  Model & tokenizer (unchanged except for LoRA hook)
    # --------------------------------------------------
    def setup_model(self, pretrained_model_name):
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name,
                    torch_dtype=torch.bfloat16)                    # bf16 to save vram

        self.vae, self.text_encoder, self.tokenizer = pipe.vae, pipe.text_encoder, pipe.tokenizer
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Set up noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name,
            subfolder="scheduler",
        )
        dtype = torch.float16 if self.mixed_precision == "fp16" else torch.float32
        # align scheduler buffers with chosen dtype
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(dtype))

        # build CubeDiff UNet (float32 → optimiser stability)
        self.model = CubeDiffModel(pretrained_model_name).to(dtype=torch.float32)
        print(f"trainer.py - CubeDiffTrainer - after build CubeDiffModel: {self.model._get_name()}")
        
        # … after building self.model = CubeDiffModel(...)
        # Use our custom LoRA injection from cl/training/lora.py
        # which directly patches the UNet’s Q/K/V/out modules.
        self.model = apply_lora(
          self.model,
          r=self.config["lora_r"],
          alpha=self.config["lora_alpha"]
        )
        self.model.enable_gradient_checkpointing()

        # Try to compile with torch.compile, but fall back to eager if libcuda.so is missing.
        # try:
        #     # you can also do backend="eager" to disable any inductor kernels
        #     self.model = torch.compile(self.model, backend="inductor")
        #     self.model.base_unet = torch.compile(self.model.base_unet, backend="inductor")
        # except AssertionError as e:
        #     print("⚠️  torch.compile(inductor) failed (missing libcuda.so?), falling back to eager:", e)
        #     self.model = torch.compile(self.model, backend="eager")
        #     self.model.base_unet = torch.compile(self.model.base_unet, backend="eager")

        # Enable gradient checkpointing on the U-Net
        self.model.base_unet.enable_gradient_checkpointing()
        print("trainer.py - Enabled gradient checkpointing on base_unet")

        # Compile with torch.compile() — thanks to TORCH_COMPILE_BACKEND="aot_eager",
        # this will use the aot_eager backend (no inductor), and suppress errors.
        # self.model = torch.compile(self.model)
        # self.model.base_unet = torch.compile(self.model.base_unet)

        tot = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params {tot/1e6:.1f} M — LoRA trainable {train/1e6:.2f} M")

    # --------------------------------------------------
    # ✱✱✱  New dataloader creator  (latents, no JPEG)  ✱✱✱
    # --------------------------------------------------
    def build_dataloaders(self):
        self.train_dataloader = get_dataloader(
                self.config["dataset"],
                batch_size = self.config["batch_size"],
                num_workers       = self.config.get("num_workers", 4))
        print(f"trainer.py - CubeDiffTrainer - after train_dataloader\n")
        if "val_dataset" in self.config:
            self.val_dataloader = get_dataloader(
                    self.config["val_dataset"],
                    batch_size = self.config["batch_size"],
                    num_workers       = max(1, self.config["num_workers"]//2))
        else:
            self.val_dataloader = None

        # Accelerator handles DDP sharding / fp16-hooks
        # objs = [self.model, self.train_dataloader]
        # if self.val_dataloader: objs.append(self.val_dataloader)
        # self.model, self.train_dataloader, *rest = self.accelerator.prepare(*objs)
        # if rest: self.val_dataloader, = rest

        # Wrap only model + train loader in Accelerator.
        # Checking “if self.val_dataloader” would call __len__()
        # on the WebDataset (no __len__ → TypeError), so skip it here.
        # shard/move model + train loader to GPU (or DDP device)
        self.model, self.train_dataloader = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
        )
        # Leave self.val_dataloader unwrapped; run it manually under
        # torch.no_grad() in eval/sampling loop if need it.
        # This is standard practice in multi-GPU setups.
        # now that we've moved into GPU/DDP, also put text_encoder and VAE on that device
        self.text_encoder.to(self.accelerator.device)
        self.vae.to(self.accelerator.device)

    # --------------------------------------------------
    #  Training loop  (shortened & adapted to latent input)
    # --------------------------------------------------
    def train(self):
        self.build_dataloaders()
        print(f"trainer.py - CubeDifftrainer - after build_dataloaders\n")
        # Optimiser only sees LoRA weights
        optim = AdamW((p for p in self.model.parameters() if p.requires_grad),
                      lr=self.config["learning_rate"], betas=(0.9,0.95), weight_decay=1e-2)
        sched = CosineAnnealingLR(optim, T_max=self.config["max_steps"])

        optim, sched = self.accelerator.prepare(optim, sched)

        sample_prompts     = ["A beautiful mountain lake at sunset with snow-capped peaks"]

        # add a new config param for sample‐level eval:
        eval_every_n_samples = self.config.get("eval_every_n_samples", 100)

        # counters for (a) how many raw samples we’ve processed,
        # and (b) threshold for next eval
        processed_samples = 0
        next_eval_at      = eval_every_n_samples
        gstep = 0
        train_losses, val_losses = [], []
        g_start_tm = time.time()
        print(f"trainer.py - CubeDiffTrainer - the training loop started with batch_size as {self.config['batch_size']}, eval_every_n_samples as {eval_every_n_samples} ...")
        for step, batch in enumerate(self.train_dataloader):
            print(f"---------------- training step {step} ----------------")
            # if isinstance(batch, dict):
            #     print(f"Batch keys: {batch.keys()}")
            #     for k, v in batch.items():
            #         print(f"Key: {k}, Type: {type(v)}, ", end="")
            #         if isinstance(v, torch.Tensor):
            #             print(f"Shape: {v.shape}")
            #         else:
            #             print(f"Value: {v}")
            start_tm = time.time()
            with self.accelerator.accumulate(self.model):
                # … inside for step, batch in enumerate(self.train_dataloader):
                lat = batch["latent"].to(self.accelerator.device)              # [B,6,4,64,64]
                input_ids      = batch["input_ids"].to(self.accelerator.device)     # [B, L]
                attention_mask = batch["attention_mask"].to(self.accelerator.device)# [B, L]

                noise     = torch.randn_like(lat)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                          (lat.shape[0],), device=lat.device)
                noisy_lat = self.noise_scheduler.add_noise(lat, noise, timesteps)

                # 1) precompute CLIP text embeddings (frozen, no grad)
                with torch.no_grad():
                    clip_out = self.text_encoder(
                        input_ids, # panorama caption token
                        attention_mask=attention_mask
                    )
                    txt_emb = clip_out.last_hidden_state

                # 2) forward through CubeDiffModel via encoder_hidden_states
                pred = self.model(
                    latents=noisy_lat,
                    timesteps=timesteps,
                    encoder_hidden_states=txt_emb
                )
                                
                # print(f"trainer.py - CubeDiffTrainer - after text_encoder - after self.model(noisy_lat, timesteps, txt_emb) - get pred\n")

                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                print(f"training loss {loss}\n")
                self.accelerator.backward(loss)
                # print(f"trainer.py - CubeDiffTrainer - after accelerator backward\n")

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
        print("in generate_samples - loaded UNet adapter, missing: ", missing)

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
    def evaluate(self):
        """Compute average MSE on the validation set."""
        if self.val_dataloader is None:
            return float("nan")

        self.model.eval()
        total, running = 0, 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                lat = batch["latent"].to(self.accelerator.device)
                ids = batch["input_ids"].to(self.accelerator.device)
                mask = batch["attention_mask"].to(self.accelerator.device)

                txt_emb = self.text_encoder(
                    ids, attention_mask=mask
                ).last_hidden_state

                noise     = torch.randn_like(lat)
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (lat.shape[0],),
                    device=lat.device
                )
                noisy_lat = self.noise_scheduler.add_noise(lat, noise, timesteps)
                pred    = self.model(
                    latents=noisy_lat,
                    timesteps=timesteps,
                    encoder_hidden_states=txt_emb
                )

                loss = torch.nn.functional.mse_loss(pred.float(), noise.float(), reduction="mean")
                running += loss.item() * lat.size(0)
                total   += lat.size(0)

        self.model.train()
        return running / total if total>0 else float("nan")

# EOF -------------------------------------------------------------------------
