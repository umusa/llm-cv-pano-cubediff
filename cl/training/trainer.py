# -------------  cl/training/trainer.py  --------------------
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

# import ctypes
# adjust to your actual path if needed
# ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so", mode=ctypes.RTLD_GLOBAL)

# import torch._dynamo
import os
#------- completely disable Dynamo/inductor so no libcuda.so check ever happens
# torch._dynamo.disable()
# torch._dynamo.config.suppress_errors = True
# ----------------------------

# Update LD_LIBRARY_PATH to include where libcuda.so actually is
os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
# Set torch compile backend
os.environ["TORCH_COMPILE_BACKEND"] = "inductor"
# force‐load the real driver

# disable Dynamo/inductor checks
import types
# import inspect

# tell torch.compile to use AOT‐eager by default, never inductor
# os.environ["TORCH_COMPILE_BACKEND"] = "aot_eager"

# Before running your training script
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"

import datetime, time, json, types, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm.auto import tqdm
import bitsandbytes as bnb

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

import wandb, psutil, threading

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DDPMScheduler

from cl.data.latent_webdataset import get_dataloader
from cl.model.architecture   import CubeDiffModel
from cl.training.seam_loss   import seam_loss

from diffusers import UNet2DConditionModel
from peft import TaskType

# -----------------------------------------------------------
# — Monkey-patch UNet2DConditionModel.forward to swallow any extra kwargs —
orig_unet_forward = UNet2DConditionModel.forward
def _patched_unet_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
    """
    Accept the three required args, ignore anything else.
    Guards against 'decoder_input_ids', 'use_cache', etc.
    """
    # print(f"trainer.py - _patched_unet_forward - sample dtype is {sample.dtype}, timestep dtype is {timestep.dtype}, encoder_hidden_states dtype is {encoder_hidden_states.dtype}\n")
    return orig_unet_forward(self, sample, timestep, encoder_hidden_states)

UNet2DConditionModel.forward = _patched_unet_forward
# -------------------------------------------------------------
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
                                       gradient_accumulation_steps=gradient_accumulation_steps,
                                    #    dataloader_pin_memory=True,
                                    #    offload_state=True,            # buffers & optimizer states → CPU
                                    #    offload_optimizer=True,        # optimizer state → CPU
                                    #    offload_parameters=True        # model weights → CPU when unused
                                    )

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

        self.lora_logger = LoRALogger(
            output_dir=Path(self.config["output_dir"]) / "lora_logs"
        )
        self.global_iter = 0
    
    # --------------------------------------------------
    #  Model & tokenizer (unchanged except for LoRA hook)
    # --------------------------------------------------
        
    def setup_model(self, pretrained_model_name: str):
        # Note: This import should be done only once
        # Import the patching module - this applies the patches automatically
        # filter out unneeded arguments for forward() of peft
        # Note: This import should be done only once
        try:
            print(f"trainer.py - CubeDiffTrainer - setup - importing peft_patch")
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
        self.model = CubeDiffModel(pretrained_model_name)
        
        # using BitsAndBytesConfig(load_in_4bit) so no need for prepare_model_for_kbit_training
        # 4) Inject LoRA *only* into the UNet backbone
        # optional 8-bit kbit training (saves VRAM)
        # — Quantize U-Net to 8-bit for both speed & <16 GB memory
        # PEFT’s helper expects an 8-bit-loaded model, freezes its weights, and wraps them in BNB’s 8-bit linear layers.
        # self.model.base_unet = prepare_model_for_kbit_training(self.model.base_unet)
        # print("[setup_model] UNet quantized to 8-bit")

        # do not compile it due to errors and no significant memory savings and speedup
        # — JIT-compile U-Net via inductor for ~2–3× faster steps
        # print(f"trainer.py - cubeDiffTrainer - setuop_model - before torch.compile(self.model.base_unet, backend='nvfuser')\n")
        # self.model.base_unet = torch.compile(self.model.base_unet, backend="inductor")
        # self.model.base_unet = torch.compile(
        #                                         self.model.base_unet,
        #                                         backend="nvfuser",
        #                                         fullgraph=True,
        #                                     )
        # print(f"trainer.py - cubeDiffTrainer - setuop_model - after torch.compile(self.model.base_unet, backend='nvfuser')\n")
        
        unet = self.model.base_unet

        # stub prepare_inputs_for_generation(sample, timestep, encoder_hidden_states)
        if not hasattr(unet, "prepare_inputs_for_generation"):
            def _prepare_inputs_for_generation(self, sample, timestep, encoder_hidden_states, **kwargs):
                return {
                    "sample": sample,
                    "timestep": timestep,
                    "encoder_hidden_states": encoder_hidden_states
                }
            unet.prepare_inputs_for_generation = types.MethodType(
                _prepare_inputs_for_generation, unet
            )

        # stub _prepare_encoder_decoder_kwargs_for_generation(decoder_input_ids, **kwargs)
        if not hasattr(unet, "_prepare_encoder_decoder_kwargs_for_generation"):
            def _prepare_encoder_decoder_kwargs_for_generation(self, decoder_input_ids, **kwargs):
                # for UNet, we don’t need extra kwargs—return empty dict
                return {}
            unet._prepare_encoder_decoder_kwargs_for_generation = types.MethodType(
                _prepare_encoder_decoder_kwargs_for_generation, unet
            )

        # reassign back
        self.model.base_unet = unet
        # — end stub —

        # 5) Inject PEFT‐LoRA adapters (Hu et al. 2021)
        lora_cfg = LoraConfig(
            r=self.config["lora_r"], # 4, 
            lora_alpha=self.config["lora_alpha"], # 16,
            # target_modules=["to_q.0","to_k.0","to_v.0","to_out.0"], # That will ensure I am actually injecting adapters into all the Q/K/V and output projections .
            target_modules=[
                            "to_q",       # matches every q-proj Linear/Conv
                            "to_k",       # matches every k-proj
                            "to_v",       # matches every v-proj
                            "to_out.0",   # matches ONLY the Linear inside the ModuleList
                            ],
            lora_dropout=0.05,
            bias="none"            
        )
        # PEFT-LoRA: target the first (Linear) element inside each ModuleList
        
        self.model.base_unet = get_peft_model(self.model.base_unet, lora_cfg)
        print(f"[LoRA] trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # Ensure all model components use the same precision
        # Cast all model components to the same dtype
        self.model.base_unet = self.model.base_unet.to(dtype=self.model_dtype)
        self.text_encoder = self.text_encoder.to(dtype=self.model_dtype)
        self.vae = self.vae.to(dtype=self.model_dtype)

        # Cast scheduler tensors
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(self.model_dtype))

        # Add debug info
        print(f"trainer.py - CubeDiffTrainer- CubeDiff Model components cast to {self.model_dtype}")

        # cast the adapted UNet to bfloat16 for memory/speed
        self.model.base_unet = self.model.base_unet.to(torch.bfloat16)

        # 5) Enable gradient checkpoints and circular padding
        #   Enable gradient checkpointing on the U-Net backbone only
        #    (saves ~30–40% memory at the cost of ~10–20% extra compute)
        # self.model.enable_gradient_checkpointing()
        print(f"trainer.py - CubeDiffTrainer - CubeDiff Model enabled gradient checkpointing\n")
        self.model.base_unet.enable_gradient_checkpointing()
        # if diffusers>=0.18, which shards attention internals to slash peak usage.
        print(f"trainer.py - CubeDiffTrainer - CubeDiff Model enabled xformers\n")
        self.model.base_unet.enable_xformers_memory_efficient_attention()
        for m in self.model.base_unet.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.padding_mode = "circular"

        # report
        tot   = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"trainer.py - CubeDiffTrainer - setup_model done - Total params {tot/1e6:.1f}M — LoRA trainable {train/1e6:.2f}M")

    # --------------------------------------------------
    # ✱✱✱  New dataloader creator  (latents, no JPEG)  ✱✱✱
    # --------------------------------------------------
    
    def build_dataloaders(self):
        print(f"trainer.py - CubeDiffTrainer - Building dataloaders with config: {self.config}")
        try:
            self.train_dataloader = get_dataloader(
                wds_path=self.config["dataset"],
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"]
                # shuffle=False,
                # pin_memory=True,
                # persistent_workers=True,
            )
            print(f"Train dataloader created successfully")
            
            # Verify the dataloader by trying to get one batch
            try:
                batch_iter = iter(self.train_dataloader)
                first_batch = next(batch_iter)
                print(f"Successfully loaded a sample batch with keys: {first_batch.keys()}")
            except StopIteration:
                print("Warning: Dataloader is empty, no samples found")
            except Exception as e:
                print(f"Warning: Failed to load sample batch: {e}")
            
            if "val_dataset" in self.config:
                self.val_dataloader = get_dataloader(
                    self.config["val_dataset"],  
                    batch_size=self.config["batch_size"],
                    num_workers=self.config["num_workers"] 
                )
            else:
                self.val_dataloader = None

            # Set random seed
            set_seed(self.config.get("seed", 42))

            # # Enable optimizations
            # print("Enabling optimizations for U-Net")
            # self.model.base_unet.enable_xformers_memory_efficient_attention()
            # self.model.base_unet.enable_gradient_checkpointing()

            print("Preparing model and dataloader with accelerator")
            self.model, self.train_dataloader = self.accelerator.prepare(
                self.model, self.train_dataloader
            )
            
            # Move components to device
            device = self.accelerator.device
            dtype = self.model_dtype
            print(f"Moving components to {device} with dtype {dtype}")
            self.perceptual_net = self.perceptual_net.to(device, dtype=dtype)
            self.text_encoder = self.text_encoder.to(device)
            self.vae = self.vae.to(device)
            
            print("Dataloader building completed successfully")
        except Exception as e:
            print(f"Error building dataloaders: {e}")
            import traceback
            traceback.print_exc()
            raise


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

        # only update the small LoRA adapter params
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        # ← switch from AdamW to 8-bit Adam for much smaller optimizer state
        optim = bnb.optim.Adam8bit(
            lora_params,
            lr=self.config["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.0
        )
        print(f"trainer.py - CubeDiffTrainer - train - optimizer Adam8bit created\n")

        sched = CosineAnnealingLR(optim, T_max=self.config["max_steps"])
        optim, sched = self.accelerator.prepare(optim, sched)

        sample_prompts        = ["A beautiful mountain lake at sunset with snow-capped peaks"]
        print(f"trainer.py - CubeDiffTrainer - train - sample_prompts is {sample_prompts}\n")

        eval_every_n_samples  = self.config.get("eval_every_n_samples", 100)
        processed_samples = 0
        next_eval_at      = eval_every_n_samples
        gstep             = 0
        train_losses, val_losses = [], []
        g_start_tm = time.time()

        for batch_indx, batch in enumerate(self.train_dataloader):
            print(f"*** - trainer.py - cubedifftrainer - train loop - Batch {batch_indx}: batch size is {batch['latent'].size(0)}")
            log_batch_memory(batch_indx, batch, stage="BEFORE accelerator.accumulate")

            start_tm = time.time()
            with self.accelerator.accumulate(self.model):
                lat = batch["latent"].to(self.accelerator.device, dtype=self.model_dtype)              # [B,6,4,64,64]
                ids = batch["input_ids"].to(self.accelerator.device)
                mask = batch["attention_mask"].to(self.accelerator.device)

                noise     = torch.randn_like(lat)
                timesteps = torch.randint(0,
                                        self.noise_scheduler.config.num_train_timesteps,
                                        (lat.shape[0],), device=lat.device)
                noisy_lat = self.noise_scheduler.add_noise(lat, noise, timesteps)

                with torch.no_grad():
                    txt_emb = self.text_encoder(ids, attention_mask=mask).last_hidden_state
                    # ← CAST TO bfloat16 so K/V come out in bf16
                    txt_emb = txt_emb.to(self.accelerator.device, dtype=self.model_dtype)

                # print(f"trainer.py - train() - before pred = self.model, lat shape is {lat.shape} and dtype is {lat.dtype}, noisy_lat shape is {noisy_lat.shape} type is {noisy_lat.dtype}, timesteps is {timesteps}, txt_emb shape is {txt_emb.shape}, type is {txt_emb.dtype}\n")

                with self.accelerator.autocast():
                     pred = self.model(
                         latents=noisy_lat,
                         timesteps=timesteps,
                         encoder_hidden_states=txt_emb
                     )

                # print(f"trainer.py - train() - after pred = self.model")

                # Add at the start of the first training iteration
                if batch_indx==0:
                    test_result = self.model(
                            latents=noisy_lat,
                            timesteps=timesteps,
                            encoder_hidden_states=txt_emb,
                            input_ids=ids,  # This should be filtered out
                            attention_mask=mask  # This should be filtered out
                        )
                    # print("Parameter filtering through model path is working!")

                # ── NEW LOSS ──
                mse_loss = F.mse_loss(pred.float(), noise.float())

                # boundary loss
                bdy = self.boundary_loss(pred) * self.config.get("boundary_weight", 0.1)

                # Convert perceptual_net to use the same precision as the model
                self.perceptual_net = self.perceptual_net.to(dtype=torch.bfloat16)

                # Then in the training loop:
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Use the same dtype as the model
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # perceptual loss
                    pred_rgb = self.decode_latents_to_rgb(pred)
                    true_rgb = self.decode_latents_to_rgb(noise)
                    fp = self.perceptual_net(pred_rgb)
                    ft = self.perceptual_net(true_rgb)
                    # perc = self.l1(fp, ft) * 0.01
                    perc = self.l1(fp, ft) * self.config.get("perceptual_weight", 0.01)

                loss = mse_loss + bdy + perc
                # loss = mse_loss

                self.accelerator.backward(loss)  # ← compute gradients
                # if self.global_iter%self.config['log_every_n_steps']==0:
                #     print(f"trainer.py - CubeDiffTrainer - train - self.config[log_every_n_steps] is {self.config['log_every_n_steps']}, self.accelerator.sync_gradients is {self.accelerator.sync_gradients} - self.global_iter is {self.global_iter}\n")
                
                if self.accelerator.sync_gradients:
                    # here all replicas have the fully synchronized, accumulated grads
                    # see each adapter’s true gradient (after accumulation + all‐GPU sync) exactly once per update, 
                    # rather than on every micro‐batch.
                    print(f"→ Update done (sync_gradients={self.accelerator.sync_gradients}) for global_iter {self.global_iter}\n")
                    
                    # Log LoRA adapter weight stats to verify fine-tuning —
                    if (self.global_iter%self.config["log_every_n_steps"])==0:
                        print(f"trainer.py - CubeDiffTrainer - train - log lora - self.global_iter is {self.global_iter}\n")
                        self.lora_logger.record_batch(self.model, self.accelerator, self.global_iter)
                        self.lora_logger.plot_up_to(self.global_iter)
                        print(f"trainer.py - CubeDiffTrainer - train - self.lora_logger.record_batch done for self.global_iter { self.global_iter}, plot_up_to done\n")

                    # 1) gradient clipping
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    # 2) step the optimizer (handles unscaling, all‐reduce, zero‐grad)
                    optim.step()
                    
                    # 3) then update your LR scheduler
                    sched.step() 

                    # 4) zero gradients (optional - can be after or before step)
                    optim.zero_grad()   # ← now clear grad for next iteration        
                self.global_iter += 1
                print(f"trainer.py - cubedifftrainer - train - after self.accelerator.sync_gradients - self.global_iter is {self.global_iter}\n")
            

            end_tm = time.time()
            print(f"in the training loop - out of accumulate - batch_indx {batch_indx} cost {end_tm - start_tm:.2f} seconds\n")

            # --- Log after computation ---
            log_batch_memory(batch_indx, batch, stage="AFTER accelerate.accumulate")

            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak CUDA memory this batch: {peak:.2f} GB")

            print(f"Batch {batch_indx} time: {end_tm - start_tm:.2f} seconds\n")
            
            # ---- logging & sampling ----------------
            print(f"trainer.py - CubeDiffTrainer - train - self.accelerator.is_main_process is {self.accelerator.is_main_process}")
            if self.accelerator.is_main_process:
                print(f"trainer.py - CubeDiffTrainer - train - logging and sampling")
                # 1) update sample count
                batch_size = batch["latent"].size(0)
                processed_samples += batch_size
                # print(f"trainer.py - CubeDiffTrainer - train - logging and sampling - processed_samples is {processed_samples}, eval_every_n_samples is {eval_every_n_samples}, next_eval_at is {next_eval_at}")

                # 2) log train loss every N steps (unchanged)
                if gstep % self.config["log_every_n_steps"] == 0:
                    train_losses.append((processed_samples, loss.item()))
                    print(f"trainer.py - logging - self.accelerator.is_main_process is {self.accelerator.is_main_process} - log_every_n_steps is {self.config['log_every_n_steps']} - global_iter is {self.global_iter} - batch_size is {batch_size}, samples {processed_samples:>4}  train-loss {loss.item():.4f}")

                # 3) every `eval_every_n_samples`, eval + sample‐gen
                if eval_every_n_samples and processed_samples >= next_eval_at:
                    val_loss = self.evaluate()
                    val_losses.append((processed_samples, val_loss))
                    print(f"→ val‐loss @ {processed_samples} samples: {val_loss:.4f}, eval_every_n_samples is {eval_every_n_samples}")

                    # generate panorama from current LoRA checkpoint 
                    self.generate_samples(sample_prompts, processed_samples)
                    next_eval_at += eval_every_n_samples
                    print(f"trainer.py - logging - after self.generate_samples\n")

                    # 4) Verify your U-Net’s LoRA adapters are actually updating
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
                                    print(f"trainer.py - logging - Found LoRA weight in module: {name}")
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
                            print(f"trainer.py - logging - LoRA weight stats - mean: {w.mean():.6f}  std: {w.std():.6f}")
                        else:
                            print("Warning: Could not find LoRA parameters for logging")
                            
                    except Exception as e:
                        print(f"train - logging - Error accessing LoRA parameters: {e}")
                        print("train - logging - Continuing training anyway...")

            gstep += 1
        g_end_tm = time.time()
        # ----------------- end of training loop ----------------------
        print(f"out of the training loop - all steps done, gstep is {gstep}, self.global_iter is {self.global_iter}, cost {g_end_tm - g_start_tm:.2f} seconds\n")

        # ----------------- save final LoRA ----------------------
        if self.accelerator.is_main_process:
            path = self.output_dir / "adapter_model.bin"
            # pull the real underlying model out of the Accelerator wrapper
            unwrapped = self.accelerator.unwrap_model(self.model)
            # grab just the U-Net’s weights
            unet_sd = unwrapped.base_unet.state_dict()
            torch.save(unet_sd, path)
            print(f"\n✔ saved U-Net adapter to {path}")
            self.lora_logger.finalize()
            print(f"trainer.py - CubeDiffTrainer - train - lora final logging done\n")

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
        print("generate_samples - loaded CubeDiffPipeline.model.base_unet from UNet adapter, missing block/component size: ", len(missing))

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

        import torch._dynamo
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True

        self.model.eval()
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                # cast to the same dtype the U-Net was compiled for
                lat = batch["latent"].to(self.accelerator.device, dtype=self.model_dtype)
                # print(f"trainer.py - evaluate() - after batch[latent].to(self.accelerator.device,  dtype=self.model_dtype), lat dtype is {lat.dtype}\n")

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

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.model(
                        latents=noisy_lat,
                        timesteps=timesteps,
                        encoder_hidden_states=txt_emb
                    )

                # 1) MSE
                mse = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                # print(f"trainer.py - evaluate() - after pred = self.model, mse is {mse}\n")

                # 2) boundary (exact same boundary_loss as in train())
                bdy = self.boundary_loss(pred.float()) * self.config.get("boundary_weight", 0.1)
                # print(f"trainer.py - evaluate() - after pred = self.model, bdy is {bdy}\n")

                # 3) perceptual
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Use the same dtype as the model
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # Use the same dtype as the model
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
                # print(f"trainer.py - evaluate() - after pred = self.model, total_loss is {total_loss}, total_sample is {total_samples}\n")

        self.model.train()
        return total_loss / total_samples if total_samples > 0 else float("nan")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class LoRALogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records = []  # will hold one dict per (iter, module)

    def record_batch(self, model, accelerator, iteration: int):
        """
        Call once per optimizer.step(): captures all lora_* params in the UNet.
        iteration: your global iteration counter (e.g. batch_idx + epoch*len(dataloader))
        """
        unet = accelerator.unwrap_model(model).base_unet
        for full_name, param in unet.named_parameters():
            if not (param.requires_grad and ".lora_" in full_name):
                continue
            module_key = full_name.rsplit(".lora_",1)[0]
            w = param.detach().cpu().view(-1).float()
            w_mean, w_std = w.mean().item(), w.std().item()
            if param.grad is None:
                g_mean = float("nan")
                g_std  = float("nan")
            else:
                g = param.grad.detach().cpu().view(-1).float()
                g_mean, g_std = g.mean().item(), g.std().item()
            self._records.append({
                "iter":      iteration,
                "module":    module_key,
                "full_name": full_name,
                "W_mean":    w_mean,
                "W_std":     w_std,
                "G_mean":    g_mean,
                "G_std":     g_std,
            })

    def plot_up_to(self, iteration: int):
        """
        Re-compute the per-iteration aggregates up to “iteration” (inclusive)
        and overwrite the four summary plots on disk.
        """
        df = pd.DataFrame(self._records)
        df = df[df["iter"] <= iteration]

        grp = df.groupby("iter")
        summary = pd.DataFrame({
            "wmean_mean": grp["W_mean"].mean(),
            "wmean_std":  grp["W_mean"].std(),
            "wstd_mean":  grp["W_std"].mean(),
            "wstd_std":   grp["W_std"].std(),
            "gmean_mean": grp["G_mean"].mean(),
            "gmean_std":  grp["G_mean"].std(),
            "gstd_mean":  grp["G_std"].mean(),
            "gstd_std":   grp["G_std"].std(),
        })
        summary.index.name = "iter"
        summary.reset_index(inplace=True)

        plots = [
            (["wmean_mean","wmean_std"], "Weight‐Mean stats", "W_mean"),
            (["wstd_mean", "wstd_std"],  "Weight‐Std  stats", "W_std"),
            (["gmean_mean","gmean_std"], "Grad‐Mean   stats", "G_mean"),
            (["gstd_mean", "gstd_std"],  "Grad‐Std    stats", "G_std"),
        ]
        for i, (cols, title, ylabel) in enumerate(plots, 1):
            fig, ax = plt.subplots(figsize=(8,3))
            # ax.plot(summary["iter"], summary[cols[0]], label=cols[0])
            # ax.plot(summary["iter"], summary[cols[1]], label=cols[1])
            ax.plot(summary["iter"], summary[cols[0]], marker="o", label=cols[0])
            ax.plot(summary["iter"], summary[cols[1]], marker="x", label=cols[1])
            ax.set_title(f"{title} up to iter {iteration}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            out = self.output_dir / f"live_plot_{i}.png"
            fig.savefig(out)
            plt.close(fig)

    def finalize(self):
        """Write CSV + four summary plots to disk."""
        df = pd.DataFrame(self._records)
        # 1) save raw history
        csv_path = self.output_dir / "lora_full_history.csv"
        df.to_csv(csv_path, index=False)
        print(f"[LoRALogger] raw history → {csv_path}")

        # 2) compute per-iteration summary
        grp = df.groupby("iter")
        summary = pd.DataFrame({
            "wmean_mean": grp["W_mean"].mean(),
            "wmean_std":  grp["W_mean"].std(),
            "wstd_mean":  grp["W_std"].mean(),
            "wstd_std":   grp["W_std"].std(),
            "gmean_mean": grp["G_mean"].mean(),
            "gmean_std":  grp["G_mean"].std(),
            "gstd_mean":  grp["G_std"].mean(),
            "gstd_std":   grp["G_std"].std(),
        })
        summary.index.name = "iter"
        summary.reset_index(inplace=True)
        summary_csv = self.output_dir / "lora_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"[LoRALogger] summary → {summary_csv}")

        # 3) four plots
        plots = [
            (["wmean_mean", "wmean_std"], "Weight‐Mean: mean & std over modules", "W_mean stats"),
            (["wstd_mean",  "wstd_std"],  "Weight‐Std: mean & std over modules", "W_std stats"),
            (["gmean_mean", "gmean_std"], "Grad‐Mean: mean & std over modules", "G_mean stats"),
            (["gstd_mean",  "gstd_std"],  "Grad‐Std: mean & std over modules", "G_std stats"),
        ]
        for i, (cols, title, ylabel) in enumerate(plots, start=1):
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(summary["iter"], summary[cols[0]], label=cols[0])
            ax.plot(summary["iter"], summary[cols[1]], label=cols[1])
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            out_png = self.output_dir / f"plot_{i}.png"
            fig.savefig(out_png)
            print(f"[LoRALogger] saved {out_png}")
            plt.close(fig)


def log_batch_memory(batch_indx, batch, stage="BEFORE"):
    batch_size = batch["latent"].size(0)
    print(f"\n*** - trainer.py - cubedifftrainer - train loop - Batch {batch_indx}: batch size is {batch_size}")
    print(f"latent shape: {batch['latent'].shape}")

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    cpu_mem = psutil.virtual_memory().used / (1024 ** 3)

    print(f"[{stage}] CUDA memory allocated: {allocated:.2f} GB")
    print(f"[{stage}] CUDA memory reserved:  {reserved:.2f} GB")
    print(f"[{stage}] CPU RAM used: {cpu_mem:.2f} GB")

# EOF -------------------------------------------------------------------------
