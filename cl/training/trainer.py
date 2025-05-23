# -------------  cl/training/trainer.py  --------------------
#─────────────────────────────────────────────────────────────────────────────
# Monkey-patch for PyTorch < 2.2, providing get_default_device() so
# diffusers/transformers pipelines don’t crash on torch.get_default_device()
import torch
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: ("cuda" if torch.cuda.is_available() else "cpu")
#─────────────────────────────────────────────────────────────────────────────

# import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

torch.cuda.empty_cache()

import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")   # <— avoid /dev/shm exhaustion on many workers

import os

# Update LD_LIBRARY_PATH to include where libcuda.so actually is
os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
# Set torch compile backend
os.environ["TORCH_COMPILE_BACKEND"] = "inductor"
# force‐load the real driver

# disable Dynamo/inductor checks
import types
import math

# Before running your training script
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"

import datetime, time, types
from pathlib import Path
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torchvision import models
import bitsandbytes as bnb

# from peft import get_peft_model, LoraConfig #, prepare_model_for_kbit_training

import wandb, psutil

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DDPMScheduler

from cl.data.latent_webdataset import get_dataloader
from cl.model.architecture   import CubeDiffModel

from diffusers import UNet2DConditionModel
from transformers import get_cosine_schedule_with_warmup, get_wsd_schedule # (2025-5-18 this made LR be exactly 0 after 29 steps, why ?)
import tarfile

# -----------------------------------------------------------
# — Monkey-patch UNet2DConditionModel.forward to swallow any extra kwargs —
orig_unet_forward = UNet2DConditionModel.forward
def _patched_unet_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
    """
    Accept the three required args, ignore anything else.
    Guards against 'decoder_input_ids', 'use_cache', etc.
    """
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

        # 1) Create a DDP kwargs handler that turns on unused-parameter detection
        # ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(mixed_precision=mixed_precision,
                                       gradient_accumulation_steps=gradient_accumulation_steps,
                                    #    kwargs_handlers=[ddp_handler], 
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
        print(f"trainer.py - CubeDiffTrainer - init - setup_model {pretrained_model_name} done\n")
        
        from torchvision.models import VGG16_Weights

        def safe_vgg16(pretrained=True, **kwargs):
            # 1) try loading from local torchvision cache
            try:
                # new-style API will look in ~/.cache/torch/hub/checkpoints first
                return models.vgg16(weights=VGG16_Weights.DEFAULT, **kwargs)
            except Exception:
                # 2) fallback to original pretrained=True (which may download)
                return models.vgg16(pretrained=pretrained, **kwargs)

        self.perceptual_net = safe_vgg16().features[:16].eval().to(torch.float32)


        for p in self.perceptual_net.parameters():
            p.requires_grad = False
        self.l1 = torch.nn.L1Loss()
        print(f"trainer.py - CubeDiffTrainer - init - vgg16 loading done")
        self.global_iter = 0
    
    def setup_model(self, pretrained_model_name: str):

        def safe_pipeline_from_pretrained(repo_id, **kwargs):
            # 1) try *only* local cache
            try:
                return StableDiffusionPipeline.from_pretrained(
                    repo_id,
                    local_files_only=True,
                    **kwargs
                )
            except (OSError, ValueError):
                # 2) fallback to the normal download→cache path
                return StableDiffusionPipeline.from_pretrained(repo_id, **kwargs)

        pipe = safe_pipeline_from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        print(f"trainer.py - CubeDiffTrainer - setup_model - StableDiffusionPipeline model ({pretrained_model_name}) loaded\n")
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
        self.model = CubeDiffModel(
            pretrained_model_name,
            skip_weight_copy=self.config["skip_weight_copy"]
        )
        
        # That will reproduce the CubeDiff authors’ ∼17 M trainable parameters and ensure meaningful gradients.
        #  enable full-rank tuning on only the inflated-attn layers
        for name, p in self.model.base_unet.named_parameters():
            # keep only the Q/K/V/O projections trainable
            # Only floating-point Tensors can track gradients.
            if p.dtype.is_floating_point and any(seg in name for seg in ("to_q", "to_k", "to_v", "to_out")):
                p.requires_grad = True
            else:
                p.requires_grad = False
        print(f"trainer.py - CubeDiffTrainer - setup_model - UNet model parameters set to requires_grad for full rank tunning\n")
        # quick sanity check
        total, trainable = 0, 0
        for p in self.model.base_unet.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"👉 Full-rank tuning: {trainable/1e6:.2f}M / {total/1e6:.1f}M params")

        self.text_encoder = self.text_encoder.to(dtype=self.model_dtype)
        self.vae = self.vae.to(dtype=self.model_dtype)

        # Cast scheduler tensors
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(self.model_dtype))

        print(f"trainer.py - CubeDiffTrainer- CubeDiff Model components cast to {self.model_dtype}")

        # Enable gradient checkpoints  on the U-Net backbone only and circular padding
        # if diffusers>=0.18, which shards attention internals to slash peak usage.
        #    (saves ~30–40% memory at the cost of ~10–20% extra compute)
        print(f"trainer.py - CubeDiffTrainer - CubeDiff Model enabled gradient checkpointing\n")
        self.model.base_unet.enable_gradient_checkpointing()
        
        print(f"trainer.py - CubeDiffTrainer - CubeDiff Model enabled xformers\n")
        self.model.base_unet.enable_xformers_memory_efficient_attention()
        for m in self.model.base_unet.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.padding_mode = "circular"

        tot   = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"trainer.py - CubeDiffTrainer - setup_model done - Total params {tot/1e6:.1f}M — LoRA trainable {train/1e6:.2f}M")

    # --------------------------------------------------
    # New dataloader creator  (latents, no JPEG) 
    # --------------------------------------------------    
    def build_dataloaders(self):
        print(f"trainer.py - CubeDiffTrainer - Building dataloaders with config: {self.config}")

        # ─── helper to count samples in a .tar ──────────────────────────
        def count_tar_samples(path: str) -> int:
            cnt = 0
            with tarfile.open(path, "r") as tar:
                for m in tar.getmembers():
                    # assume each .pt is one sample
                    if m.name.endswith(".pt"):
                        cnt += 1
            return cnt
        
        try:
            self.train_size = count_tar_samples(self.config["dataset"])
            print(f"  ▶ Train samples: {self.train_size}")
            self.train_dataloader = get_dataloader(
                wds_path=self.config["dataset"],
                batch_size=self.config["batch_size"],
                data_size=self.train_size,
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
                self.val_size = count_tar_samples(self.config["val_dataset"])
                print(f"  ▶ Val   samples: {self.val_size}")
                self.val_dataloader = get_dataloader(
                    self.config["val_dataset"],  
                    batch_size=self.config["batch_size"],
                    data_size=self.val_size,
                    num_workers=self.config["num_workers"] 
                )
                print("Val dataloader created successfully")
            else:
                self.val_dataloader = None

            # Set random seed
            set_seed(self.config.get("seed", 42))

            print("Preparing model and dataloader with accelerator")
            self.model, self.train_dataloader = self.accelerator.prepare(
                self.model, self.train_dataloader
            )
            
            if "val_dataset" in self.config:
                _, self.val_dataloader = self.accelerator.prepare(
                    self.model, self.val_dataloader
                )
            
            # Move components to device (preserve FP32 for perceptual_net)
            # U-Net, VAE, and text encoder benefit from BF16 for speed/memory.
            # Move components to device (preserve FP32 for perceptual_net)
            device = self.accelerator.device
            print(f"Moving components to {device}; keeping perceptual_net in FP32")
            self.perceptual_net = self.perceptual_net.to(device)                  # keep FP32
            self.text_encoder   = self.text_encoder.to(device, dtype=self.model_dtype)
            self.vae            = self.vae.to(device, dtype=self.model_dtype)
            
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
            B, Face, C, H, W = x.shape
            x5 = x
        elif x.dim() == 4:
            Bf, C, H, W = x.shape
            Face = self.config.get("num_faces", 6)
            B = Bf // Face
            x5 = x.view(B, Face, C, H, W)
        else:
            raise ValueError(f"boundary_loss: unexpected tensor dim {x.dim()}")

        losses = []
        for i in range(Face):
            # right edge of face i vs left edge of face (i+1)%F
            r = x5[:, i, :, :, -1]      # [B, C, H]
            l = x5[:, (i+1) % Face, :, :,  0]  # [B, C, H]
            losses.append(self.l1(r, l))
        return sum(losses) / Face
    
    # --------------------------------------------------
    #  Training loop  (shortened & adapted to latent input)
    # --------------------------------------------------
    def train(self):
        self.build_dataloaders()
        num_epochs = self.config.get("num_epochs", 1)
        world_size      = self.accelerator.num_processes
        samples_per_rank = self.train_size // world_size
        total_samples   = self.train_size * num_epochs
        batch_size      = self.config["batch_size"]
        batch_num_per_rank = samples_per_rank // batch_size
        self.epsion_mse_warmup = num_epochs * 0.1

        print(f"trainer.py - CubeDiffTrainer - train data - self.train_size is {self.train_size}, world_size is {world_size}, samples_per_rank is {samples_per_rank}, num_epochs is {num_epochs}, total_samples is {total_samples}, batch_size is {batch_size}, batch_num_per_rank is {batch_num_per_rank}\n")
        # only update the small LoRA adapter params
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        # ← switch from AdamW to 8-bit Adam for much smaller optimizer state
        self.optimizer = bnb.optim.Adam8bit(
            lora_params,
            lr=self.config["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.01  # Add weight decay for regularization
        )
        print(f"trainer.py - CubeDiffTrainer - train - optimizer Adam8bit created\n")

        # the LR will decay smoothly over exactly the number of updates you actually perform, 
        # instead of “waiting” through 700 steps that never happen
        train_size    = self.train_size                          # e.g. 651
        world_size    = self.accelerator.num_processes           # 8
        # bs_per_gpu    = self.config["batch_size"]                # 2
        accum_steps   = self.config["gradient_accum_steps"]      # 4
        
        # LR schedule -------------------------------------
        # 2) local per–GPU batch size
        local_batch_size = self.train_dataloader.batch_size # 2

        # → global batch size per update
        global_batch = local_batch_size * world_size * accum_steps # = 64 

        # → how many updates (optimizer.step calls) per epoch
        self.updates_per_epoch = math.ceil(train_size / global_batch) # ≈11 = 651/64
        true_steps   = math.ceil(train_size * self.config["num_epochs"] / global_batch) # 220

        # then override your config:
        from torch.optim.lr_scheduler import CosineAnnealingLR # SequentialLR, LinearLR, ConstantLR, 
        self.lr_scheduler  = CosineAnnealingLR(self.optimizer, T_max=true_steps)

        # → total training updates over all epochs
        self.total_updates = self.updates_per_epoch * self.config["num_epochs"] # ≈220 = 11*20

        # compute the boundary warm-up length exactly as the scheduler
        boundary_warmup = max(1, int(self.total_updates * 0.1))  # 10% of total updates

        print(f"Scheduler: local_batch_size ={local_batch_size}, global_batch = {global_batch}, updates_per_epoch  = {self.updates_per_epoch}, true_steps  = {true_steps}")

        # ─── DEBUG: print out the very first 50 LR values ───
        if self.accelerator.is_main_process:
            import copy
            tmp = copy.deepcopy(self.lr_scheduler)
            print(f"debug LR - step 0: {tmp.get_last_lr()[0]:.3e}")  # before any .step()

            # print learning rates for all steps 
            for i in range(1, true_steps + 1):
                tmp.step()
                # if i in check:
                print(f"step {i:>3}: {tmp.get_last_lr()[0]:.3e}")
        # ------------------------------------------------
        
        # print(f"trainer.py - CubeDiffTrainer - train - lr_scheduler created - self.total_updates {self.total_updates}, num_warmup_steps {num_warmup_steps} - stable_steps {stable_steps} - decay_steps {decay_steps}\n")
        # now wrap both optimizer and scheduler exactly once
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer,
            self.lr_scheduler
        )        

        sample_prompts        = ["A beautiful mountain lake at sunset with snow-capped peaks"]
        print(f"trainer.py - CubeDiffTrainer - train - sample_prompts is {sample_prompts}\n")

        eval_every_n_samples  = self.config.get("eval_every_n_samples", 100)
        processed_samples = 0
        next_eval_at      = eval_every_n_samples
        gstep             = 0
        train_losses, val_losses = [], []
        g_start_tm = time.time()
        ep_g_start_tm = time.time()
        
        rank = self.accelerator.local_process_index  
        for epoch in range(num_epochs):
            epoch_processed = 0
            print(f"▶️ Starting epoch {epoch}/{num_epochs}")
            for batch_indx, batch in enumerate(self.train_dataloader):
                real_batch_size = batch["latent"].size(0)
                print(f"\t*** rank {rank} - epoch {epoch} - train loop - Batch {batch_indx}: real_batch_size is {real_batch_size}\n")
                # how many samples this GPU just got?
                micro_bs = batch["latent"].size(0)
                print(
                    f"\t[Epoch {epoch}/{num_epochs}] "
                    f"\tRank {rank} | GPU {self.accelerator.device} "
                    f"\tmicro-batch #{batch_indx}: {micro_bs} samples"
                )
                
                start_tm = time.time()
                with self.accelerator.accumulate(self.model):
                    lat = batch["latent"].to(self.accelerator.device, dtype=self.model_dtype)              # [B,6,4,64,64]
                    ids = batch["input_ids"].to(self.accelerator.device)
                    mask = batch["attention_mask"].to(self.accelerator.device)
                    
                    print(f"trainer.py - train() - after batch = accelerator.accumulate, lat shape is {lat.shape} and dtype is {lat.dtype}, ids shape is {ids.shape} and dtype is {ids.dtype}, mask shape is {mask.shape} and dtype is {mask.dtype}\n")
                    
                    noise     = torch.randn_like(lat)
                    timesteps = torch.randint(0,
                                            self.noise_scheduler.config.num_train_timesteps,
                                            (lat.shape[0],), device=lat.device)
                    noisy_lat = self.noise_scheduler.add_noise(lat, noise, timesteps)
                    print(f"trainer.py - train() - after batch = accelerator.accumulate, noisy_lat shape is {noisy_lat.shape} and dtype is {noisy_lat.dtype}\n")

                    with torch.no_grad():
                        txt_emb = self.text_encoder(ids, attention_mask=mask).last_hidden_state
                        # ← CAST TO bfloat16 so K/V come out in bf16
                        txt_emb = txt_emb.to(self.accelerator.device, dtype=self.model_dtype)
                        del ids  # Free memory as soon as possible
                    # print(f"trainer.py - train() - before pred = self.model, lat shape is {lat.shape} and dtype is {lat.dtype}, noisy_lat shape is {noisy_lat.shape} type is {noisy_lat.dtype}, timesteps is {timesteps}, txt_emb shape is {txt_emb.shape}, type is {txt_emb.dtype}\n")

                    # CubeDiff requires randomly dropping each modality 10% of the time during training so the model learns text-only, image-only, 
                    # and joint modes. Without this, it overfits to always having both conditions and fails to generalize.
                    # each micro-batch randomly drops text or image conditioning at 10 % each
                    # — Classifier-Free Guidance (CFG) drops (§4.5):
                    #   10% drop text, 10% drop image, 80% full cond
                    bs = txt_emb.size(0)
                    rnd = torch.rand(bs, device=txt_emb.device)
                    # drop text embeddings
                    drop_txt = rnd < 0.1
                    if drop_txt.any():
                        txt_emb[drop_txt] = 0
                    # drop image conditioning mask
                    drop_img = (rnd >= 0.1) & (rnd < 0.2)
                    if drop_img.any():
                        mask[drop_img] = 0

                    with self.accelerator.autocast():
                        pred = self.model(
                            latents=noisy_lat,
                            timesteps=timesteps,
                            encoder_hidden_states=txt_emb
                        )

                    # print(f"trainer.py - train() - after pred = self.model")

                    # ── NEW LOSS ──
                    if noise.size(2) != pred.size(2):
                        noise = noise[:, :, : pred.size(2), :, :]
                    
                    # epsilon mse ------------------------------------------------------------
                    mse_loss = F.mse_loss(pred.float(), noise.float()).mean()
                    if rank==0:
                        print(f"rank {rank} - epoch {epoch} - gstep is {gstep} - epsion_mse_warmup is {self.epsion_mse_warmup}, mse_loss (mse_epsilon) is {mse_loss.item()}")
                    
                    loss = mse_loss.float()  / self.config["gradient_accum_steps"]
                    
                    if rank==0:
                        # print(f"check-loss-terms - rank {rank} - epoch {epoch} - gstep is {gstep} - global_iter is {self.global_iter} - total loss is {loss.item()}, mse_loss is {mse_loss.item()}, bdy is {bdy.item()}, perc is {perc.item()}")
                        print(f"check-loss-terms - rank {rank} - epoch {epoch} - gstep is {gstep} - global_iter is {self.global_iter} - total loss is {loss.item()}, mse_loss is {mse_loss.item()}")
                    
                    # collect loss                    
                    self.accelerator.backward(loss)  # ← compute gradients

                    #  collect total steps
                    processed_samples += real_batch_size
                    local_count = torch.tensor(processed_samples, device=self.accelerator.device)
                    total_processed_samples = self.accelerator.reduce(local_count, reduction="sum")
                
                    # 2) Only once per actual optimizer step (i.e. when sync_gradients=True):
                    if self.accelerator.sync_gradients:
                        # here all replicas have the fully synchronized, accumulated grads
                        # see each adapter’s true gradient (after accumulation + all‐GPU sync) exactly once per update, 
                        # rather than on every micro‐batch.
                        print(f"\t\tRank {rank} → Update done (sync_gradients={self.accelerator.sync_gradients}) for epoch {epoch}, global_iter {self.global_iter}")
                        
                        # compute global average loss per sample 
                        # 1) form weighted‐sum and count tensors
                        loss_sum   = loss.detach() * batch_size
                        loss_count      = torch.tensor(batch_size, device=self.accelerator.device)

                        # 2) all‐reduce both to get global sums
                        loss_sum   = self.accelerator.reduce(loss_sum,   reduction="sum")
                        loss_count      = self.accelerator.reduce(loss_count,      reduction="sum")

                        # 3) only main process turns sums into the global average
                        if self.accelerator.is_main_process:
                            avg_train_loss = (loss_sum / loss_count).item()
                            total_processed_samples = total_processed_samples.item()
                            train_losses.append((total_processed_samples, avg_train_loss))
                            print(f"\t\tRank {rank} → Update done (sync_gradients={self.accelerator.sync_gradients}) for epoch {epoch}, global_iter {self.global_iter}, total_processed_samples is {total_processed_samples}")
                        
                        # 4) Clip gradients
                        # This prevents any single batch from sending the loss curve on a wild ride.
                        # Gradient clipping (1.0) is standard in diffusion training to prevent occasional large updates, smoothing out your “bounces” .
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 0.5)

                        # 5) Apply optimizer update
                        self.optimizer.step()

                        # 6) Step the LR scheduler (once per real update)
                        self.lr_scheduler.step()

                        # 2d) Clear gradients for the next accumulation
                        # zero gradients (optional - can be after or before step)
                        self.optimizer.zero_grad()

                         # 2e) Increment per‐rank step counter
                        gstep += 1
                        
                        #–– print out the LR for the first few updates ––#
                        if self.accelerator.is_main_process:
                            lr = self.lr_scheduler.get_last_lr()[0]
                            print(f"rank {rank}, LR at step {gstep}: {lr:.3e}")

                    # print(f"\t\tRank {rank} - train - after self.accelerator.sync_gradients - epoch {epoch} - self.global_iter is {self.global_iter}")
                    self.global_iter += 1

                end_tm = time.time()
                print(f"\tRank {rank} - epoch {epoch} - out of accumulate - batch_indx {batch_indx} cost {end_tm - start_tm:.2f} seconds")
    
                # ---- logging & sampling ----------------              
                epoch_processed += real_batch_size                
                pct_epoch = epoch_processed / samples_per_rank * 100
                pct_total = (processed_samples * world_size) / total_samples * 100

                # collect loss
                if self.accelerator.is_main_process:
                    print(f"rank {rank} - Epoch {epoch}/{num_epochs} ▶ {pct_epoch:.1f}% | Overall ▶ {pct_total:.1f}%")
                
                prev_evaL_at = next_eval_at
                if (processed_samples >= next_eval_at) or (batch_indx == batch_num_per_rank - 1):
                    # 1) every rank hits this barrier
                    print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - gstep {gstep} - waiting at barrier for eval - self.accelerator.is_main_process is {self.accelerator.is_main_process}, before self.accelerator.wait_for_everyone() - processed_samples is {processed_samples}, eval_every_n_samples is {eval_every_n_samples}, next_eval_at is {next_eval_at}")
                    temp_s_time = time.time()
                    self.accelerator.wait_for_everyone()                    
                    temp_e_time = time.time()
                    print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - gstep {gstep} - self.accelerator.is_main_process is {self.accelerator.is_main_process}, after self.accelerator.wait_for_everyone() - cost {temp_e_time - temp_s_time:.2f} seconds")
                
                    if eval_every_n_samples:
                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - eval_every_n_samples is {eval_every_n_samples} - evaluating ...")
                        temp_s_time = time.time()
                        val_loss = self.evaluate(rank, gstep, boundary_warmup) # this cost some time
                        temp_e_time = time.time()
                        if self.accelerator.is_main_process:
                            val_losses.append((total_processed_samples, val_loss))
                            print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - → val‐loss @ {processed_samples} samples: {val_loss:.4f}, eval_every_n_samples is {eval_every_n_samples} - evaluation done - cost {temp_e_time - temp_s_time:.2f} seconds")
                    next_eval_at += eval_every_n_samples                    
                
                # only rank 0 actually runs eval & sampling
                # update sample count on the main rank (only it tracks & saves losses)
                if self.accelerator.is_main_process:                    
                    # 1) log train loss every N steps
                    if gstep % self.config["log_loss_every_n_steps"] == 0:
                        self._plot_loss_curves(total_processed_samples, train_losses, val_losses, gstep, self.total_updates)
                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - plotted loss - gstep {gstep} - self.accelerator.is_main_process is {self.accelerator.is_main_process} - log_loss_every_n_steps is {self.config['log_loss_every_n_steps']} - global_iter is {self.global_iter-1} - batch_size is {batch_size}, samples {processed_samples:>4}  train-loss {loss.item():.4f}")
                    # 2) every `eval_every_n_samples`, first sync _all_ ranks…
                    #  eval + sample‐gen
                    if (eval_every_n_samples and processed_samples >= prev_evaL_at) or (batch_indx == batch_num_per_rank - 1):
                        # generate panorama from current checkpoints 
                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - generate_samples ...")
                        temp_s_time = time.time()
                        self.generate_samples(rank, sample_prompts, total_processed_samples)
                        temp_e_time = time.time()
                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - generate_samples done - cost {temp_e_time - temp_s_time:.2f} seconds")    
                        
                # gstep += 1
            g_end_tm = time.time()
            # ----------------- end of training loop ----------------------
            print(f"Rank {rank} - epoch {epoch} - batch_indx {batch_indx} - out of the training loop - all steps done, gstep is {gstep}, self.global_iter is {self.global_iter}, cost {g_end_tm - g_start_tm:.2f} seconds\n")
        
        ep_g_end_tm = time.time()
        print(f"Rank {rank} - epoch {epoch} - out of the epoch training loop - all epochs done, gstep is {gstep}, self.global_iter is {self.global_iter}, cost {ep_g_end_tm - ep_g_start_tm:.2f} seconds\n")
        # ----------------- save final LoRA ----------------------
        if self.accelerator.is_main_process:
            path = self.output_dir / f"adapter_model.bin"
            # pull the real underlying model out of the Accelerator wrapper
            unwrapped = self.accelerator.unwrap_model(self.model)
            # grab just the U-Net’s weights
            unet_sd = unwrapped.base_unet.state_dict()
            torch.save(unet_sd, path)
            print(f"\nRank {rank} - ✔ saved U-Net adapter to {path}")
            # self.lora_logger.finalize()
            print(f"Rank {rank} - trainer.py - CubeDiffTrainer - train - lora final logging done")

            # ───────────────────────────────────────────────────────────────
            # after all training, plot train & val curves
            try:
                # unpack
                steps_tr, loss_tr = zip(*train_losses) if train_losses else ([],[])
                steps_va, loss_va = zip(*val_losses)   if val_losses   else ([],[])
                # coerce EVERY element to a host‐side Python scalar
                steps_tr = [
                    int(s.item()) if torch.is_tensor(s) else int(s)
                    for s in steps_tr
                ]
                loss_tr  = [
                    float(l.item()) if torch.is_tensor(l) else float(l)
                    for l in loss_tr
                ]
                steps_va = [
                    int(s.item()) if torch.is_tensor(s) else int(s)
                    for s in steps_va
                ]
                loss_va  = [
                    float(l.item()) if torch.is_tensor(l) else float(l)
                    for l in loss_va
                ]

                plt.figure(figsize=(6,4))
                plt.plot(steps_tr, loss_tr, label="train")
                if steps_va:
                    plt.plot(steps_va, loss_va, label="val")
                plt.xlabel("step")
                plt.ylabel("MSE loss")
                plt.legend()
                plt.tight_layout()
                out = self.output_dir / f"loss_curve.png"
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
            B, Face, C, H, W = lat.shape
            lat = lat.reshape(B * Face, C, H, W)

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
    def generate_samples(self, rank, prompts, step):
        print(f"\trank {rank} - generate_samples - prompts is {prompts}, step is {step}")
        from cl.inference.pipeline import CubeDiffPipeline
        if not self.accelerator.is_main_process: 
            return
        
        tmp_ckpt = self.output_dir / f"tmp_{step}.bin"

        # pull the *underlying* model off of the accelerator
        unwrapped = self.accelerator.unwrap_model(self.model)
        # print(f"\t\trank {rank} - generate_samples - got unwrapped")

        # get just the UNet weights
        unet_sd   = unwrapped.base_unet.state_dict()
        torch.save(unet_sd, tmp_ckpt)
        print(f"\t\trank {rank} - generate_samples - unet_sd tmp_ckpt saved at {tmp_ckpt}")

        # 1) instantiate pipeline normally
        pipe = CubeDiffPipeline(
            pretrained_model_name="runwayml/stable-diffusion-v1-5"
        )
        print(f"\t\trank {rank} - generate_samples - got pipeline for runwayml/stable-diffusion-v1-5")

        # 2) read your U-Net weights back
        adapter_state = torch.load(tmp_ckpt, map_location="cuda")
        print(f"\t\trank {rank} - generate_samples - got adapter_state from tmp_ckpt {tmp_ckpt}")

        # 3) cast them to the same dtype as the pipeline’s U-Net
        unet_dtype = next(pipe.model.base_unet.parameters()).dtype
        for k, v in adapter_state.items():
            adapter_state[k] = v.to(unet_dtype)
        
        # 4) load *only* into the U-Net
        missing, unexpected = pipe.model.base_unet.load_state_dict(adapter_state, strict=False)
        print(f"\t\trank {rank} - generate_samples - loaded CubeDiffPipeline.model.base_unet from UNet adapter, missing block/component size: ", len(missing))

        print(f"\t\trank {rank} - generate_samples -  create inference pipe and generate the pnoaram based on the given prompts")
        for i,p in enumerate(prompts):
            # why the generated panorama is black ?!
            pano = pipe.generate(p, num_inference_steps=30, guidance_scale=7.5)
            temp = self.images_dir / f"step{step}_{i}.jpg"
            pano.save(self.images_dir / f"step{step}_{i}.jpg")
            print(f"\trank {rank} - generate_samples - prompt is {p}, saved pano at {temp}")

        tmp_ckpt.unlink()
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
        print(f"\t\trank {rank} - generate_samples - gstep is {step},  unet tmp_ckpt {tmp_ckpt} was removed")

    # ---------------------------------------------------------------------
    # New: run a full pass over val_dataloader and return avg loss.
    # reuse boundary_loss helper inside train()
    # and decode_latents_to_rgb + self.perceptual_net    
    
    def evaluate(self, rank, gstep, boundary_warmup):
        """Compute avg. (MSE + boundary + perceptual) loss on the validation set."""
        if self.val_dataloader is None:
            return float("nan")

        import torch._dynamo
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True

        self.model.eval()
        total_loss, total_samples = 0.0, 0

        # Set a maximum number of batches to evaluate to prevent infinite loops
        max_val_batches = (self.val_size + self.config["batch_size"] - 1) // self.config["batch_size"]
        processed_batches = 0
        print(f"\trank {rank} - trainer.py - evaluate() - self.val_size is {self.val_size} - max_val_batches is {max_val_batches}\n")
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Add batch counter for safety
                processed_batches += 1
                if processed_batches > max_val_batches:
                    print(f"Warning: Processed {processed_batches} validation batches, expected only {max_val_batches}. Breaking out of loop.")
                    break
                    
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
                # print(f"trainer.py - evaluate() - after pred = self.model, mse is {mse}\n")
                # The model predicts only the 4 “true” latent channels;
                # our `noise` still has 5 (4 + mask). Drop the mask so shapes align.
                if noise.size(2) != pred.size(2):
                    noise = noise[:, :, : pred.size(2), :, :]                

                # epsilon mse loss 
                mse_loss = F.mse_loss(pred.float(), noise.float()).mean()
                loss = mse_loss.float() #  / self.config["gradient_accum_steps"]
                total_loss   += loss.item() * lat.size(0)
                total_samples+= lat.size(0)

        # 1) make local-sum & count tensors
        device       = self.accelerator.device
        loss_sum     = torch.tensor(total_loss,   device=device)
        loss_count   = torch.tensor(total_samples,device=device)

        # 2) all‐reduce sums
        loss_sum     = self.accelerator.reduce(loss_sum,   reduction="sum")
        loss_count   = self.accelerator.reduce(loss_count, reduction="sum")

        # 3) only the main process turns into a scalar avg
        if self.accelerator.is_main_process:
            avg = (loss_sum / loss_count).item()
            print(f"\trank {rank} - evaluate() - avg loss is {avg}, count_sum is {loss_count}\n")
        else:
            avg = None

        self.model.train()        
        return avg

    def _plot_loss_curves(self,
        total_processed_samples: int,
        train_losses, val_losses,
        gstep: int,
        total_updates: int):
        """
        train_losses: list of (samples, loss)
        val_losses:   list of (samples, loss)
        gstep:        current number of update steps done so far
        total_updates: the total number of update steps you plan to run
        """
        # unpack
        steps_tr, loss_tr = zip(*train_losses) if train_losses else ([],[])
        steps_va, loss_va = zip(*val_losses)   if val_losses   else ([],[])

        # coerce EVERY element to a host‐side Python scalar
        steps_tr = [
            int(s.item()) if torch.is_tensor(s) else int(s)
            for s in steps_tr
        ]
        loss_tr  = [
            float(l.item()) if torch.is_tensor(l) else float(l)
            for l in loss_tr
        ]
        steps_va = [
            int(s.item()) if torch.is_tensor(s) else int(s)
            for s in steps_va
        ]
        loss_va  = [
            float(l.item()) if torch.is_tensor(l) else float(l)
            for l in loss_va
        ]

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(steps_tr, loss_tr, label="train")
        if steps_va:
            ax.plot(steps_va, loss_va, label="val")

        ax.set_title(
            f"Loss up to {total_processed_samples} samples\n"
            f"(update steps so far: {gstep}/{total_updates})"
        )
        ax.set_xlabel("samples processed")
        ax.set_ylabel("loss")
        ax.legend()

        # ─── mark the current gstep on the samples axis ───
        if 1 <= gstep <= len(steps_tr):
            sample_at_gstep = steps_tr[gstep-1]
            # vertical line at that sample
            ax.axvline(sample_at_gstep, color="black", linestyle="--", alpha=0.6)
        else:
            sample_at_gstep = None

        # ─── now add a twin‐x to label the gstep ───
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        if sample_at_gstep is not None:
            ax2.set_xticks([sample_at_gstep])
            ax2.set_xticklabels([f"gstep={gstep}"])
        else:
            # fallback: just show start and end
            ax2.set_xticks([ax.get_xlim()[0], ax.get_xlim()[1]])
            ax2.set_xticklabels([f"gstep=0", f"gstep={total_updates}"])
        ax2.set_xlabel("update steps")

        plt.tight_layout()
        out = self.output_dir / f"loss_curve_up_to_total_processed_samples.png"
        plt.savefig(out)
        plt.close(fig)
        print(f"✔ saved loss curve → {out}")

# EOF -------------------------------------------------------------------------
