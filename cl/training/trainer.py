# -------------  cl/training/trainer.py  --------------------
import torch
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import bitsandbytes as bnb

# from peft import get_peft_model, LoraConfig #, prepare_model_for_kbit_training

import wandb, psutil

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DeepSpeedPlugin
from diffusers import StableDiffusionPipeline, DDPMScheduler

from cl.data.latent_webdataset import get_dataloader
from cl.model.architecture   import CubeDiffModel

from diffusers import UNet2DConditionModel
from transformers import get_linear_schedule_with_warmup
# from peft import TaskType
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

        self.accelerator = Accelerator(mixed_precision=mixed_precision,
                                       gradient_accumulation_steps=gradient_accumulation_steps,
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
        # Without enough weight on seam and perceptual terms, the model ignores overlap learning; and running VGG in bf16 loses detail.
        # The VGG16 perceptual network, however, must stay in full precision to provide stable gradient signals for texture and seam consistency
        # run VGG in full-precision for stable texture/perceptual gradients
        # it is faithful to CubeDiff’s paper (§3.2–3.4) and its ablations (A.11), 
        # and they mesh with industry best practices in multi-view diffusion and low-rank adaptation.
        self.perceptual_net = models.vgg16(pretrained=True).features[:16].eval().to(torch.float32)
        for p in self.perceptual_net.parameters():
            p.requires_grad = False
        self.l1 = torch.nn.L1Loss()

        # self.lora_logger = LoRALogger(
        #     output_dir=Path(self.config["output_dir"]) / "lora_logs"
        # )
        self.global_iter = 0
    
    # --------------------------------------------------
    #  Model & tokenizer (unchanged except for LoRA hook)
    # --------------------------------------------------
        
    def setup_model(self, pretrained_model_name: str):
        # Note: This import should be done only once
        # Import the patching module - this applies the patches automatically
        # filter out unneeded arguments for forward() of peft
        # Note: This import should be done only once
        # try:
        #     print(f"trainer.py - CubeDiffTrainer - setup - importing peft_patch")
        #     from cl.training import peft_patch  # This is the revised_peft_patch.py you created
        # except ImportError:
        #     print("⚠️ peft_patch module not found, continuing without patching")
            
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
        self.model = CubeDiffModel(
            pretrained_model_name,
            skip_weight_copy=self.config["skip_weight_copy"]
        )
        # ─── Move the *entire* CubeDiffModel onto GPU ───
        # This ensures positional_encoding, attention layers, etc. all live on self.device
        # self.model = self.model.to(self.device)

        # That will reproduce the CubeDiff authors’ ∼17 M trainable parameters and ensure meaningful gradients.
        # — 3.A: drop LoRA; enable full-rank tuning on only the inflated-attn layers
        for name, p in self.model.base_unet.named_parameters():
            # keep only the Q/K/V/O projections trainable
            # UNet was loaded in 4-bit quantized form (int4/int8 tensors), and you then try to call p.requires_grad = True on those integer-typed weights. 
            # Only floating-point Tensors can track gradients.
            if p.dtype.is_floating_point and any(seg in name for seg in ("to_q", "to_k", "to_v", "to_out")):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # quick sanity check
        total, trainable = 0, 0
        for p in self.model.base_unet.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"👉 Full-rank tuning: {trainable/1e6:.2f}M / {total/1e6:.1f}M params")

        # -------------------------------------------------------
        # 4) Inject LoRA *only* into the UNet backbone
        # unet = self.model.base_unet

        # # stub prepare_inputs_for_generation(sample, timestep, encoder_hidden_states)
        # if not hasattr(unet, "prepare_inputs_for_generation"):
        #     def _prepare_inputs_for_generation(self, sample, timestep, encoder_hidden_states, **kwargs):
        #         return {
        #             "sample": sample,
        #             "timestep": timestep,
        #             "encoder_hidden_states": encoder_hidden_states
        #         }
        #     unet.prepare_inputs_for_generation = types.MethodType(
        #         _prepare_inputs_for_generation, unet
        #     )

        # # stub _prepare_encoder_decoder_kwargs_for_generation(decoder_input_ids, **kwargs)
        # if not hasattr(unet, "_prepare_encoder_decoder_kwargs_for_generation"):
        #     def _prepare_encoder_decoder_kwargs_for_generation(self, decoder_input_ids, **kwargs):
        #         # for UNet, we don’t need extra kwargs—return empty dict
        #         return {}
        #     unet._prepare_encoder_decoder_kwargs_for_generation = types.MethodType(
        #         _prepare_encoder_decoder_kwargs_for_generation, unet
        #     )
        # reassign back
        # self.model.base_unet = unet
        # — end stub —
        # # -------------------------------------------------------

        # CubeDiff authors fine-tune all ∼17 M inflated-attention weights directly, not adapters. 
        # The LoRA stats (attach 4) show near-zero updates—which explains the seams and noise in panorama.
        # 5) Inject PEFT‐LoRA adapters (Hu et al. 2021)
        # lora_cfg = LoraConfig(
        #     # If LoRA weight‐std plateaus ~0.02 but grads stay ~1e-6, increase its step size.
        #     r=self.config.get("lora_r", 4),
        #     # bump alpha to 32 for larger effective LR in LoRA subspace
        #     lora_alpha=self.config.get("lora_alpha", 32),
        #     # That will ensure I am actually injecting adapters into all the Q/K/V and output projections .
        #     target_modules=[
        #                     "to_q",       # matches every q-proj Linear/Conv
        #                     "to_k",       # matches every k-proj
        #                     "to_v",       # matches every v-proj
        #                     "to_out.0",   # matches ONLY the Linear inside the ModuleList
        #                     ],
        #     lora_dropout=0.05,
        #     bias="none"            
        # ) ----------------------------------

        # # PEFT-LoRA: target the first (Linear) element inside each ModuleList
        # self.model.base_unet = get_peft_model(self.model.base_unet, lora_cfg)
        # print(f"[LoRA] trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # Ensure all model components use the same precision
        # Cast all model components to the same dtype
        # self.model.base_unet = self.model.base_unet.to(dtype=self.model_dtype)
        # self.model.base_unet = UNet2DConditionModel.from_pretrained(
        #     pretrained_model_name,
        #     subfolder="unet",
        #     torch_dtype=self.model_dtype,          # <— tell HF to load as fp16 or bf16/float32
        #     revision=self.config.get("revision", None)
        #     # remove any load_in_4bit / load_in_8bit flags here
        # )
        self.text_encoder = self.text_encoder.to(dtype=self.model_dtype)
        self.vae = self.vae.to(dtype=self.model_dtype)

        # Cast scheduler tensors
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(self.model_dtype))

        # Add debug info
        print(f"trainer.py - CubeDiffTrainer- CubeDiff Model components cast to {self.model_dtype}")

        # cast the adapted UNet to bfloat16 for memory/speed
        # self.model.base_unet = self.model.base_unet.to(torch.bfloat16)

        # 5) Enable gradient checkpoints and circular padding
        #   Enable gradient checkpointing on the U-Net backbone only
        #    (saves ~30–40% memory at the cost of ~10–20% extra compute)
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

        print(f"trainer.py - CubeDiffTrainer - train data - self.train_size is {self.train_size}, world_size is {world_size}, samples_per_rank is {samples_per_rank}, num_epochs is {num_epochs}, total_samples is {total_samples}, batch_size is {batch_size}, batch_num_per_rank is {batch_num_per_rank}\n")
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

        # the LR will decay smoothly over exactly the number of updates you actually perform, 
        # instead of “waiting” through 700 steps that never happen
        train_size    = self.train_size                          # e.g. 700
        world_size    = self.accelerator.num_processes           # 8
        bs_per_gpu    = self.config["batch_size"]                # 2
        accum_steps   = self.config["gradient_accum_steps"]      # 4
        epochs        = self.config["num_epochs"]                # 18

        global_batch = bs_per_gpu * accum_steps * world_size
        true_steps   = math.ceil(train_size * epochs / global_batch)

        # then override your config:
        # sched = CosineAnnealingLR(optim, T_max=true_steps)
        # sched = CosineAnnealingLR(optim, T_max=true_steps, eta_min=1e-6)
        warmup     = self.config.get("warmup_steps", 1000)
        # optim, sched = self.accelerator.prepare(optim, sched)
        sched = get_linear_schedule_with_warmup( optim,  num_warmup_steps=warmup, num_training_steps=true_steps)

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
                
                # log_batch_memory(rank, epoch, batch_indx, batch, stage="BEFORE accelerator.accumulate")

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
                        del ids  # Free memory as soon as possible
                    # print(f"trainer.py - train() - before pred = self.model, lat shape is {lat.shape} and dtype is {lat.dtype}, noisy_lat shape is {noisy_lat.shape} type is {noisy_lat.dtype}, timesteps is {timesteps}, txt_emb shape is {txt_emb.shape}, type is {txt_emb.dtype}\n")

                    # CubeDiff requires randomly dropping each modality 10% of the time during training so the model learns text-only, image-only, 
                    # and joint modes . Without this, it overfits to always having both conditions and fails to generalize.
                    # each micro-batch randomly drops text or image conditioning at 10 % each
                    # — Classifier-Free Guidance drops (§4.5):
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
                    # The model predicts only the 4 “true” latent channels;
                    # our `noise` still has 5 (4 + mask). Drop the mask so shapes align.
                    # reasons: 
                    # CubeDiff’s U-Net output (pred) has shape [B,6,4,H,W] (only the original 4 latent dims) 
                    # noise was sampled to match the 5-channel input (latent + mask), giving [B,6,5,H,W].
                    # By slicing off the last (mask) channel—noise[:, :, :pred.size(2), ...]—you restore a [B,6,4,H,W] tensor that 
                    # lines up perfectly with pred for the mean-squared‐error.
                    # benifits:
                    # keeps your mask channel around for conditioning but prevents it from polluting your denoising loss. After this update, 
                    # your MSE, boundary, and perceptual losses will all compute correctly, and you’ll begin to see your training curves converge 
                    # rather than erroring out.
                    if noise.size(2) != pred.size(2):
                        noise = noise[:, :, : pred.size(2), :, :]
                    mse_loss = F.mse_loss(pred.float(), noise.float())

                    # boundary loss
                    # Without enough weight on seam and perceptual terms, the model ignores overlap learning; and running VGG in bf16 loses detail.
                    # boundary loss (upweight to 1.0 to really force seam consistency)
                    # it is faithful to CubeDiff’s paper (§3.2–3.4) and its ablations (A.11), 
                    # and they mesh with industry best practices in multi-view diffusion and low-rank adaptation.
                    bdy = self.boundary_loss(pred) * self.config.get("boundary_weight", 1.0)

                    # ── Perceptual loss in FULL FP32 ──
                    # Disable autocast so VGG stays in FP32 and gradients are stable.
                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        # Decode latents → RGB in FP32
                        pred_rgb = self.decode_latents_to_rgb(pred).float()
                        true_rgb = self.decode_latents_to_rgb(noise).float()
                        # Perceptual features & L1 in FP32
                        fp = self.perceptual_net(pred_rgb)
                        ft = self.perceptual_net(true_rgb)
                        perc = self.l1(fp, ft) * self.config.get("perceptual_weight", 0.1)

                    # Ensure all three terms are FP32 before summing
                    loss = mse_loss.float() + bdy.float() + perc.float()
                    
                    # collect loss                    
                    self.accelerator.backward(loss)  # ← compute gradients

                    #  collect total steps
                    processed_samples += real_batch_size
                    local_count = torch.tensor(processed_samples, device=self.accelerator.device)
                    total_processed_samples = self.accelerator.reduce(local_count, reduction="sum")
                
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

                        # Log LoRA adapter weight stats to verify fine-tuning —
                        # if (self.global_iter%self.config["log_lora_every_n_steps"])==0:
                            # print(f"\t\tRank {rank} - trainer.py - CubeDiffTrainer - train - log lora - epoch {epoch} - self.global_iter is {self.global_iter}")
                            # self.lora_logger.record_batch(self.model, self.accelerator, self.global_iter)
                            # if self.accelerator.is_main_process:
                                # self.lora_logger.plot_up_to(self.global_iter, total_processed_samples)
                                # print(f"\t\tRank {rank} - train - self.lora_logger.record_batch done for epoch {epoch}, self.global_iter { self.global_iter}, plot_up_to done")

                        # 1) gradient clipping
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                        # 2) step the optimizer (handles unscaling, all‐reduce, zero‐grad)
                        optim.step()
                        
                        # 3) then update your LR scheduler
                        sched.step() 

                        # 4) zero gradients (optional - can be after or before step)
                        optim.zero_grad()   # ← now clear grad for next iteration        
                    # print(f"\t\tRank {rank} - train - after self.accelerator.sync_gradients - epoch {epoch} - self.global_iter is {self.global_iter}")
                    self.global_iter += 1

                end_tm = time.time()
                print(f"\tRank {rank} - epoch {epoch} - out of accumulate - batch_indx {batch_indx} cost {end_tm - start_tm:.2f} seconds")

                # --- Log after computation ---
                # log_batch_memory(rank, epoch, batch_indx, batch, stage="AFTER accelerate.accumulate")

                # peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                # print(f"\tRank {rank} epoch {epoch} - batch_indx {batch_indx} - gstep {gstep} - time: {end_tm - start_tm:.2f} seconds - Peak CUDA memory this batch: {peak:.2f} GB")
                    
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
                        val_loss = self.evaluate(rank) # this cost some time
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
                        self._plot_loss_curves(total_processed_samples, train_losses, val_losses)
                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - plotted loss - gstep {gstep} - self.accelerator.is_main_process is {self.accelerator.is_main_process} - log_loss_every_n_steps is {self.config['log_loss_every_n_steps']} - global_iter is {self.global_iter-1} - batch_size is {batch_size}, samples {processed_samples:>4}  train-loss {loss.item():.4f}")
                    # 2) every `eval_every_n_samples`, first sync _all_ ranks…
                    #  eval + sample‐gen
                    if (eval_every_n_samples and processed_samples >= prev_evaL_at) or (batch_indx == batch_num_per_rank - 1):
                        # generate panorama from current LoRA checkpoint
                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - generate_samples ...")
                        temp_s_time = time.time()
                        self.generate_samples(rank, sample_prompts, total_processed_samples)
                        temp_e_time = time.time()
                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - generate_samples done - cost {temp_e_time - temp_s_time:.2f} seconds")    
                        
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
                                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - trainer.py - Verify U-Net’s LoRA are actually updating - Found LoRA weight in module: {name}")
                                        break
                            
                            # Option 2: Original approach (fallback)
                            elif hasattr(unwrapped.base_unet, "lora_layers") and len(unwrapped.base_unet.lora_layers) > 0:
                                lora_params = unwrapped.base_unet.lora_layers[0].weight
                                print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - trainer.py - Verify U-Net’s LoRA are actually updating - base_unet has lora_layers with size>0")

                            # Option 3: Search for any LoRA-like parameters
                            else:
                                for name, param in unwrapped.base_unet.named_parameters():
                                    if 'lora' in name.lower() and param.requires_grad:
                                        lora_params = param
                                        print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - Verify U-Net’s LoRA are actually updating - Found LoRA parameter: {name}")
                                        break
                            
                            if lora_params is not None:
                                w = lora_params.detach().cpu().view(-1)
                                print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - trainer.py - Verify U-Net’s LoRA are actually updating - LoRA weight stats - mean: {w.mean():.6f}  std: {w.std():.6f}")
                            else:
                                print("\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - Warning: Could not find LoRA parameters for logging")
                                
                        except Exception as e:
                            print(f"\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - train - Verify U-Net’s LoRA are actually updating - Error accessing LoRA parameters: {e}")
                            print("\tRank {rank} - epoch {epoch} - batch_indx {batch_indx} - train - Verify U-Net’s LoRA are actually updating - Continuing training anyway...")

                gstep += 1
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

        # 2) read your U-Net weights back
        adapter_state = torch.load(tmp_ckpt, map_location="cuda")
        
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
    # ✱✱✱  New: run a full pass over val_dataloader and return avg loss. ✱✱✱
    # reuse boundary_loss helper inside train()
    # and decode_latents_to_rgb + self.perceptual_net    
    
    def evaluate(self, rank):
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
                mse_loss = F.mse_loss(pred.float(), noise.float())

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
                    perc = self.l1(fp, ft) * self.config.get("perceptual_weight", 0.1)

                loss = mse_loss + bdy + perc
                total_loss   += loss.item() * lat.size(0)
                total_samples+= lat.size(0)
                # print(f"trainer.py - evaluate() - after pred = self.model, total_loss is {total_loss}, total_sample is {total_samples}\n")

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

    def _plot_loss_curves(self, total_processed_samples: int, train_losses, val_losses):
        """Plot & save train/val losses up to `up_to` total_processed_samples."""
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
        plt.title(f"Training & Validation Loss up to {total_processed_samples} samples")
        plt.xlabel("samples")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        out = self.output_dir / f"loss_curve_up_to_total_processed_examples.png"
        plt.savefig(out)
        plt.close()
        print(f"✔ live loss‐curve up to total_processed_samples {total_processed_samples} → {out}")

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
        # 1) loop every LoRA param on GPU
        for full_name, param in unet.named_parameters():
            if not (param.requires_grad and ".lora_" in full_name):
                continue
            module_key = full_name.rsplit(".lora_", 1)[0]

            # 2) local stats on device
            p_data = param.detach()
            flat_p = p_data.view(-1)
            w_mean_t = flat_p.mean()
            w_std_t  = flat_p.std(unbiased=False)

            if param.grad is not None:
                g_data = param.grad.detach()
                flat_g = g_data.view(-1)
                g_mean_t = flat_g.mean()
                g_std_t  = flat_g.std(unbiased=False)
            else:
                # ensure a tensor for reduction
                g_mean_t = torch.tensor(float("nan"), device=accelerator.device)
                g_std_t  = torch.tensor(float("nan"), device=accelerator.device)

            # 3) all-reduce across all GPUs (average)
            w_mean = accelerator.reduce(w_mean_t, reduction="mean")
            w_std  = accelerator.reduce(w_std_t,  reduction="mean")
            g_mean = accelerator.reduce(g_mean_t, reduction="mean")
            g_std  = accelerator.reduce(g_std_t, reduction="mean")

            # 4) only record on the main process
            if accelerator.is_main_process:
                self._records.append({
                    "iter":      iteration,
                    "module":    module_key,
                    "full_name": full_name,
                    "W_mean":    w_mean.item(),
                    "W_std":     w_std.item(),
                    "G_mean":    g_mean.item(),
                    "G_std":     g_std.item(),
                })

    def plot_up_to(self, iteration: int, total_processed_samples: int = None):
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
            ax.set_title(f"{title} up to iter {iteration} with total_processed_samples {total_processed_samples}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            out = self.output_dir / f"lora_live_plot_{i}.png"
            fig.savefig(out)
            plt.close(fig)

    def finalize(self):
        """Write CSV + four summary plots to disk."""
        df = pd.DataFrame(self._records)
        # 1) save raw history
        csv_path = self.output_dir / "lora_full_history.csv"
        df.to_csv(csv_path, index=False)
        # print(f"[LoRALogger] raw history → {csv_path}")

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
        summary_csv = self.output_dir / "lora_final_summary.csv"
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
            out_png = self.output_dir / f"lora_final_plot_{i}.png"
            fig.savefig(out_png)
            print(f"[LoRALogger] saved {out_png}")
            plt.close(fig)


def log_batch_memory(rank, epoch, batch_indx, batch, stage="BEFORE"):
    batch_size = batch["latent"].size(0)
    print(f"\tRank {rank} - epoch {epoch} - Batch {batch_indx}: batch size is {batch_size} - log_batch_memory")
    print(f"\t\tRank {rank} - latent shape: {batch['latent'].shape}")

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    cpu_mem = psutil.virtual_memory().used / (1024 ** 3)

    print(f"\t\tRank {rank} - [{stage}] CUDA memory allocated: {allocated:.2f} GB")
    print(f"\t\tRank {rank} - [{stage}] CUDA memory reserved:  {reserved:.2f} GB")
    print(f"\t\tRank {rank} - [{stage}] CPU RAM used: {cpu_mem:.2f} GB")

# EOF -------------------------------------------------------------------------
