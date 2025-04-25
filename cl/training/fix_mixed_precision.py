import torch
from torch.optim import AdamW
import types
import time
from cl.training.system_config import monitor_memory, clear_gpu_memory, detailed_memory_usage

def fix_mixed_precision_issue(trainer):
    """
    Fix issues with mixed precision training, particularly the unscale_() error.
    
    This function patches the training loop to ensure gradients are properly handled
    when using mixed precision training with gradient accumulation.
    """
    
    # Store the original train method
    original_train = trainer.train
    
    def patched_train(self, train_dataset, val_dataset=None, num_train_epochs=30000):
        """Patched version of the train method that fixes mixed precision issues"""
        print("Using patched train method to fix mixed precision issues")
        
        # Empty CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Prepare dataloaders
        train_dataloader, val_dataloader = self.prepare_dataloaders(train_dataset, val_dataset)
        
        # Collect trainable parameters properly
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Ensure parameters are float32 for optimizer stability
                if param.dtype != torch.float32:
                    param.data = param.data.to(dtype=torch.float32)
                trainable_params.append(param)
        
        # Initialize optimizer with properly collected parameters
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_train_epochs,
            eta_min=self.config.min_learning_rate,
        )
        
        # Prepare everything with accelerator
        self.model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, lr_scheduler
        )
        
        if val_dataloader:
            val_dataloader = self.accelerator.prepare(val_dataloader)
            
        # Call original train method but without the training part
        # Instead we'll define our own fixed training loop here
        
        # Setup for generating sample images
        # if hasattr(self.config, 'sample_prompts') and self.config.sample_prompts:
        #     sample_prompts = self.config.sample_prompts
        # else:
        #     sample_prompts = [
        #         "A beautiful mountain lake at sunset with snow-capped peaks",
        #         "A futuristic cityscape with tall skyscrapers and flying vehicles"
        #     ]
        
        # Training loop
        global_step = 0
        import datetime
        start_time = datetime.datetime.now()
        
        # Create a local training log file
        import os
        log_file = os.path.join(self.logs_dir, "training_log.csv")
        with open(log_file, 'w') as f:
            f.write("step,loss,learning_rate,time_elapsed\n")
        
        # Metrics dictionary
        metrics = {'loss': [], 'step': [], 'lr': [], 'time': []}
        
        self.model.train()
        from tqdm.auto import tqdm
        progress_bar = tqdm(range(num_train_epochs), desc="Training")
        
        print("fix_mixed_precision.py - fix_mixed_precision_issue - Patched main training loop started with", num_train_epochs, "epochs ...")
        print(f"self.config is {self.config}")
        # Main training loop 
        # Call before your training starts
        detailed_memory_usage()
        try:
            for epoch in range(num_train_epochs):
                torch.cuda.empty_cache()
                epoch_start_time = time.time()
                print(f"------------ fix_mixed_precision.py - fix_mixed_precision_issue - Patched main training loop started with epoch {epoch} ----------")
                for step, batch in enumerate(train_dataloader):
                    torch.cuda.empty_cache()
                    step_start_time = time.time()
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - training step started with step {step}, batch_size is {self.config.batch_size} ...")
                    # Extract batch data
                    faces = batch["faces"]  # [batch, 6, C, H, W]
                    captions = batch["caption"]
                    
                    # Process text
                    tokens = self.tokenizer(
                        captions, 
                        padding="max_length", 
                        max_length=self.tokenizer.model_max_length, 
                        truncation=True, 
                        return_tensors="pt"
                    ).input_ids.to(self.accelerator.device)
                    
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - after process captions text")

                    with torch.no_grad():
                        text_embeddings = self.text_encoder(tokens)[0]
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - after text_embeddings = self.text_encoder(tokens)[0]")

                    # Process face images with VAE
                    latents = []
                    vae_param = next(self.vae.parameters())
                    vae_dtype = vae_param.dtype
                    vae_device = vae_param.device
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - process each facew started ...")
                    for i in range(faces.shape[1]):
                        face = faces[:, i]
                        if face.shape[1] == 4 and face.shape[2] == 64 and face.shape[3] == 64:
                            # Already a latent, no need to encode
                            face_latent = face
                        else:                          
                            # Ensure correct format
                            if face.shape[-1] == 3:  # If channels are last
                                face = face.permute(0, 3, 1, 2)  # NHWC -> NCHW
                            
                            # Convert to VAE dtype and device
                            face = face.to(dtype=vae_dtype, device=vae_device)
                            
                            # Normalize if needed
                            if face.max() > 1.0:
                                face = face / 127.5 - 1.0
                            
                            # Encode with VAE
                            # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - Encode with VAE - face shape is {face.shape}")
                            with torch.no_grad():
                                face_latent = self.vae.encode(face).latent_dist.sample() * 0.18215
                        
                        latents.append(face_latent)
                    
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - all faces processed")
                    # Stack latents
                    latents = torch.stack(latents, dim=1)
                    
                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, 
                        self.noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=self.accelerator.device,
                    )
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - ssert noise")

                    # The key fix: proper gradient accumulation with accelerator
                    # This avoids the unscale_() error by ensuring the optimizer step
                    # is called correctly within the accumulation context
                    with self.accelerator.accumulate(self.model):
                        # Add noise to latents (use float32 for stability)
                        latents_f32 = latents.float()
                        noise_f32 = noise.float()
                        noisy_latents = self.noise_scheduler.add_noise(
                            latents_f32, noise_f32, timesteps
                        ).to(latents.dtype)
                        # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - added noise")

                        # Get model prediction
                        noise_pred = self.model(noisy_latents, timesteps, text_embeddings)
                        # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - get model prediction")

                        # Compute loss (use float32 for stability)
                        if self.config.prediction_type == "epsilon":
                            target = noise
                        elif self.config.prediction_type == "v_prediction":
                            target = self.noise_scheduler.get_velocity(
                                latents.float(), noise.float(), timesteps
                            )
                        else:
                            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
                        # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - \
                            #   self.config.prediction_type is {self.config.prediction_type}, set target")

                        # Make sure both tensors have the same dtype for loss computation
                        noise_pred = noise_pred.float()
                        target = target.float()
                        
                        loss = torch.nn.functional.mse_loss(noise_pred, target)
                        # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - get mse_loss between noise_pred, target")

                        # Backward pass
                        self.accelerator.backward(loss)
                        # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - accelerator backward")

                        # Clip gradients if needed
                        if self.accelerator.sync_gradients:                 # <─ only on last micro‑batch
                            if self.config.max_grad_norm > 0:
                                self.accelerator.clip_grad_norm_(
                                    filter(lambda p: p.requires_grad, self.model.parameters()),
                                    self.config.max_grad_norm,
                                )
                            
                            # Update model parameters - within the accumulation context
                            # update parameters + scaler in one call
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)              # clear grads for next accumulation window
                            # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - self.accelerator.sync_gradients is {self.accelerator.sync_gradients}")
                        # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - after self.accelerator.sync_gradients")

                    # Log metrics and save checkpoints
                    if global_step % self.config.log_every_n_steps == 0:
                        time_elapsed = (datetime.datetime.now() - start_time).total_seconds() / 60.0
                        current_lr = lr_scheduler.get_last_lr()[0]
                        loss_value = loss.item()
                        
                        # Update metrics
                        metrics['loss'].append(loss_value)
                        metrics['step'].append(global_step)
                        metrics['lr'].append(current_lr)
                        metrics['time'].append(time_elapsed)
                        
                        # Update progress bar
                        progress_bar.set_postfix(loss=loss_value, lr=f"{current_lr:.2e}")
                        
                        # Log to wandb
                        if self.wandb_run:
                            self.wandb_run.log({
                                "train/loss": loss_value,
                                "train/lr": current_lr,
                                "train/epoch": epoch,
                                "train/step": global_step,
                                "train/time_minutes": time_elapsed
                            })
                    
                    # Save checkpoint
                    if global_step % self.config.save_every_n_steps == 0:
                        self.save_checkpoint(os.path.join(self.output_dir, f"checkpoint-{global_step}"))
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - Save checkpoint")

                    # Generate samples
                    # if hasattr(self.config, 'sample_every_n_steps') and \
                    # self.config.sample_every_n_steps > 0 and \
                    # global_step % self.config.sample_every_n_steps == 0:
                    #     self.generate_samples(sample_prompts, global_step)
                    # print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - generate_samples")

                    # Evaluate model
                    if val_dataloader and global_step % self.config.eval_every_n_steps == 0:
                        eval_loss = self.evaluate(val_dataloader)
                        
                        # Log validation loss
                        if self.wandb_run:
                            self.wandb_run.log({"val/loss": eval_loss})
                    
                    print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - Evaluate model with eval_loss as {eval_loss}")

                    global_step += 1
                    progress_bar.update(1)
                    step_end_time = time.time()
                    # # Break if maximum steps reached
                    # if global_step >= num_train_epochs:
                    #     break
                    clear_gpu_memory()
                    # monitor_memory()
                    torch.cuda.empty_cache()
                    print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - training step {step} done, cost {step_end_time - step_start_time} seconds")
                # Break if maximum steps reached
                # if global_step >= num_train_epochs:
                #     break
            epoch_end_time = time.time()
            torch.cuda.empty_cache()
            clear_gpu_memory()
            # monitor_memory()
            print(f"*** fix_mixed_precision.py - fix_mixed_precision_issue - epoch {epoch} done, cost {epoch_end_time - epoch_start_time} seconds")

        except Exception as e:
            print(f"Runtime error: {str(e)}")
            print(f"Training error: {str(e)}")
            print(f"\nError during training: {str(e)}")
            # Save checkpoint before exiting
            self.save_checkpoint(os.path.join(self.output_dir, "error_checkpoint"))
            raise
        
        print(f"------------ fix_mixed_precision.py - fix_mixed_precision_issue - Patched main training loop done with num_train_epochs as \
              {num_train_epochs} and global_step as {global_step} and dataloader size is {global_step/num_train_epochs}\n")

        return global_step
    
    # Replace the method
    trainer.train = types.MethodType(patched_train, trainer)
    
    return trainer