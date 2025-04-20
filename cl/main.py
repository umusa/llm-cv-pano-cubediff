import os
import argparse
import yaml
import torch
from datetime import datetime
import json
from types import SimpleNamespace

from data.preprocessing import preprocess_panorama_dataset
from data.dataset import CubemapDataset, get_dataloader
from model.architecture import CubeDiffModel
from training.trainer import CubeDiffTrainer
from inference.pipeline import CubeDiffPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="CubeDiff: 360° Panorama Generation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default_config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["preprocess", "train", "inference"], 
        required=True, 
        help="Operation mode"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        help="Input directory for preprocessing"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory for preprocessing or inference"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Checkpoint path for inference"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="Text prompt for inference"
    )
    parser.add_argument(
        "--negative_prompt", 
        type=str, 
        default="low quality, blurry, distorted",
        help="Negative prompt for inference"
    )
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dict to namespace for easier access
    config = SimpleNamespace()
    
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(config, key, SimpleNamespace(**value))
        else:
            setattr(config, key, value)
    
    return config

def preprocess(args, config):
    if not args.input_dir or not args.output_dir:
        raise ValueError("Input and output directories must be specified for preprocessing")
    
    print(f"Preprocessing panoramas from {args.input_dir} to {args.output_dir}")
    preprocess_panorama_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        face_size=512,
    )
    
    print("Preprocessing complete")

def train(args, config):
    print("Starting training...")
    
    # Override output directory if provided
    output_dir = args.output_dir or config.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = os.path.join(output_dir, f"config_{timestamp}.yaml")
    
    # Convert config back to dict for saving
    config_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, SimpleNamespace):
            config_dict[key] = vars(value)
        else:
            config_dict[key] = value
    
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f)
    
    print(f"Configuration saved to {config_save_path}")
    
    # Create datasets
    train_dataset = CubemapDataset(
        data_dir=config.data.data_dir,
        captions_file=config.data.captions_file
    )
    
    # Split dataset into train/val if needed
    if hasattr(config.data, "val_split") and config.data.val_split > 0:
        from torch.utils.data import random_split
        
        val_size = int(len(train_dataset) * config.data.val_split)
        train_size = len(train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        val_dataset = None
    
    print(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = CubeDiffTrainer(
        config=config,
        pretrained_model_name=config.model.pretrained_model_name,
        output_dir=output_dir,
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps
    )
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_train_epochs=config.training.train_steps
    )
    
    print("Training complete")

def inference(args, config):
    if not args.prompt:
        raise ValueError("Text prompt must be specified for inference")
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print("Warning: No checkpoint specified, using base model")
    
    output_dir = args.output_dir or "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = CubeDiffPipeline(
        pretrained_model_name=config.model.pretrained_model_name,
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Generate panorama
    print(f"Generating panorama for prompt: {args.prompt}")
    panorama = pipeline.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=config.inference.num_inference_steps,
        guidance_scale=config.inference.guidance_scale,
        height=config.inference.height,
        width=config.inference.width,
        output_type="pil"
    )
    
    # Save panorama
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"panorama_{timestamp}.jpg")
    panorama.save(output_path)
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": config.inference.num_inference_steps,
        "guidance_scale": config.inference.guidance_scale,
        "checkpoint": checkpoint_path,
    }
    
    metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Panorama saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.mode == "preprocess":
        preprocess(args, config)
    elif args.mode == "train":
        train(args, config)
    elif args.mode == "inference":
        inference(args, config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()