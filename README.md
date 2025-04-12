# CubeDiff Implementation

Implementation of the "CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation" paper (January 2025).

This repository contains a complete implementation of the CubeDiff architecture for generating high-quality 360° panoramas from text prompts and narrow field-of-view (NFoV) images.

## Overview

CubeDiff is a novel approach to panorama generation that leverages the cubemap representation - dividing the 360° panorama into six perspective views, each with a 90° field-of-view. The architecture includes specialized components for ensuring consistency across the cubemap faces:

1. **Synchronized GroupNorm**: Ensures color consistency across the six faces by normalizing feature activations across both spatial and inter-view dimensions.
2. **Inflated Attention**: Extends the attention mechanism across all six cube faces, enabling the model to learn relationships and dependencies between different views.
3. **Cubemap Positional Encodings**: Adds spatial awareness by incorporating positional information derived from the 3D geometry of the cube.

## Repository Structure

- `cubediff_utils.py`: Utility functions for cubemap conversion
- `cubediff_models.py`: Model components and layers
- `cubediff_dataset.py`: Dataset classes for training
- `cubediff_trainer.py`: Training functionality
- `cubediff_inference.py`: Inference functionality
- `cubediff_optimization.py`: Memory optimization techniques
- `train_cubediff.py`: Script for training the model
- `run_inference.py`: Script for running inference
- `CubeDiff_Example.ipynb`: Jupyter notebook demonstrating key components and usage

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cubediff.git
cd cubediff

# Install dependencies
pip install -r requirements.txt
```

## Hardware Requirements

The implementation is optimized for GCP Vertex AI with the following specifications:
- 4 NVIDIA L4 GPUs (22GB VRAM each)
- Memory-optimized for maximum efficiency

## Usage

### Training

```bash
# Single GPU training
python train_cubediff.py \
    --model_id="runwayml/stable-diffusion-v1-5" \
    --dataset_path="/path/to/panorama_dataset" \
    --prompts_file="/path/to/prompts.txt" \
    --batch_size=1 \
    --num_epochs=10 \
    --learning_rate=1e-5 \
    --output_dir="./output" \
    --save_every=500 \
    --mixed_precision=True

# Distributed training (4 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_cubediff.py \
    --model_id="runwayml/stable-diffusion-v1-5" \
    --dataset_path="/path/to/panorama_dataset" \
    --prompts_file="/path/to/prompts.txt" \
    --batch_size=1 \
    --num_epochs=10 \
    --learning_rate=1e-5 \
    --output_dir="./output" \
    --save_every=500 \
    --mixed_precision=True \
    --optimize_memory
```

### Inference

```bash
# Text-to-panorama generation
python run_inference.py \
    --checkpoint_path="./output/checkpoints/unet_final" \
    --prompt="A beautiful mountain landscape at sunset with a lake in the foreground" \
    --output_dir="./generated" \
    --steps=50 \
    --guidance_scale=7.5 \
    --save_faces

# Image-to-panorama generation
python run_inference.py \
    --checkpoint_path="./output/checkpoints/unet_final" \
    --prompt="A beautiful mountain landscape" \
    --input_image="path/to/image.jpg" \
    --output_dir="./generated" \
    --steps=50 \
    --guidance_scale=7.5 \
    --save_faces
```

## Creating a Dataset

The implementation expects a directory of equirectangular panorama images and a text file with corresponding prompts:

```
dataset/
  ├── panorama_0001.jpg
  ├── panorama_0002.jpg
  └── ...
prompts.txt
  ├── "A beautiful mountain landscape at sunset"
  ├── "A city skyline with tall buildings"
  └── ...
```

You can use the utility function to create a small example dataset:

```python
from cubediff_utils import create_example_dataset
create_example_dataset(num_samples=10, output_dir="./example_data")
```

## Jupyter Notebook

The included `CubeDiff_Example.ipynb` notebook demonstrates all key components of the implementation:
- Cubemap conversion
- Model loading and inflated attention
- Inference with pretrained models
- Memory optimization techniques

## Memory Optimization

For the L4 GPUs on Vertex AI, the implementation includes several memory optimization techniques:
- Mixed precision training (FP16/BF16)
- Gradient checkpointing
- Optimized batch size (1 per GPU)
- Selective layer freezing (only attention layers are trained)

## License

This implementation is for research purposes only. Please refer to the original CubeDiff paper for additional information.

## Citation

```
@misc{kalischek2025cubediffrepurposingdiffusionbasedimage,
  title={CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation},
  author={Nikolai Kalischek and Michael Oechsle and Fabian Manhardt and Philipp Henzler and Konrad Schindler and Federico Tombari},
  year={2025},
  eprint={2501.17162},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2501.17162},
}
```

## Acknowledgements

This implementation is based on the research paper "CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation" and leverages components from HuggingFace Diffusers and Transformers libraries.