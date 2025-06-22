# Task-Adaptive Low-Dose CT Reconstruction and Segmentation

This project implements a task-adaptive approach for low-dose CT image reconstruction and segmentation. It combines image quality enhancement with accurate segmentation to provide better diagnostic capabilities while reducing radiation exposure.

## Project Overview

The project consists of three main components:
1. Low-dose CT reconstruction using U-Net
2. CT image segmentation (3 classes)
3. Task-adaptive reconstruction model that uses segmentation accuracy as a regularization term in the reconstruction process

## Directory Structure

```
.
├── benchmarks/
│   └── benchmarks.py         # Performance evaluation scripts
├── dataset/
│   ├── create_numpy_dataset.py   # Dataset creation and preprocessing
│   └── visualize_samples.py      # Dataset visualization tools
├── inference/
│   └── inference.py          # Model inference and evaluation
├── models/
│   ├── train_ldct_unet.py       # Low-dose CT reconstruction model
│   ├── train_seg_unet.py        # Segmentation model
│   └── train_task_adapted.py    # Task-adaptive reconstruction model
└── weights/                  # Pre-trained model weights
    ├── best_reconstruction_unet_model_link.txt
    ├── best_seg_unet_model_link.txt
    └── best_task_adaptive_model_link.txt
```

## Features

- **Low-dose Simulation**: Realistic simulation of low-dose CT using Poisson noise in the Radon domain
- **Image Reconstruction**: U-Net based architecture for high-quality image reconstruction
- **Segmentation**: Multi-class segmentation with 3 target classes
- **Task-Adaptive Learning**: Reconstruction model optimized using both image quality and segmentation accuracy metrics
- **Evaluation Metrics**: PSNR and SSIM for reconstruction quality; Dice coefficient for segmentation accuracy

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-image
- nibabel (for NIFTI file handling)
- matplotlib

## Usage

### 1. Dataset Preparation

```bash
# Create the dataset from raw CT scans
python dataset/create_numpy_dataset.py

# Visualize samples from the dataset
python dataset/visualize_samples.py
```

### 2. Model Training

```bash
# Train the reconstruction U-Net
python models/train_ldct_unet.py

# Train the segmentation U-Net
python models/train_seg_unet.py

# Train the task-adaptive model
python models/train_task_adapted.py
```

### 3. Inference

```bash
# Run inference on test images
python inference/inference.py
```

## Model Architecture

All models use a U-Net architecture with:
- 4 downsampling blocks in the encoder
- 4 upsampling blocks in the decoder
- Skip connections between encoder and decoder
- Batch normalization and ReLU activation
- Final sigmoid activation for reconstruction / softmax for segmentation

### Task-Adaptive Training

The task-adaptive model combines:
- Mean Squared Error (MSE) loss for reconstruction quality
- Dice loss from a pre-trained segmentation model as a regularization term
- A weighted combination of both losses to balance image quality and segmentation accuracy

## Results

The models are evaluated using:
- PSNR (Peak Signal-to-Noise Ratio) for reconstruction quality
- SSIM (Structural Similarity Index) for perceptual quality
- Dice coefficient for segmentation accuracy

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Contact

For questions and feedback, please open an issue in the GitHub repository.