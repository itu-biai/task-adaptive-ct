# Task-Adaptive Low-Dose CT Reconstruction and Segmentation

This project implements a task-adaptive approach for low-dose CT image reconstruction and segmentation. It combines image quality enhancement with accurate segmentation to provide better diagnostic capabilities while reducing radiation exposure.

## Project Overview

The project consists of four main components:
1. **Low-dose CT reconstruction** using U-Net
2. **CT image segmentation** (3 classes: background, liver, tumor)
3. **Task-adaptive reconstruction** model that uses segmentation accuracy as a regularization term
4. **Joint training** approach for simultaneous reconstruction and segmentation

## Directory Structure

```
.
├── dataset/
│   └── parallel-numpy-dataset.py  # Parallel dataset creation and preprocessing
├── inference/
│   ├── comprehensive_visualization.py  # Advanced visualization and analysis
│   └── segmentation_inference.py       # Comprehensive segmentation evaluation
├── models/
│   ├── train_base_unet.py         # Base U-Net reconstruction model
│   ├── train_seg_unet.py          # Segmentation U-Net model
│   ├── train_task_adapted.py      # Task-adaptive reconstruction model
│   └── joint_training_model.py    # Joint reconstruction and segmentation training
├── LICENSE                        # GPL v3.0 license
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore file
```

## Key Features

- **Low-dose Simulation**: Realistic simulation of low-dose CT using dose reduction in the Radon domain
- **Image Reconstruction**: U-Net based architecture for high-quality image reconstruction from low-dose data
- **Multi-class Segmentation**: 3-class segmentation (background, liver, tumor) with dedicated U-Net
- **Task-Adaptive Learning**: Reconstruction model optimized using both MSE and Dice loss for segmentation accuracy
- **Joint Training**: Simultaneous optimization of reconstruction and segmentation with configurable loss weighting
- **Advanced Optimization**: Adam optimizer with learning rate scheduling (StepLR, ReduceLROnPlateau, CosineAnnealingLR)
- **Early Stopping**: Configurable early stopping with patience and minimum epochs
- **Comprehensive Evaluation**: Multiple reconstruction methods compared (FBP, BM3D, U-Net variants)
- **Evaluation Metrics**: PSNR and SSIM for reconstruction quality; Dice coefficient for segmentation accuracy
- **Parallel Processing**: Multi-core support for dataset creation, training, and inference
- **Advanced Visualization**: Comprehensive plotting and analysis with ROI-based evaluation
- **Automatic Logging**: Timestamped model checkpoints, training logs, and result visualization
- **Reproducible Training**: Configurable random seeds and deterministic training
- **Modular Design**: Separate training scripts for different approaches with shared U-Net architectures

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nsefercioglu/task_adaptive_ct.git
   cd task_adaptive_ct
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -n task_adaptive_ct python=3.9
   conda activate task_adaptive_ct
   pip install -r requirements.txt
   ```

## Requirements

The project dependencies are listed in `requirements.txt`:
- PyTorch (≥1.9.0) and torchvision (≥0.10.0) for deep learning
- NumPy (≥1.21.0) for numerical computations
- scikit-image (≥0.18.0) for image processing
- nibabel (≥3.2.0) for NIFTI file handling
- matplotlib (≥3.4.0) for visualization
- bm3d (≥4.0.0) for BM3D denoising algorithm
- scipy (≥1.7.0) for scientific computing
- tqdm (≥4.61.0) for progress bars

## Configuration

Before running any scripts, you need to configure the environment. All scripts use configuration sections at the top of each file. Key parameters to adjust:

- **Dataset paths**: Update dataset directory paths in CONFIG sections
  - `noisy_dir`/`low_dose_dir`: Path to low-dose CT images
  - `clean_dir`/`full_dose_dir`: Path to full-dose CT images  
  - `image_dir`: Path to full-dose images for segmentation training
  - `mask_dir`/`seg_dir`: Path to segmentation masks
- **Training parameters**: Adjust `batch_size` (16-64), `learning_rate` (1e-3), `num_epochs` (200) as needed
- **Model parameters**: Configure `num_classes` (3), loss weights, optimizer settings
- **Output paths**: Set `output_dir` for saving models, logs, and results
- **Hardware settings**: Adjust `num_workers` (64) and `batch_size` based on your system capabilities

## Usage

### 1. Dataset Preparation

The dataset preparation script processes NIFTI volumes and creates low-dose simulations:

```bash
# Create and preprocess the dataset with parallel execution
# This script will:
# - Load NIFTI volumes and segmentation masks
# - Resize images to 512x512
# - Simulate low-dose CT using dose reduction
# - Save preprocessed data as numpy arrays for training
python dataset/parallel-numpy-dataset.py
```

Configure the input and output paths in the CONFIG section of the script before running.

### 2. Model Training

```bash
# Train the base reconstruction U-Net
python models/train_base_unet.py

# Train the segmentation U-Net
python models/train_seg_unet.py

# Train the task-adaptive reconstruction model
# --alpha controls the weight for Dice loss (1-alpha for MSE loss)
python models/train_task_adapted.py --alpha 0.5

# Train both models jointly
# --alpha controls the weight for Dice loss (1-alpha for MSE loss)
python models/joint_training_model.py --alpha 0.5
```

### 3. Inference and Visualization

```bash
# Run comprehensive visualization and analysis
python inference/comprehensive_visualization.py

# Perform segmentation evaluation
python inference/segmentation_inference.py
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
- **Mean Squared Error (MSE) loss** for reconstruction quality
- **Dice loss** from a pre-trained segmentation model as a regularization term
- A **weighted combination** of both losses to balance image quality and segmentation accuracy

### Joint Training

The joint training approach:
- Trains reconstruction and segmentation models **simultaneously**
- Uses a **combined loss function** with configurable weighting
- Enables **end-to-end optimization** for both tasks

## Evaluation Metrics

The models are evaluated using:
- **PSNR (Peak Signal-to-Noise Ratio)** for reconstruction quality
- **SSIM (Structural Similarity Index)** for perceptual quality  
- **Dice coefficient** for segmentation accuracy
- **Masked metrics** using circular ROI for focused evaluation

## Benchmark Results

The following baseline methods are compared:
- **Low-dose images** (input)
- **FBP reconstruction** (Filtered Back Projection)
- **BM3D denoising**
- **Base U-Net reconstruction**
- **Task-adaptive reconstruction** (various alpha values)
- **Joint training reconstruction** (various alpha values)

Results are automatically logged during inference and visualization with PSNR, SSIM, and Dice metrics.

## Pre-trained Weights

Pre-trained model weights are not included in this repository. You can train the models using the provided scripts. Training logs and model checkpoints will be saved with timestamps for version control.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this code in your research, please consider citing:

```bibtex
@article{sefercioglu2025task,
  title={Task-Adaptive Low-Dose CT Reconstruction},
  author={Sefercioglu, Necati and Unal, Mehmet Ozan and Ertas, Metin and Yildirim, Isa},
  journal={arXiv preprint arXiv:2511.07094},
  year={2025}
}
```

## Contact

For questions and feedback, please open an issue in the GitHub repository.
