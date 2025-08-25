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
│   └── parallel-numpy-dataset.py  # Parallel dataset creation from NIFTI files
├── inference/
│   ├── comprehensive_visualization.py  # Complete visualization of all methods
│   └── segmentation_inference.py       # Comprehensive segmentation evaluation
├── models/
│   ├── train_base_unet.py         # Base U-Net reconstruction model training
│   ├── train_seg_unet.py          # Segmentation U-Net model training  
│   ├── train_task_adapted.py      # Task-adaptive reconstruction model training
│   └── joint_training_model.py    # Joint reconstruction and segmentation training
├── LICENSE                        # GPL v3.0 license
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation
└── .gitignore                     # Git ignore file
```

## Key Features

- **Low-dose Simulation**: Realistic simulation of low-dose CT using Poisson noise in the Radon domain
- **Image Reconstruction**: U-Net based architecture for high-quality image reconstruction
- **Segmentation**: Multi-class segmentation with 3 target classes
- **Task-Adaptive Learning**: Reconstruction model optimized using both image quality and segmentation accuracy metrics
- **Evaluation Metrics**: PSNR and SSIM for reconstruction quality; Dice coefficient for segmentation accuracy
- **Modular Design**: Separate training scripts for different approaches
- **Comprehensive Evaluation**: Multiple reconstruction methods compared including FBP, BM3D, and U-Net variants
- **Parallel Processing**: Multi-core support for faster dataset creation and evaluation
- **Comprehensive Visualization**: Complete comparison grids and segmentation overlays
- **Automatic Logging**: Detailed result tracking with JSON and plot outputs
- **Reproducible Training**: Configurable random seeds and systematic evaluation
- **GPU Acceleration**: CUDA support for training and inference
- **Command Line Interface**: Flexible training with alpha parameter configuration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/task_adaptive_ct.git
   cd task_adaptive_ct
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -n task_adaptive_ct python=3.9
   conda activate task_adaptive_ct
   pip install -r requirements.txt
   ```

## Requirements

The project dependencies are listed in `requirements.txt` and include:

### Core ML and Scientific Computing
- **PyTorch (>=1.9.0)** and **torchvision (>=0.10.0)** for deep learning
- **NumPy (>=1.21.0)** for numerical computations  
- **SciPy (>=1.7.0)** for scientific computing

### Image Processing and Computer Vision
- **scikit-image (>=0.18.0)** for image processing (radon transforms, PSNR/SSIM)
- **matplotlib (>=3.4.0)** for visualization and plotting
- **bm3d (>=4.0.0)** for BM3D denoising algorithm

### Medical Imaging
- **nibabel (>=3.2.0)** for NIFTI file handling

### Data Processing and Analysis
- **pandas (>=1.3.0)** for data analysis and CSV output
- **tqdm (>=4.61.0)** for progress bars

The project also uses standard Python libraries: `os`, `datetime`, `json`, `argparse`, `copy`, `random`, `functools`, `multiprocessing`.

## Configuration

Before running any scripts, you need to configure the environment. All scripts use configuration sections at the top of each file. Key parameters to adjust:

- **Dataset paths**: Update `low_dose_dir`, `full_dose_dir`, `seg_dir` to your data locations
- **Training parameters**: Adjust `batch_size`, `learning_rate`, `num_epochs` as needed
- **Model parameters**: Configure `num_classes`, loss weights, etc.
- **Output paths**: Set directories for saving models, logs, and results

## Usage

### 1. Dataset Preparation

```bash
# Create the dataset from raw NIFTI files with parallel processing
python dataset/parallel-numpy-dataset.py
```

**Configuration**: Edit the `CONFIG` dictionary at the top of the script to set:
- `volume_folder`: Path to input NIFTI volume files  
- `segmentation_folder`: Path to input NIFTI segmentation files
- `output_dir`: Output directory for preprocessed .npy files
- `target_size`: Target image size (default: 512x512)
- `dose_fraction`: Low-dose simulation parameter (default: 0.75)

### 2. Model Training

#### Base U-Net Reconstruction Model
```bash
python models/train_base_unet.py
```

#### Segmentation U-Net Model  
```bash
python models/train_seg_unet.py
```

#### Task-Adaptive Reconstruction Model
```bash
# Train with different alpha values for loss weighting
python models/train_task_adapted.py --alpha 0.1
python models/train_task_adapted.py --alpha 0.5  
python models/train_task_adapted.py --alpha 0.9
```

#### Joint Training Model
```bash
# Train reconstruction and segmentation models jointly
python models/joint_training_model.py --alpha 0.1
python models/joint_training_model.py --alpha 0.5
python models/joint_training_model.py --alpha 0.9
```

**Alpha Parameter**: Controls the balance between reconstruction loss (MSE) and segmentation loss (Dice):
- α = 0.1: 90% reconstruction, 10% segmentation  
- α = 0.5: 50% reconstruction, 50% segmentation
- α = 0.9: 10% reconstruction, 90% segmentation

### 3. Evaluation and Inference

#### Comprehensive Segmentation Evaluation
```bash
python inference/segmentation_inference.py
```
Generates detailed metrics (PSNR, SSIM, Dice) for all reconstruction methods and saves results to CSV/JSON.

#### Complete Visualization
```bash  
python inference/comprehensive_visualization.py
```
Creates side-by-side comparison grids and segmentation overlays for random test samples.

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

## Evaluation Methods

The following reconstruction methods are comprehensively compared:

### Baseline Methods
- **Low-dose images** (input) - Direct low-dose CT scans
- **FBP reconstruction** (Filtered Back Projection) - Classical reconstruction
- **BM3D denoising** - Block-matching 3D denoising on low-dose images

### Deep Learning Methods
- **Base U-Net reconstruction** - Standard U-Net trained for reconstruction only
- **Joint training reconstruction** (α = 0.1, 0.5, 0.9) - Simultaneous reconstruction and segmentation
- **Task-adaptive reconstruction** (α = 0.1, 0.5, 0.9) - Reconstruction guided by pretrained segmentation

### Evaluation Metrics
- **PSNR/SSIM**: Image quality metrics (both full image and masked ROI)
- **Dice Score**: Segmentation accuracy on reconstructed images
- **Processing Time**: Computational efficiency comparison

Results are automatically saved in multiple formats:
- **CSV files**: Tabular results with statistics
- **JSON files**: Detailed results with raw data
- **Visualization plots**: Comparison grids and overlays

## Output Structure

Training and evaluation scripts generate organized outputs:

```
output_directories/
├── logs/                           # Training logs and plots
│   ├── base-unet/                 # Base U-Net training outputs
│   ├── seg-unet/                  # Segmentation training outputs  
│   ├── joint-training-alpha*/     # Joint training outputs (by alpha)
│   └── task-adaptive-alpha*/      # Task-adaptive training outputs
├── model-weights/                 # Saved model weights (.pth files)
├── inference-results/             # Evaluation results and metrics
└── visualizations/                # Generated comparison plots
```

Each training run creates:
- **Detailed logs**: Training progress and metrics
- **Training curves**: Loss and validation plots
- **Model weights**: Best performing models  
- **Configuration files**: Reproducible settings
- **Results summaries**: Performance statistics

## Pre-trained Weights

Pre-trained model weights are not included in this repository due to size constraints. Train the models using the provided scripts. All model weights are automatically saved to `/path/to/model-weights/` with descriptive names:

- `base_unet_reconstruction.pth` - Base reconstruction model
- `seg_unet.pth` - Segmentation model  
- `joint_recon_unet_alpha*.pth` - Joint training models
- `task_adaptive_unet_alpha*.pth` - Task-adaptive models

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this code in your research, please consider citing:

```bibtex
@misc{task_adaptive_ct,
  title={To be filled},
  author={To be filled},
  year={To be filled},
  url={To be filled}
}
```

## Contact

For questions and feedback, please open an issue in the GitHub repository.
