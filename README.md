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
├── benchmarks/
│   ├── benchmarks.py                   # Performance evaluation scripts
│   ├── parallel_benchmarks.py          # Parallel version for faster processing
│   └── benchmark_results.txt           # Benchmark results
├── dataset/
│   ├── create_numpy_dataset.py         # Dataset creation and preprocessing
│   ├── parallel-numpy-dataset.py       # Parallel dataset creation
├── inference/
│   ├── bounding-box.py                 # Interactive ROI analysis
│   ├── segmentation_inference.py       # Comprehensive segmentation evaluation
│   └── segmentation_task_adaptive.py   # Task-adaptive segmentation inference
│   └── segmentation_joint_training.py  # Joint training segmentation inference
├── models/
│   ├── train_base_unet.py              # Base U-Net reconstruction model
│   ├── train_seg_unet.py               # Segmentation U-Net model
│   ├── train_task_adapted.py           # Task-adaptive reconstruction model
│   └── joint_training_model.py         # Joint reconstruction and segmentation training
├── LICENSE                             # GPL v3.0 license
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore file
```

## Features

- **Low-dose Simulation**: Realistic simulation of low-dose CT using Poisson noise in the Radon domain
- **Image Reconstruction**: U-Net based architecture for high-quality image reconstruction
- **Segmentation**: Multi-class segmentation with 3 target classes
- **Task-Adaptive Learning**: Reconstruction model optimized using both image quality and segmentation accuracy metrics
- **Evaluation Metrics**: PSNR and SSIM for reconstruction quality; Dice coefficient for segmentation accuracy

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

The project dependencies are listed in `requirements.txt`:
- PyTorch and torchvision for deep learning
- NumPy for numerical computations
- scikit-image for image processing
- nibabel for NIFTI file handling
- matplotlib for visualization
- bm3d for BM3D denoising algorithm
- scipy for scientific computing
- tqdm for progress bars

## Usage

**Note**: Before running any scripts, you'll need to update the dataset paths in the configuration sections of each script to point to your local data directories.

### 1. Dataset Preparation

```bash
# Create the dataset from raw NIFTI files
python dataset/create_numpy_dataset.py

# For faster processing with parallel execution
python dataset/parallel-numpy-dataset.py
```

### 2. Model Training

```bash
# Train the base reconstruction U-Net
python models/train_base_unet.py

# Train the segmentation U-Net
python models/train_seg_unet.py

# Train the task-adaptive reconstruction model
python models/train_task_adapted.py

# Train both models jointly
python models/joint_training_model.py
```

### 3. Benchmarks

```bash
# Run performance benchmarks
python benchmarks/benchmarks.py

# For faster parallel processing
python benchmarks/parallel_benchmarks.py
```

### 4. Inference

```bash
# Interactive bounding box analysis
python inference/bounding-box.py

# Comprehensive segmentation evaluation
python inference/segmentation_inference.py

# Task-adaptive segmentation inference
python inference/segmentation_task_adaptive.py

# Joint training segmentation inference
python inference/segmentation_joint_training.py
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
- **Joint training reconstruction**
- **Task-adaptive reconstruction**

Results are automatically saved to `benchmarks/benchmark_results.txt`.

## Pre-trained Weights

Pre-trained model weights are not included in this repository. You can train the models using the provided scripts. Training logs and model checkpoints will be saved with timestamps for version control.

## Configuration

All scripts use configuration sections at the top of each file. Key parameters to adjust:

- **Dataset paths**: Update `low_dose_dir`, `full_dose_dir`, `seg_dir` to your data locations
- **Training parameters**: Adjust `batch_size`, `learning_rate`, `num_epochs` as needed
- **Model parameters**: Configure `num_classes`, loss weights, etc.
- **Output paths**: Set directories for saving models, logs, and results

## Features

- **Modular design** with separate training scripts for different approaches
- **Comprehensive evaluation** with multiple reconstruction methods
- **Parallel processing** support for faster dataset creation and benchmarking
- **Interactive analysis** tools for ROI-based evaluation
- **Automatic logging** and result visualization
- **Reproducible training** with configurable random seeds

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

For questions and feedback, please open an issue in [the GitHub repository](https://github.com/itu-biai/task_adaptive_ct/issues).
