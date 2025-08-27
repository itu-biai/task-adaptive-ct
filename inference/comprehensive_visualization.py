"""
Comprehensive Visualization Script for CT Reconstruction and Segmentation

This script generates detailed visualizations for random test samples:
1. Reconstruction grid: Shows all reconstruction methods side by side
2. Segmentation overlays: Shows segmentation results overlaid on images

Author: Enhanced for comprehensive evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.transform import radon, iradon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import bm3d
from scipy.ndimage import zoom
from datetime import datetime
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

"""#### Configuration"""

CONFIG = {
    # Visualization parameters
    'num_samples': 50,  # Number of random samples to visualize
    'batch_size': 16,   # Optimized for memory and parallelization
    'num_workers': 64,  # Number of parallel processes
    'plot_workers': 16, # Separate workers for plot generation
    'train_test_split': 0.8,
    'num_classes': 3,
    
    # Random seed for reproducible sample selection
    'vis_random_seed': 42,
    
    # BM3D and FBP parameters
    'sigma_psd': 0.075,
    'theta': None,
    'n_processes': cpu_count(),
    
    # Dataset directories
    'low_dose_dir': '/YEREL/dataset/low_dose',
    'full_dose_dir': '/YEREL/dataset/full_dose',
    'seg_dir': '/YEREL/dataset/segmentation_slices',
    
    # Model weights paths
    'seg_model_path': '/ari/users/nsefercioglu/model-weights/seg_unet.pth',
    'base_unet_model_path': '/ari/users/nsefercioglu/model-weights/base_unet_reconstruction.pth',
    'joint_models': {
        'alpha01': '/ari/users/nsefercioglu/model-weights/joint_recon_unet_alpha01.pth',
        'alpha05': '/ari/users/nsefercioglu/model-weights/joint_recon_unet_alpha05.pth',
        'alpha09': '/ari/users/nsefercioglu/model-weights/joint_recon_unet_alpha09.pth'
    },
    'task_adaptive_models': {
        'alpha01': '/ari/users/nsefercioglu/model-weights/task_adaptive_unet_alpha01.pth',
        'alpha05': '/ari/users/nsefercioglu/model-weights/task_adaptive_unet_alpha05.pth',
        'alpha09': '/ari/users/nsefercioglu/model-weights/task_adaptive_unet_alpha09.pth'
    },
    
    # Output directory
    'output_dir': '/ari/users/nsefercioglu/visualizations'
}

"""#### Setup and Logging"""

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Use output directory directly (overwrite previous outputs)
run_dir = CONFIG['output_dir']
os.makedirs(run_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # For logging purposes only

print(f"Visualization outputs will be saved to: {run_dir}")

# Save configuration
config_path = os.path.join(run_dir, "visualization_config.json")
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=2)

# Setup logging
log_file_path = os.path.join(run_dir, "visualization_log.txt")
log_file = open(log_file_path, 'w')
log_file.write(f"Visualization Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write("="*80 + "\n\n")
log_file.flush()

def log_and_print(message):
    """Print message and write to log file"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

"""#### Dataset Classes"""

def load_npy_from_folder(folder_path):
    """Load .npy files from folder"""
    images = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        img = np.load(file_path)
        images.append(img)
    return np.stack(images, axis=0)

class VisualizationDataset(Dataset):
    """Dataset for visualization with segmentation data"""
    def __init__(self, low_dose_dir, full_dose_dir, seg_dir, transform_recon=None):
        self.transform_recon = transform_recon
        
        # Preload all data
        self.low_dose = load_npy_from_folder(low_dose_dir)
        self.full_dose = load_npy_from_folder(full_dose_dir)
        self.seg_maps = load_npy_from_folder(seg_dir)
        
        # Verify datasets match
        assert len(self.low_dose) == len(self.full_dose) == len(self.seg_maps), \
            "Mismatch in number of files across directories"
    
    def __len__(self):
        return len(self.low_dose)
    
    def __getitem__(self, idx):
        low_dose_img = self.low_dose[idx]
        full_dose_img = self.full_dose[idx]
        seg_map = self.seg_maps[idx]
        
        # Apply transformations
        if self.transform_recon:
            low_dose_img = self.transform_recon(low_dose_img)
            full_dose_img = self.transform_recon(full_dose_img)
        
        # Convert segmentation map to tensor
        seg_map_tensor = torch.from_numpy(seg_map.astype(np.int64))
        
        return low_dose_img, full_dose_img, seg_map_tensor

"""#### U-Net Model Architecture"""

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class U_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(U_Net, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        cat4 = torch.cat([u4, d4], dim=1)
        c4 = self.conv4(cat4)
        u3 = self.up3(c4)
        cat3 = torch.cat([u3, d3], dim=1)
        c3 = self.conv3(cat3)
        u2 = self.up2(c3)
        cat2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(cat2)
        u1 = self.up1(c2)
        cat1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(cat1)
        out = self.final_conv(c1)

        if self.out_channels == 1:
            return torch.sigmoid(out)
        else:
            return out

"""#### Reconstruction Functions"""

def _process_single_fbp(args):
    """Process a single image with FBP reconstruction"""
    low_dose_np, theta = args
    sinogram = radon(low_dose_np, theta=theta, circle=False)
    reconstruction = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
    
    # Resize if necessary
    if reconstruction.shape != (512, 512):
        scale_h = 512 / reconstruction.shape[0]
        scale_w = 512 / reconstruction.shape[1]
        reconstruction = zoom(reconstruction, (scale_h, scale_w), order=1)
    
    return reconstruction

def _process_single_bm3d(args):
    """Process a single image with BM3D denoising"""
    low_dose_np, sigma_psd = args
    denoised = bm3d.bm3d(low_dose_np, sigma_psd=sigma_psd)
    return denoised

def apply_fbp_reconstruction_parallel(low_dose_batch, theta=CONFIG['theta']):
    """Apply FBP reconstruction to a batch of low-dose images using parallel processing"""
    if theta is None:
        theta = np.linspace(0., 180., max(512, 512), endpoint=False)
    
    batch_size = low_dose_batch.shape[0]
    
    # Convert batch to numpy arrays
    low_dose_np_list = []
    for i in range(batch_size):
        low_dose_np = low_dose_batch[i].squeeze(0).cpu().numpy()  # Shape: (H, W)
        low_dose_np_list.append(low_dose_np)
    
    # Prepare arguments for parallel processing
    process_args = [(low_dose_np, theta) for low_dose_np in low_dose_np_list]
    
    # Process in parallel
    with Pool(processes=CONFIG['n_processes']) as pool:
        fbp_reconstructions = pool.map(_process_single_fbp, process_args)
    
    fbp_reconstructions = np.stack(fbp_reconstructions, axis=0)  # Shape: (B, H, W)
    
    # Convert back to tensor
    fbp_tensor = torch.from_numpy(fbp_reconstructions.astype(np.float32)).unsqueeze(1).to(device)
    return fbp_tensor

def apply_bm3d_denoising_parallel(low_dose_batch, sigma_psd=CONFIG['sigma_psd']):
    """Apply BM3D denoising to a batch of low-dose images using parallel processing"""
    batch_size = low_dose_batch.shape[0]
    
    # Convert batch to numpy arrays
    low_dose_np_list = []
    for i in range(batch_size):
        low_dose_np = low_dose_batch[i].squeeze(0).cpu().numpy()  # Shape: (H, W)
        low_dose_np_list.append(low_dose_np)
    
    # Prepare arguments for parallel processing
    process_args = [(low_dose_np, sigma_psd) for low_dose_np in low_dose_np_list]
    
    # Process in parallel
    with Pool(processes=CONFIG['n_processes']) as pool:
        bm3d_denoised = pool.map(_process_single_bm3d, process_args)
    
    bm3d_denoised = np.stack(bm3d_denoised, axis=0)  # Shape: (B, H, W)
    
    # Convert back to tensor
    bm3d_tensor = torch.from_numpy(bm3d_denoised.astype(np.float32)).unsqueeze(1).to(device)
    return bm3d_tensor

"""#### Metric Calculations"""

def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    pred_img = pred_np[0, 0]  # Single image
    target_img = target_np[0, 0]
    data_range = target_img.max() - target_img.min()
    
    return peak_signal_noise_ratio(target_img, pred_img, data_range=data_range)

def calculate_ssim(pred, target):
    """Calculate SSIM between prediction and target"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    pred_img = pred_np[0, 0]  # Single image
    target_img = target_np[0, 0]
    data_range = target_img.max() - target_img.min()
    
    return structural_similarity(target_img, pred_img, data_range=data_range)

def calculate_dice(pred, target, num_classes=3, eps=1e-6):
    """Calculate Dice coefficient for segmentation"""
    pred_labels = torch.argmax(pred, dim=1)  # [B, H, W]
    
    if target.dim() == 4 and target.shape[1] == num_classes:
        target_labels = torch.argmax(target, dim=1)
    else:
        target_labels = target
    
    # One-hot encode
    pred_oh = torch.nn.functional.one_hot(pred_labels, num_classes).permute(0, 3, 1, 2).float()
    target_oh = torch.nn.functional.one_hot(target_labels.long(), num_classes).permute(0, 3, 1, 2).float()
    
    # Dice computation
    intersection = (pred_oh * target_oh).sum(dim=[2, 3])
    pred_sum = pred_oh.sum(dim=[2, 3])
    target_sum = target_oh.sum(dim=[2, 3])
    dice_per_case = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    mean_dice_per_case = dice_per_case.mean(dim=1)
    
    return mean_dice_per_case[0].item()  # Return single value

"""#### Model Loading"""

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_and_print(f"Using device: {device}")

# Load all models
log_and_print("Loading pre-trained models...")

# Segmentation model
seg_model = U_Net(in_channels=1, out_channels=CONFIG['num_classes']).to(device)
seg_model.load_state_dict(torch.load(CONFIG['seg_model_path'], map_location=device))
seg_model.eval()

# Base U-Net reconstruction model
base_recon_model = U_Net(in_channels=1, out_channels=1).to(device)
base_recon_model.load_state_dict(torch.load(CONFIG['base_unet_model_path'], map_location=device))
base_recon_model.eval()

# Joint training models
joint_recon_models = {}
for alpha, path in CONFIG['joint_models'].items():
    model = U_Net(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    joint_recon_models[alpha] = model

# Task-adaptive models
task_adaptive_models = {}
for alpha, path in CONFIG['task_adaptive_models'].items():
    model = U_Net(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    task_adaptive_models[alpha] = model

log_and_print(f"All models loaded successfully")

"""#### Dataset Setup"""

# Create dataset
transform_recon = transforms.ToTensor()
dataset = VisualizationDataset(
    CONFIG['low_dose_dir'],
    CONFIG['full_dose_dir'],
    CONFIG['seg_dir'],
    transform_recon=transform_recon
)

# Split dataset (same as training)
torch.manual_seed(1773)
generator = torch.Generator().manual_seed(1773)
train_size = int(CONFIG['train_test_split'] * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

log_and_print(f"Test dataset contains {len(test_dataset)} samples")

# Select random samples for visualization
random.seed(CONFIG['vis_random_seed'])
sample_indices = random.sample(range(len(test_dataset)), CONFIG['num_samples'])
log_and_print(f"Selected {CONFIG['num_samples']} random samples for visualization")

"""#### Visualization Functions"""

def _create_reconstruction_grid_plot_worker(args):
    """Worker function for parallel plot creation"""
    sample_idx, reconstructions, metrics, save_path = args
    
    # Ensure the matplotlib backend is non-interactive for multiprocessing
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Define method names and order
    method_names = [
        'Low-Dose', 'FBP', 'BM3D', 'Base U-Net',
        'Joint α=0.1', 'Joint α=0.5', 'Joint α=0.9',
        'Task-Adaptive α=0.1', 'Task-Adaptive α=0.5', 'Task-Adaptive α=0.9',
        'Ground Truth'
    ]
    
    method_keys = [
        'low_dose', 'fbp', 'bm3d', 'base_unet',
        'joint_alpha01', 'joint_alpha05', 'joint_alpha09',
        'task_adaptive_alpha01', 'task_adaptive_alpha05', 'task_adaptive_alpha09',
        'ground_truth'
    ]
    
    # Create 3x4 grid (11 images)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Reconstruction Comparison - Sample {sample_idx}', fontsize=20, fontweight='bold')
    
    # Hide the last subplot (12th position)
    axes[2, 3].axis('off')
    
    for idx, (method_name, method_key) in enumerate(zip(method_names, method_keys)):
        if idx >= 11:  # Skip after 11 images
            break
            
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        if method_key in reconstructions:
            # Display image
            img = reconstructions[method_key]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            
            # Add metrics to title if available
            title = method_name
            if method_key in metrics and method_key != 'ground_truth':
                psnr = metrics[method_key]['psnr']
                ssim = metrics[method_key]['ssim']
                dice = metrics[method_key]['dice']
                title += f'\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}, Dice: {dice:.3f}'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
        
        ax.axis('off')
    
    # All 12 positions should be filled now
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def create_reconstruction_grid_plot(sample_idx, reconstructions, metrics, save_path):
    """Create grid plot showing all reconstruction methods"""
    args = (sample_idx, reconstructions, metrics, save_path)
    _create_reconstruction_grid_plot_worker(args)
    
    # Define method names and order
    method_names = [
        'Low-Dose', 'FBP', 'BM3D', 'Base U-Net',
        'Joint α=0.1', 'Joint α=0.5', 'Joint α=0.9',
        'Task-Adaptive α=0.1', 'Task-Adaptive α=0.5', 'Task-Adaptive α=0.9',
        'Ground Truth'
    ]
    
    method_keys = [
        'low_dose', 'fbp', 'bm3d', 'base_unet',
        'joint_alpha01', 'joint_alpha05', 'joint_alpha09',
        'task_adaptive_alpha01', 'task_adaptive_alpha05', 'task_adaptive_alpha09',
        'ground_truth'
    ]
    
    # Create 3x4 grid (11 images + 1 empty)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Reconstruction Comparison - Sample {sample_idx}', fontsize=20, fontweight='bold')
    
    for idx, (method_name, method_key) in enumerate(zip(method_names, method_keys)):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        if method_key in reconstructions:
            # Display image
            img = reconstructions[method_key]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            
            # Add metrics to title if available
            title = method_name
            if method_key in metrics and method_key != 'ground_truth':
                psnr = metrics[method_key]['psnr']
                ssim = metrics[method_key]['ssim']
                dice = metrics[method_key]['dice']
                title += f'\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}, Dice: {dice:.3f}'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
        else:
            ax.axis('off')  # Hide unused subplot
        
        ax.axis('off')
    
    # All 12 positions should be filled now
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _create_segmentation_overlay_plot_worker(args):
    """Worker function for parallel segmentation plot creation"""
    sample_idx, reconstructions, segmentations, gt_seg, metrics, save_path = args
    
    # Ensure the matplotlib backend is non-interactive for multiprocessing
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Updated ordering to include full-dose segmentation before ground truth
    method_names = [
        'Low-Dose', 'FBP', 'BM3D', 'Base U-Net',
        'Joint α=0.1', 'Joint α=0.5', 'Joint α=0.9',
        'Task-Adaptive α=0.1', 'Task-Adaptive α=0.5', 'Task-Adaptive α=0.9',
        'Full-Dose Seg', 'Ground Truth'
    ]

    method_keys = [
        'low_dose', 'fbp', 'bm3d', 'base_unet',
        'joint_alpha01', 'joint_alpha05', 'joint_alpha09',
        'task_adaptive_alpha01', 'task_adaptive_alpha05', 'task_adaptive_alpha09',
        'full_dose_seg', 'ground_truth'
    ]

    # Create 3x4 grid (12 images)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Segmentation Overlay Comparison - Sample {sample_idx}', fontsize=20, fontweight='bold')

    for idx, (method_name, method_key) in enumerate(zip(method_names, method_keys)):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        if method_key in reconstructions:
            img = reconstructions[method_key]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)

            # Determine segmentation to overlay
            if method_key == 'ground_truth':
                seg = gt_seg
            elif method_key == 'full_dose_seg':
                seg = segmentations['full_dose_seg']
            else:
                seg = segmentations[method_key]

            # Overlay segmentation with transparency
            ax.imshow(seg, cmap='jet', alpha=0.4, vmin=0, vmax=2)
            
            # Add metrics to title if available
            title = method_name
            if method_key in metrics and method_key not in ['ground_truth', 'full_dose_seg']:
                psnr = metrics[method_key]['psnr']
                ssim = metrics[method_key]['ssim']
                dice = metrics[method_key]['dice']
                title += f'\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}, Dice: {dice:.3f}'
            elif method_key == 'full_dose_seg' and method_key in metrics:
                dice = metrics[method_key]['dice']
                title += f'\nPSNR: N/A, SSIM: N/A, Dice: {dice:.3f}'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
        else:
            ax.axis('off')

        ax.axis('off')

    # All 12 positions should be filled now

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def create_segmentation_overlay_plot(sample_idx, reconstructions, segmentations, gt_seg, metrics, save_path):
    """Create segmentation overlay plot showing ALL reconstruction methods in the same 3x4 grid"""
    args = (sample_idx, reconstructions, segmentations, gt_seg, metrics, save_path)
    _create_segmentation_overlay_plot_worker(args)

    # Updated ordering to include full-dose segmentation before ground truth
    method_names = [
        'Low-Dose', 'FBP', 'BM3D', 'Base U-Net',
        'Joint α=0.1', 'Joint α=0.5', 'Joint α=0.9',
        'Task-Adaptive α=0.1', 'Task-Adaptive α=0.5', 'Task-Adaptive α=0.9',
        'Full-Dose Seg', 'Ground Truth'
    ]

    method_keys = [
        'low_dose', 'fbp', 'bm3d', 'base_unet',
        'joint_alpha01', 'joint_alpha05', 'joint_alpha09',
        'task_adaptive_alpha01', 'task_adaptive_alpha05', 'task_adaptive_alpha09',
        'full_dose_seg', 'ground_truth'
    ]

    # Create 3x4 grid (12 images)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Segmentation Overlay Comparison - Sample {sample_idx}', fontsize=20, fontweight='bold')

    for idx, (method_name, method_key) in enumerate(zip(method_names, method_keys)):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        if method_key in reconstructions:
            img = reconstructions[method_key]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)

            # Determine segmentation to overlay
            if method_key == 'ground_truth':
                seg = gt_seg
            elif method_key == 'full_dose_seg':
                seg = segmentations['full_dose_seg']
            else:
                seg = segmentations[method_key]

            # Overlay segmentation with transparency
            ax.imshow(seg, cmap='jet', alpha=0.4, vmin=0, vmax=2)
            
            # Add metrics to title if available
            title = method_name
            if method_key in metrics and method_key not in ['ground_truth', 'full_dose_seg']:
                psnr = metrics[method_key]['psnr']
                ssim = metrics[method_key]['ssim']
                dice = metrics[method_key]['dice']
                title += f'\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}, Dice: {dice:.3f}'
            elif method_key == 'full_dose_seg' and method_key in metrics:
                dice = metrics[method_key]['dice']
                title += f'\nPSNR: N/A, SSIM: N/A, Dice: {dice:.3f}'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
        else:
            ax.axis('off')

        ax.axis('off')

    # All 12 positions should be filled now

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

"""#### Main Visualization Loop"""

def process_batch(batch_indices, test_dataset, device):
    """Process a batch of samples and return their reconstructions, metrics, and segmentations"""
    batch_data = [test_dataset[idx] for idx in batch_indices]
    low_dose_batch = torch.stack([data[0] for data in batch_data]).to(device)
    full_dose_batch = torch.stack([data[1] for data in batch_data]).to(device)
    seg_map_batch = torch.stack([data[2] for data in batch_data]).to(device)
    
    batch_reconstructions = []
    batch_metrics = []
    batch_segmentations = []
    
    with torch.no_grad():
        # Generate reconstructions for the entire batch
        reconstructions = {
            'low_dose': low_dose_batch.cpu().numpy(),
            'fbp': apply_fbp_reconstruction_parallel(low_dose_batch).cpu().numpy(),
            'bm3d': apply_bm3d_denoising_parallel(low_dose_batch).cpu().numpy(),
            'base_unet': base_recon_model(low_dose_batch).cpu().numpy()
        }
        
        # Joint training reconstructions
        for alpha, model in joint_recon_models.items():
            reconstructions[f'joint_{alpha}'] = model(low_dose_batch).cpu().numpy()
        
        # Task-adaptive reconstructions
        for alpha, model in task_adaptive_models.items():
            reconstructions[f'task_adaptive_{alpha}'] = model(low_dose_batch).cpu().numpy()
        
        # Ground truth
        reconstructions['ground_truth'] = full_dose_batch.cpu().numpy()
        
        # Calculate metrics and segmentations for each sample in batch
        for i in range(len(batch_indices)):
            sample_reconstructions = {}
            sample_metrics = {}
            sample_segmentations = {}
            
            # Extract single sample reconstructions
            for method_key, recon_batch in reconstructions.items():
                sample_reconstructions[method_key] = recon_batch[i, 0]
            
            # Add full-dose ground truth image for the full-dose segmentation display
            sample_reconstructions['full_dose_seg'] = full_dose_batch[i, 0].cpu().numpy()
            
            # Calculate metrics and segmentations
            for method_key in ['low_dose', 'fbp', 'bm3d', 'base_unet'] + \
                             [f'joint_{alpha}' for alpha in joint_recon_models.keys()] + \
                             [f'task_adaptive_{alpha}' for alpha in task_adaptive_models.keys()]:
                
                if method_key in reconstructions and method_key != 'ground_truth':
                    recon_tensor = torch.from_numpy(reconstructions[method_key][i:i+1]).to(device)
                    full_dose_single = full_dose_batch[i:i+1]
                    seg_map_single = seg_map_batch[i:i+1]
                    
                    # Calculate metrics
                    psnr = calculate_psnr(recon_tensor, full_dose_single)
                    ssim = calculate_ssim(recon_tensor, full_dose_single)
                    
                    # Calculate segmentation
                    seg_pred = seg_model(recon_tensor)
                    dice = calculate_dice(seg_pred, seg_map_single, num_classes=CONFIG['num_classes'])
                    
                    # Store metrics
                    sample_metrics[method_key] = {
                        'psnr': psnr,
                        'ssim': ssim,
                        'dice': dice
                    }
                    
                    # Store segmentation
                    seg_pred_labels = torch.argmax(seg_pred, dim=1).squeeze(0).cpu().numpy()
                    sample_segmentations[method_key] = seg_pred_labels
            
            # Calculate full-dose segmentation (pretrained segmentation model on full-dose ground truth)
            full_dose_single = full_dose_batch[i:i+1]
            seg_map_single = seg_map_batch[i:i+1]
            
            # Calculate segmentation for full-dose ground truth
            seg_pred_full_dose = seg_model(full_dose_single)
            dice_full_dose = calculate_dice(seg_pred_full_dose, seg_map_single, num_classes=CONFIG['num_classes'])
            
            # Store metrics for full-dose (N/A for PSNR and SSIM)
            sample_metrics['full_dose_seg'] = {
                'psnr': float('nan'),  # N/A
                'ssim': float('nan'),  # N/A
                'dice': dice_full_dose
            }
            
            # Store segmentation
            seg_pred_full_dose_labels = torch.argmax(seg_pred_full_dose, dim=1).squeeze(0).cpu().numpy()
            sample_segmentations['full_dose_seg'] = seg_pred_full_dose_labels
            
            batch_reconstructions.append(sample_reconstructions)
            batch_metrics.append(sample_metrics)
            batch_segmentations.append(sample_segmentations)
    
    return batch_reconstructions, batch_metrics, batch_segmentations

def generate_visualizations():
    """Generate visualizations for selected samples using batched processing"""
    
    log_and_print("Starting visualization generation...")
    
    # Process samples in batches
    batch_size = min(CONFIG['batch_size'], CONFIG['num_samples'])
    num_batches = (len(sample_indices) + batch_size - 1) // batch_size
    total_plots = CONFIG['num_samples'] * 2  # Each sample has 2 plots
    
    progress_bar = tqdm(total=total_plots, desc="Generating plots", unit="plot")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sample_indices))
        batch_indices = sample_indices[start_idx:end_idx]
        current_batch_size = len(batch_indices)
        
        try:
            # Process batch
            batch_reconstructions, batch_metrics, batch_segmentations = process_batch(batch_indices, test_dataset, device)
            
            # Prepare plot arguments for all samples in the current batch
            recon_args = []
            seg_args = []
            
            for i, (reconstructions, metrics, segmentations) in enumerate(zip(batch_reconstructions, batch_metrics, batch_segmentations)):
                curr_idx = start_idx + i
                reconstruction_plot_path = os.path.join(run_dir, f"sample_{curr_idx+1:02d}_reconstructions.png")
                segmentation_plot_path = os.path.join(run_dir, f"sample_{curr_idx+1:02d}_segmentations.png")
                
                # Get ground truth segmentation for this specific sample
                _, _, curr_seg_map = test_dataset[batch_indices[i]]
                curr_gt_seg_np = curr_seg_map.numpy()
                
                # Arguments for reconstruction plot
                recon_args.append((curr_idx+1, reconstructions, metrics, reconstruction_plot_path))
                
                # Arguments for segmentation plot
                seg_args.append((curr_idx+1, reconstructions, segmentations, curr_gt_seg_np, metrics, segmentation_plot_path))
            
            # Process plots in parallel with dedicated workers
            with Pool(processes=min(len(recon_args) + len(seg_args), CONFIG['plot_workers'])) as pool:
                # Run reconstruction plots
                pool.map(_create_reconstruction_grid_plot_worker, recon_args, chunksize=max(1, len(recon_args)//CONFIG['plot_workers']))
                # Run segmentation plots  
                pool.map(_create_segmentation_overlay_plot_worker, seg_args, chunksize=max(1, len(seg_args)//CONFIG['plot_workers']))
            
            # Update progress bar after plot generation
            plots_in_batch = current_batch_size * 2  # 2 plots per sample
            progress_bar.update(plots_in_batch)
            progress_bar.set_postfix({
                'Batch': f"{batch_idx + 1}/{num_batches}",
                'Completed': f"{min((batch_idx + 1) * batch_size, CONFIG['num_samples'])}/{CONFIG['num_samples']} samples"
            }, refresh=True)
            
        except Exception as e:
            log_and_print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    
    progress_bar.close()
    log_and_print(f"Visualization generation completed!")
    log_and_print(f"Generated {CONFIG['num_samples'] * 2} plots in total")

"""#### Main Execution"""

if __name__ == "__main__":
    try:
        # Generate visualizations
        generate_visualizations()
        
        # Create summary
        summary_info = {
            'timestamp': timestamp,
            'num_samples': CONFIG['num_samples'],
            'sample_indices': sample_indices,
            'plots_per_sample': 2,
            'total_plots_generated': CONFIG['num_samples'] * 2,
            'output_directory': run_dir,
            'random_seed': CONFIG['vis_random_seed'],
            'config': CONFIG
        }
        
        # Save summary
        summary_path = os.path.join(run_dir, 'visualization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        log_and_print(f"\nVisualization completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_and_print(f"Summary saved to: {summary_path}")
        log_and_print(f"All visualizations saved to: {run_dir}")
        
    except Exception as e:
        log_and_print(f"Error during visualization: {str(e)}")
        import traceback
        log_and_print(f"Traceback: {traceback.format_exc()}")
        
    finally:
        # Close log file
        log_file.close()
