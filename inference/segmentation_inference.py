"""#### Install Libraries"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

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

"""#### CONFIG"""

CONFIG = {
    # Inference parameters
    'batch_size': 64,
    'num_workers': 64,
    'train_test_split': 0.8,  # 80% training, 20% testing
    'num_classes': 3,  # Number of segmentation classes
    
    # BM3D parameters
    'sigma_psd': 0.075,
    
    # FBP parameters
    'theta': None,  # None for auto-calculation
    
    # Parallelization parameters
    'n_processes': cpu_count(),  # Use all available CPU cores
    
    # Dataset directories
    'low_dose_dir': '.../dataset/low_dose',
    'full_dose_dir': '.../dataset/full_dose',
    'seg_dir': '.../dataset/segmentation_slices',
    
    # Model weights paths
    'seg_model_path': '.../model-weights/seg_unet.pth',
    'base_unet_model_path': '.../model-weights/base_unet_reconstruction.pth',
    'joint_models': {
        'alpha01': '.../model-weights/joint_recon_unet_alpha01.pth',
        'alpha05': '.../model-weights/joint_recon_unet_alpha05.pth',
        'alpha09': '.../model-weights/joint_recon_unet_alpha09.pth'
    },
    'task_adaptive_models': {
        'alpha01': '.../model-weights/task_adaptive_unet_alpha01.pth',
        'alpha05': '.../model-weights/task_adaptive_unet_alpha05.pth',
        'alpha09': '.../model-weights/task_adaptive_unet_alpha09.pth'
    },
    
    # Output directory for plots and logs
    'output_dir': '.../inference-results'
}

"""#### Output Directory and Logging Setup"""

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Use output directory directly (overwrite previous outputs)
run_dir = CONFIG['output_dir']
os.makedirs(run_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # For logging purposes only

print(f"Segmentation inference outputs will be saved to: {run_dir}")

# Save configuration to run directory
config_path = os.path.join(run_dir, "config.json")
config_to_save = CONFIG.copy()
with open(config_path, 'w') as f:
    json.dump(config_to_save, f, indent=2)

# Initialize log file
log_file_path = os.path.join(run_dir, "segmentation_inference_log.txt")
log_file = open(log_file_path, 'w')
log_file.write(f"Segmentation Inference Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write("="*80 + "\n")
log_file.write(f"Configuration:\n")
for key, value in config_to_save.items():
    log_file.write(f"  {key}: {value}\n")
log_file.write("="*80 + "\n\n")
log_file.flush()

def log_and_print(message):
    """Print message and write to log file"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

"""#### Dataset"""

# Function to preload .npy files from a folder
def load_npy_from_folder(folder_path):
    images = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        img = np.load(file_path)  # Shape: (H, W), already normalized to [0, 1] during dataset creation
        images.append(img)
    return np.stack(images, axis=0)  # Shape: (N, H, W)

# Unified Dataset for Segmentation Inference
class SegmentationInferenceDataset(Dataset):
    def __init__(self, low_dose_dir, full_dose_dir, seg_dir, transform_recon=None):
        self.transform_recon = transform_recon

        # Preload all data (already normalized to [0, 1] during dataset creation)
        self.low_dose = load_npy_from_folder(low_dose_dir)    # Shape: (N, H, W)
        self.full_dose = load_npy_from_folder(full_dose_dir)  # Shape: (N, H, W)
        self.seg_maps = load_npy_from_folder(seg_dir)         # Shape: (N, H, W)

        # Verify that all datasets have the same number of samples
        assert len(self.low_dose) == len(self.full_dose) == len(self.seg_maps), \
            "Mismatch in number of files across directories"

    def __len__(self):
        return len(self.low_dose)

    def __getitem__(self, idx):
        # Get images (already normalized to [0, 1]) and segmentation map
        low_dose_img = self.low_dose[idx]   # Shape: (H, W)
        full_dose_img = self.full_dose[idx] # Shape: (H, W)
        seg_map = self.seg_maps[idx]        # Shape: (H, W)

        # Apply transformations to images (e.g., ToTensor)
        if self.transform_recon:
            low_dose_img = self.transform_recon(low_dose_img)   # Shape: (1, H, W)
            full_dose_img = self.transform_recon(full_dose_img) # Shape: (1, H, W)

        # Convert segmentation map to int64 tensor without altering class labels
        seg_map_tensor = torch.from_numpy(seg_map.astype(np.int64))  # [H, W]

        return low_dose_img, full_dose_img, seg_map_tensor

# Define transform for images
transform_recon = transforms.ToTensor()  # Images are already normalized to [0, 1] during dataset creation

# Instantiate the dataset
dataset = SegmentationInferenceDataset(
    CONFIG['low_dose_dir'],
    CONFIG['full_dose_dir'],
    CONFIG['seg_dir'],
    transform_recon=transform_recon
)

# Set random seed for reproducibility
torch.manual_seed(1773)
generator = torch.Generator().manual_seed(1773)

# Split dataset into training and testing sets
train_size = int(CONFIG['train_test_split'] * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# Create test DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers']
)

log_and_print(f"Dataset loaded with {len(dataset)} total samples")
log_and_print(f"Test dataset contains {len(test_dataset)} samples")

"""#### Comprehensive Metrics"""

def create_circular_mask(h, w, center=None, radius=None):
    """Create circular mask for masked metrics"""
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(w, h) / 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.float32)

def calculate_psnr(pred, target, data_range=None):
    """Calculate PSNR for a batch of images. Returns list of PSNR values."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    psnr_values = []
    for i in range(pred_np.shape[0]):
        # Single-channel images, index with [0] to get [H, W]
        pred_img = pred_np[i, 0]
        target_img = target_np[i, 0]
        if data_range is None:
            data_range = target_img.max() - target_img.min()
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=data_range)
        psnr_values.append(psnr)
    return psnr_values

def calculate_ssim(pred, target, data_range=None):
    """Calculate SSIM for a batch of images. Returns list of SSIM values."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_values = []
    for i in range(pred_np.shape[0]):
        # Single-channel images, index with [0] to get [H, W]
        pred_img = pred_np[i, 0]
        target_img = target_np[i, 0]
        if data_range is None:
            data_range = target_img.max() - target_img.min()
        ssim = structural_similarity(target_img, pred_img, data_range=data_range)
        ssim_values.append(ssim)
    return ssim_values

def calculate_masked_psnr(pred, target, mask, data_range=None):
    """Calculate PSNR within masked region. Returns list of PSNR values."""
    mask_tensor = torch.from_numpy(mask).to(pred.device).view(1, 1, 512, 512)
    masked_pred = pred * mask_tensor
    masked_target = target * mask_tensor
    return calculate_psnr(masked_pred, masked_target, data_range)

def calculate_masked_ssim(pred, target, mask, data_range=None):
    """Calculate SSIM within masked region. Returns list of SSIM values."""
    mask_tensor = torch.from_numpy(mask).to(pred.device).view(1, 1, 512, 512)
    masked_pred = pred * mask_tensor
    masked_target = target * mask_tensor
    return calculate_ssim(masked_pred, masked_target, data_range)

def calculate_dice(pred, target, num_classes=3, threshold=0.5, eps=1e-6, mask=None):
    """Calculate Dice coefficient using argmax (multi-class) or threshold (binary). Returns [B]."""
    if num_classes > 1:
        # Convert predictions to class labels via argmax
        pred_labels = torch.argmax(pred, dim=1)  # [B, H, W]

        # Handle target formatting
        if target.dim() == 4 and target.shape[1] == num_classes:
            target_labels = torch.argmax(target, dim=1)
        else:
            target_labels = target

        # One-hot encode back to [B, C, H, W]
        pred = torch.nn.functional.one_hot(pred_labels, num_classes).permute(0, 3, 1, 2).float()
        target = torch.nn.functional.one_hot(target_labels.long(), num_classes).permute(0, 3, 1, 2).float()
    else:
        # Binary case falls back to threshold approach
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        if target.dim() == 3:
            target = target.unsqueeze(1).float()

    # Apply mask if provided
    if mask is not None:
        mask_tensor = torch.from_numpy(mask).to(pred.device).view(1, 1, 512, 512)
        mask_tensor = mask_tensor.expand(-1, pred.size(1), -1, -1)
        pred = pred * mask_tensor
        target = target * mask_tensor

    # Dice computation
    intersection = (pred * target).sum(dim=[2, 3])
    pred_sum = pred.sum(dim=[2, 3])
    target_sum = target.sum(dim=[2, 3])
    dice_per_case = (2 * intersection + eps) / (pred_sum + target_sum + eps)

    # EXCLUDE BACKGROUND (channel 0) WHEN AVERAGING MULTI-CLASS DICE
    if num_classes > 1 and dice_per_case.shape[1] > 1:
        foreground_dice = dice_per_case[:, 1:]
        pred_sum_fg = pred_sum[:, 1:]
        target_sum_fg = target_sum[:, 1:]

        valid = (pred_sum_fg + target_sum_fg) > 0  # [B, C-1] boolean mask
        if valid.any():
            valid_counts = valid.sum(dim=1).clamp(min=1)  # per-case valid class count
            mean_dice_per_case = (foreground_dice * valid.float()).sum(dim=1) / valid_counts
        else:
            mean_dice_per_case = torch.zeros(dice_per_case.shape[0], device=pred.device)
    else:
        mean_dice_per_case = dice_per_case.mean(dim=1)

    return mean_dice_per_case



"""#### U-Net Model"""

# U-Net components
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
            return torch.sigmoid(out)  # Output normalized to [0, 1] for reconstruction
        else:
            return out  # Raw logits for segmentation

"""#### Setup"""

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_and_print(f"Using device: {device}")

# Load pre-trained models
log_and_print("Loading pre-trained models...")

# Load Segmentation model
seg_model = U_Net(in_channels=1, out_channels=CONFIG['num_classes']).to(device)
seg_model.load_state_dict(torch.load(CONFIG['seg_model_path'], map_location=device))
seg_model.eval()
log_and_print(f"Segmentation model loaded from: {CONFIG['seg_model_path']}")

# Load Base U-Net reconstruction model
base_recon_model = U_Net(in_channels=1, out_channels=1).to(device)
base_recon_model.load_state_dict(torch.load(CONFIG['base_unet_model_path'], map_location=device))
base_recon_model.eval()
log_and_print(f"Base U-Net reconstruction model loaded from: {CONFIG['base_unet_model_path']}")

# Load Joint training reconstruction models (all alphas)
joint_recon_models = {}
for alpha, path in CONFIG['joint_models'].items():
    model = U_Net(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    joint_recon_models[alpha] = model
    log_and_print(f"Joint training reconstruction model ({alpha}) loaded from: {path}")

# Load Task-adaptive reconstruction models (all alphas)
task_adaptive_models = {}
for alpha, path in CONFIG['task_adaptive_models'].items():
    model = U_Net(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    task_adaptive_models[alpha] = model
    log_and_print(f"Task-adaptive reconstruction model ({alpha}) loaded from: {path}")

# Create circular mask for masked metrics
mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)
log_and_print(f"Circular mask created for masked metrics (radius: 256 pixels)")

"""#### Segmentation Inference Functions"""

def _process_single_fbp(args):
    """Process a single image with FBP reconstruction (for multiprocessing)"""
    low_dose_np, theta = args
    
    # Apply FBP
    sinogram = radon(low_dose_np, theta=theta, circle=False)
    reconstruction = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
    
    # Resize if necessary
    if reconstruction.shape != (512, 512):
        scale_h = 512 / reconstruction.shape[0]
        scale_w = 512 / reconstruction.shape[1]
        reconstruction = zoom(reconstruction, (scale_h, scale_w), order=1)
    
    return reconstruction

def _process_single_bm3d(args):
    """Process a single image with BM3D denoising (for multiprocessing)"""
    low_dose_np, sigma_psd = args
    
    # Apply BM3D
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

# Keep original functions for fallback/comparison
def apply_fbp_reconstruction(low_dose_batch, theta=CONFIG['theta']):
    """Apply FBP reconstruction to a batch of low-dose images (sequential version)"""
    if theta is None:
        theta = np.linspace(0., 180., max(512, 512), endpoint=False)
    
    batch_size = low_dose_batch.shape[0]
    fbp_reconstructions = []
    
    for i in range(batch_size):
        # Convert tensor to numpy
        low_dose_np = low_dose_batch[i].squeeze(0).cpu().numpy()  # Shape: (H, W)
        
        # Apply FBP
        sinogram = radon(low_dose_np, theta=theta, circle=False)
        reconstruction = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
        
        fbp_reconstructions.append(reconstruction)
    
    fbp_reconstructions = np.stack(fbp_reconstructions, axis=0)  # Shape: (B, H, W)
    
    # Resize if necessary
    if fbp_reconstructions.shape[-2:] != (512, 512):
        resized_reconstructions = []
        for i in range(batch_size):
            scale_h = 512 / fbp_reconstructions.shape[-2]
            scale_w = 512 / fbp_reconstructions.shape[-1]
            resized = zoom(fbp_reconstructions[i], (scale_h, scale_w), order=1)
            resized_reconstructions.append(resized)
        fbp_reconstructions = np.stack(resized_reconstructions, axis=0)
    
    # Convert back to tensor
    fbp_tensor = torch.from_numpy(fbp_reconstructions.astype(np.float32)).unsqueeze(1).to(device)
    return fbp_tensor

def apply_bm3d_denoising(low_dose_batch, sigma_psd=CONFIG['sigma_psd']):
    """Apply BM3D denoising to a batch of low-dose images (sequential version)"""
    batch_size = low_dose_batch.shape[0]
    bm3d_denoised = []
    
    for i in range(batch_size):
        # Convert tensor to numpy
        low_dose_np = low_dose_batch[i].squeeze(0).cpu().numpy()  # Shape: (H, W)
        
        # Apply BM3D
        denoised = bm3d.bm3d(low_dose_np, sigma_psd=sigma_psd)
        bm3d_denoised.append(denoised)
    
    bm3d_denoised = np.stack(bm3d_denoised, axis=0)  # Shape: (B, H, W)
    
    # Convert back to tensor
    bm3d_tensor = torch.from_numpy(bm3d_denoised.astype(np.float32)).unsqueeze(1).to(device)
    return bm3d_tensor

def evaluate_comprehensive_metrics():
    """Evaluate comprehensive metrics (PSNR, SSIM, Dice) for all reconstruction methods"""
    
    log_and_print(f"\nStarting comprehensive evaluation...")
    log_and_print(f"Processing {len(test_loader)} batches with batch size {CONFIG['batch_size']} ({len(test_loader.dataset)} total samples)")
    log_and_print(f"Using {CONFIG['n_processes']} CPU processes for FBP and BM3D parallelization")
    log_and_print(f"Metrics: PSNR, SSIM, Dice (both normal and masked)")
    
    # Initialize result storage with nested structure
    results = {}
    
    # Initialize all method combinations
    methods = ['low_dose', 'fbp', 'bm3d', 'base_unet']
    for alpha in CONFIG['joint_models'].keys():
        methods.append(f'joint_{alpha}')
    for alpha in CONFIG['task_adaptive_models'].keys():
        methods.append(f'task_adaptive_{alpha}')
    # Add full-dose ground truth for segmentation evaluation
    methods.append('full_dose_gt')
    
    for method in methods:
        results[method] = {
            'psnr': [], 'ssim': [], 'dice': [],
            'masked_psnr': [], 'masked_ssim': []
        }
    
    # Process test dataset
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Comprehensive Evaluation", unit="batch")
        for batch_idx, (low_dose, full_dose, seg_map) in enumerate(progress_bar):
            low_dose = low_dose.to(device)
            full_dose = full_dose.to(device)
            seg_map = seg_map.to(device)
            
            # Generate all reconstructions once per batch for efficiency
            reconstructions = {}
            
            # 1. Low-dose (no reconstruction)
            reconstructions['low_dose'] = low_dose
            
            # 2. FBP reconstruction
            reconstructions['fbp'] = apply_fbp_reconstruction_parallel(low_dose)
            
            # 3. BM3D denoising
            reconstructions['bm3d'] = apply_bm3d_denoising_parallel(low_dose)
            
            # 4. Base U-Net reconstruction
            reconstructions['base_unet'] = base_recon_model(low_dose)
            
            # 5. Joint training reconstructions (all alphas)
            for alpha, model in joint_recon_models.items():
                reconstructions[f'joint_{alpha}'] = model(low_dose)
            
            # 6. Task-adaptive reconstructions (all alphas)
            for alpha, model in task_adaptive_models.items():
                reconstructions[f'task_adaptive_{alpha}'] = model(low_dose)
            
            # Calculate metrics for each reconstruction method
            for method, recon_images in reconstructions.items():
                # PSNR and SSIM calculations (reconstruction metrics)
                psnr_scores = calculate_psnr(recon_images, full_dose)
                ssim_scores = calculate_ssim(recon_images, full_dose)
                masked_psnr_scores = calculate_masked_psnr(recon_images, full_dose, mask)
                masked_ssim_scores = calculate_masked_ssim(recon_images, full_dose, mask)
                
                # Dice calculation (segmentation metric)
                seg_pred = seg_model(recon_images)
                dice_scores = calculate_dice(seg_pred, seg_map, num_classes=CONFIG['num_classes'])
                
                # Store results
                results[method]['psnr'].extend(psnr_scores)
                results[method]['ssim'].extend(ssim_scores)
                results[method]['dice'].extend(dice_scores.cpu().numpy())
                results[method]['masked_psnr'].extend(masked_psnr_scores)
                results[method]['masked_ssim'].extend(masked_ssim_scores)
            
            # Calculate metrics for full-dose ground truth (segmentation only)
            seg_pred_full_dose = seg_model(full_dose)
            dice_scores_full_dose = calculate_dice(seg_pred_full_dose, seg_map, num_classes=CONFIG['num_classes'])
            
            # Store results for full-dose (N/A for PSNR and SSIM)
            results['full_dose_gt']['psnr'].extend([np.nan] * full_dose.shape[0])  # N/A values
            results['full_dose_gt']['ssim'].extend([np.nan] * full_dose.shape[0])  # N/A values
            results['full_dose_gt']['dice'].extend(dice_scores_full_dose.cpu().numpy())
            results['full_dose_gt']['masked_psnr'].extend([np.nan] * full_dose.shape[0])  # N/A values
            results['full_dose_gt']['masked_ssim'].extend([np.nan] * full_dose.shape[0])  # N/A values
            
            # Update progress bar
            progress_bar.set_postfix({
                'Batch': f"{batch_idx + 1}/{len(test_loader)}",
                'Methods': len(reconstructions)
            })
    
    return results

"""#### Results Analysis and Saving"""

def analyze_and_save_results(results):
    """Calculate statistics and save comprehensive results to JSON"""
    
    log_and_print("\n" + "="*80)
    log_and_print("COMPREHENSIVE EVALUATION RESULTS")
    log_and_print("="*80)
    
    # Calculate statistics for all metrics
    statistics = {}
    
    metrics = ['psnr', 'ssim', 'dice', 'masked_psnr', 'masked_ssim']
    
    for method, method_results in results.items():
        statistics[method] = {}
        method_display = method.replace('_', ' ').title()
        
        log_and_print(f"\n{method_display}:")
        
        for metric in metrics:
            values = np.array(method_results[metric])
            
            # Handle N/A values for full-dose ground truth PSNR/SSIM
            if method == 'full_dose_gt' and metric in ['psnr', 'ssim', 'masked_psnr', 'masked_ssim']:
                # All values should be NaN for these metrics
                statistics[method][metric] = {
                    'mean': 'N/A',
                    'std': 'N/A',
                    'count': len(values)
                }
                # Display results
                metric_display = metric.replace('_', ' ').upper()
                log_and_print(f"  {metric_display}: N/A")
            else:
                # Normal processing for valid metrics
                # Filter out NaN values for computation
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                else:
                    mean_val = np.nan
                    std_val = np.nan
                
                statistics[method][metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'count': len(values)
                }
                
                # Display results
                metric_display = metric.replace('_', ' ').upper()
                log_and_print(f"  {metric_display}: {mean_val:.4f} ± {std_val:.4f}")
        
        log_and_print(f"  Sample Count: {len(values)}")
    
    # Save detailed results to JSON
    results_summary = {
        'statistics': statistics,
        'raw_results': {
            method: {
                metric: [float(x) for x in values] 
                for metric, values in method_results.items()
            } 
            for method, method_results in results.items()
        },
        'config': config_to_save,
        'timestamp': timestamp,
        'evaluation_info': {
            'total_samples': len(list(results.values())[0]['psnr']),
            'methods_evaluated': list(results.keys()),
            'metrics_computed': metrics,
            'mask_radius': 256
        }
    }
    
    json_path = os.path.join(run_dir, 'comprehensive_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    log_and_print(f"Detailed JSON results saved to: {json_path}")
    
    # Save summary results to text file
    summary_path = os.path.join(run_dir, 'comprehensive_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Comprehensive Evaluation Results Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total samples evaluated: {len(list(results.values())[0]['psnr'])}\n")
        f.write(f"Methods evaluated: {len(results)}\n")
        f.write(f"Metrics computed: {', '.join(metrics)}\n")
        f.write(f"Mask radius: 256 pixels\n\n")
        
        for method, stats in statistics.items():
            method_name = method.replace('_', ' ').title()
            f.write(f"{method_name}:\n")
            for metric in metrics:
                metric_display = metric.replace('_', ' ').upper()
                mean_val = stats[metric]['mean']
                std_val = stats[metric]['std']
                if mean_val == 'N/A':
                    f.write(f"  {metric_display}: N/A\n")
                else:
                    f.write(f"  {metric_display}: {mean_val:.4f} ± {std_val:.4f}\n")
            f.write(f"  Sample Count: {stats[metrics[0]]['count']}\n\n")
    
    log_and_print(f"Summary results saved to: {summary_path}")
    
    return statistics

def create_results_visualization(statistics):
    """Create comprehensive visualizations of the results"""
    
    methods = list(statistics.keys())
    method_names = [method.replace('_', ' ').title() for method in methods]
    
    # Create a 2x3 subplot for all metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Evaluation Results Across Reconstruction Methods', fontsize=16, fontweight='bold')
    
    metrics = ['psnr', 'ssim', 'dice', 'masked_psnr', 'masked_ssim']
    metric_titles = ['PSNR (dB)', 'SSIM', 'Dice Score', 'Masked PSNR (dB)', 'Masked SSIM']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx // 3, idx % 3]
        
        # Filter methods based on metric type
        if metric in ['psnr', 'ssim', 'masked_psnr', 'masked_ssim']:
            # Exclude full_dose_gt from PSNR/SSIM plots
            filtered_methods = [m for m in methods if m != 'full_dose_gt']
            filtered_method_names = [method.replace('_', ' ').title() for method in filtered_methods]
        else:
            # Include all methods for Dice plots
            filtered_methods = methods
            filtered_method_names = [method.replace('_', ' ').title() for method in filtered_methods]
        
        means = []
        stds = []
        for method in filtered_methods:
            mean_val = statistics[method][metric]['mean']
            std_val = statistics[method][metric]['std']
            # Handle N/A values
            if mean_val == 'N/A':
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(mean_val)
                stds.append(std_val)
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, (m, s) in enumerate(zip(means, stds)) if not np.isnan(m)]
        if valid_indices:
            valid_means = [means[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            valid_names = [filtered_method_names[i] for i in valid_indices]
            
            bars = ax.bar(range(len(valid_means)), valid_means, yerr=valid_stds, capsize=3, alpha=0.8, 
                         color=[colors[i % len(colors)] for i in range(len(valid_means))])
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_xticks(range(len(valid_means)))
            ax.set_xticklabels(valid_names, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, valid_means, valid_stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
            
            # Set reasonable y-limits
            y_max = max([m + s for m, s in zip(valid_means, valid_stds)])
            ax.set_ylim(0, y_max * 1.1)
        else:
            # No valid data to plot
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Hide the last unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save comprehensive plot
    comprehensive_plot_path = os.path.join(run_dir, "comprehensive_evaluation_results.png")
    plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_and_print(f"Comprehensive visualization saved to: {comprehensive_plot_path}")
    
    # Create separate plots for easier reading
    for metric, title in zip(metrics, metric_titles):
        plt.figure(figsize=(12, 8))
        
        # Filter methods based on metric type
        if metric in ['psnr', 'ssim', 'masked_psnr', 'masked_ssim']:
            # Exclude full_dose_gt from PSNR/SSIM plots
            filtered_methods = [m for m in methods if m != 'full_dose_gt']
            filtered_method_names = [method.replace('_', ' ').title() for method in filtered_methods]
        else:
            # Include all methods for Dice plots
            filtered_methods = methods
            filtered_method_names = [method.replace('_', ' ').title() for method in filtered_methods]
        
        means = []
        stds = []
        for method in filtered_methods:
            mean_val = statistics[method][metric]['mean']
            std_val = statistics[method][metric]['std']
            # Handle N/A values
            if mean_val == 'N/A':
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(mean_val)
                stds.append(std_val)
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, (m, s) in enumerate(zip(means, stds)) if not np.isnan(m)]
        if valid_indices:
            valid_means = [means[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            valid_names = [filtered_method_names[i] for i in valid_indices]
            
            bars = plt.bar(valid_names, valid_means, yerr=valid_stds, capsize=5, alpha=0.8, 
                          color=[colors[i % len(colors)] for i in range(len(valid_means))])
            
            plt.title(f'{title} Across Reconstruction Methods', fontsize=16, fontweight='bold')
            plt.ylabel(title, fontsize=14)
            plt.xlabel('Reconstruction Method', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, valid_means, valid_stds):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            y_max = max([m + s for m, s in zip(valid_means, valid_stds)])
            plt.ylim(0, y_max * 1.1)
            
            plt.tight_layout()
            
            # Save individual plot
            metric_plot_path = os.path.join(run_dir, f"{metric}_comparison.png")
            plt.savefig(metric_plot_path, dpi=300, bbox_inches='tight')
        else:
            # Create empty plot with message
            plt.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{title} Across Reconstruction Methods', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save individual plot
            metric_plot_path = os.path.join(run_dir, f"{metric}_comparison.png")
            plt.savefig(metric_plot_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    log_and_print(f"Individual metric visualizations saved to: {run_dir}")

"""#### Main Execution"""

if __name__ == "__main__":
    try:
        # Run comprehensive evaluation
        results = evaluate_comprehensive_metrics()
        
        # Analyze and save results
        statistics = analyze_and_save_results(results)
        
        # Create visualizations
        create_results_visualization(statistics)
        
        # Log completion
        log_and_print(f"\nComprehensive evaluation completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_and_print(f"All outputs saved to: {run_dir}")
        
    except Exception as e:
        log_and_print(f"Error during comprehensive evaluation: {str(e)}")
        import traceback
        log_and_print(f"Traceback: {traceback.format_exc()}")
        
    finally:
        # Close log file
        log_file.close()


