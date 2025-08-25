"""#### Install Libraries"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.transform import radon, iradon
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
    'low_dose_dir': '/YEREL/dataset/low_dose',
    'full_dose_dir': '/YEREL/dataset/full_dose',
    'seg_dir': '/YEREL/dataset/segmentation_slices',
    
    # Model weights paths
    'seg_model_path': '/ari/users/nsefercioglu/model-weights/seg_unet.pth',
    'base_unet_model_path': '/ari/users/nsefercioglu/model-weights/base_unet_reconstruction.pth',
    'joint_recon_model_path': '/ari/users/nsefercioglu/model-weights/joint_recon_unet.pth',
    'task_adaptive_model_path': '/ari/users/nsefercioglu/model-weights/task_adaptive_unet.pth',
    
    # Output directory for plots and logs
    'output_dir': '/ari/users/nsefercioglu/inference-results'
}

"""#### Output Directory and Logging Setup"""

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Create unique run directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(CONFIG['output_dir'], f"segmentation_inference_run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

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

        # Convert segmentation map to int64
        seg_map = seg_map.astype(np.int64)

        return low_dose_img, full_dose_img, seg_map

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

"""#### Dice Metric"""

def calculate_dice(pred, target, num_classes=3, threshold=0.5, eps=1e-6):
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

    # Dice computation
    intersection = (pred * target).sum(dim=[2, 3])
    pred_sum = pred.sum(dim=[2, 3])
    target_sum = target.sum(dim=[2, 3])
    dice_per_case = (2 * intersection + eps) / (pred_sum + target_sum + eps)
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

# Load Joint training reconstruction model
joint_recon_model = U_Net(in_channels=1, out_channels=1).to(device)
joint_recon_model.load_state_dict(torch.load(CONFIG['joint_recon_model_path'], map_location=device))
joint_recon_model.eval()
log_and_print(f"Joint training reconstruction model loaded from: {CONFIG['joint_recon_model_path']}")

# Load Task-adaptive reconstruction model
task_adaptive_model = U_Net(in_channels=1, out_channels=1).to(device)
task_adaptive_model.load_state_dict(torch.load(CONFIG['task_adaptive_model_path'], map_location=device))
task_adaptive_model.eval()
log_and_print(f"Task-adaptive reconstruction model loaded from: {CONFIG['task_adaptive_model_path']}")

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

def evaluate_segmentation_on_reconstructed_images():
    """Evaluate segmentation performance on all reconstruction methods"""
    
    log_and_print(f"\nStarting segmentation inference on all reconstruction methods...")
    log_and_print(f"Processing {len(test_loader)} batches with batch size {CONFIG['batch_size']} ({len(test_loader.dataset)} total samples)")
    log_and_print(f"Using {CONFIG['n_processes']} CPU processes for FBP and BM3D parallelization")
    
    # Initialize result storage
    results = {
        'low_dose': [],
        'fbp': [],
        'bm3d': [],
        'base_unet': [],
        'joint_training': [],
        'task_adaptive': []
    }
    
    # Process test dataset
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Segmentation Inference", unit="batch")
        for batch_idx, (low_dose, full_dose, seg_map) in enumerate(progress_bar):
            low_dose = low_dose.to(device)
            full_dose = full_dose.to(device)
            seg_map = seg_map.to(device)
            
            # 1. Low-dose images segmentation
            seg_pred_low_dose = seg_model(low_dose)
            dice_low_dose = calculate_dice(seg_pred_low_dose, seg_map, num_classes=CONFIG['num_classes'])
            results['low_dose'].extend(dice_low_dose.cpu().numpy())
            
            # 2. FBP reconstructed images segmentation (parallel)
            fbp_reconstructed = apply_fbp_reconstruction_parallel(low_dose)
            seg_pred_fbp = seg_model(fbp_reconstructed)
            dice_fbp = calculate_dice(seg_pred_fbp, seg_map, num_classes=CONFIG['num_classes'])
            results['fbp'].extend(dice_fbp.cpu().numpy())
            
            # 3. BM3D denoised images segmentation (parallel)
            bm3d_denoised = apply_bm3d_denoising_parallel(low_dose)
            seg_pred_bm3d = seg_model(bm3d_denoised)
            dice_bm3d = calculate_dice(seg_pred_bm3d, seg_map, num_classes=CONFIG['num_classes'])
            results['bm3d'].extend(dice_bm3d.cpu().numpy())
            
            # 4. Base U-Net reconstructed images segmentation
            base_unet_reconstructed = base_recon_model(low_dose)
            seg_pred_base_unet = seg_model(base_unet_reconstructed)
            dice_base_unet = calculate_dice(seg_pred_base_unet, seg_map, num_classes=CONFIG['num_classes'])
            results['base_unet'].extend(dice_base_unet.cpu().numpy())
            
            # 5. Joint training reconstructed images segmentation
            joint_reconstructed = joint_recon_model(low_dose)
            seg_pred_joint = seg_model(joint_reconstructed)
            dice_joint = calculate_dice(seg_pred_joint, seg_map, num_classes=CONFIG['num_classes'])
            results['joint_training'].extend(dice_joint.cpu().numpy())
            
            # 6. Task-adaptive reconstructed images segmentation
            task_adaptive_reconstructed = task_adaptive_model(low_dose)
            seg_pred_task_adaptive = seg_model(task_adaptive_reconstructed)
            dice_task_adaptive = calculate_dice(seg_pred_task_adaptive, seg_map, num_classes=CONFIG['num_classes'])
            results['task_adaptive'].extend(dice_task_adaptive.cpu().numpy())
            
            # Update progress bar with current batch info
            progress_bar.set_postfix({
                'Batch': f"{batch_idx + 1}/{len(test_loader)}",
                'Samples': f"{(batch_idx + 1) * CONFIG['batch_size']}/{len(test_loader.dataset)}"
            })
    
    return results

"""#### Results Analysis and Saving"""

def analyze_and_save_results(results):
    """Calculate statistics and save results"""
    
    log_and_print("\n" + "="*80)
    log_and_print("SEGMENTATION INFERENCE RESULTS")
    log_and_print("="*80)
    
    # Calculate statistics
    statistics = {}
    for method, dice_scores in results.items():
        dice_scores = np.array(dice_scores)
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        
        statistics[method] = {
            'mean': float(mean_dice),
            'std': float(std_dice),
            'count': len(dice_scores)
        }
        
        # Display results
        method_name = method.replace('_', ' ').title()
        log_and_print(f"{method_name}:")
        log_and_print(f"  Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
        log_and_print(f"  Sample Count: {len(dice_scores)}")
        log_and_print("")
    
    # Save detailed results to JSON
    results_summary = {
        'statistics': statistics,
        'raw_results': {method: [float(x) for x in scores] for method, scores in results.items()},
        'config': config_to_save,
        'timestamp': timestamp
    }
    
    results_path = os.path.join(run_dir, 'segmentation_inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    log_and_print(f"Detailed results saved to: {results_path}")
    
    # Save summary results to text file
    summary_path = os.path.join(run_dir, 'segmentation_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Segmentation Inference Results Summary\n")
        f.write("="*50 + "\n\n")
        for method, stats in statistics.items():
            method_name = method.replace('_', ' ').title()
            f.write(f"{method_name}:\n")
            f.write(f"  Dice Score: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Sample Count: {stats['count']}\n\n")
    
    log_and_print(f"Summary results saved to: {summary_path}")
    
    return statistics

def create_results_visualization(statistics):
    """Create a bar plot visualization of the results"""
    
    methods = list(statistics.keys())
    method_names = [method.replace('_', ' ').title() for method in methods]
    means = [statistics[method]['mean'] for method in methods]
    stds = [statistics[method]['std'] for method in methods]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(method_names, means, yerr=stds, capsize=5, alpha=0.8, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
    
    plt.title('Segmentation Performance (Dice Score) Across Reconstruction Methods', fontsize=16, fontweight='bold')
    plt.ylabel('Dice Score', fontsize=14)
    plt.xlabel('Reconstruction Method', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, max(means) + max(stds) + 0.1)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(run_dir, "segmentation_results_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_and_print(f"Results visualization saved to: {plot_path}")

"""#### Main Execution"""

if __name__ == "__main__":
    try:
        # Run segmentation inference
        results = evaluate_segmentation_on_reconstructed_images()
        
        # Analyze and save results
        statistics = analyze_and_save_results(results)
        
        # Create visualization
        create_results_visualization(statistics)
        
        # Log completion
        log_and_print(f"\nSegmentation inference completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_and_print(f"All outputs saved to: {run_dir}")
        
    except Exception as e:
        log_and_print(f"Error during segmentation inference: {str(e)}")
        import traceback
        log_and_print(f"Traceback: {traceback.format_exc()}")
        
    finally:
        # Close log file
        log_file.close()
