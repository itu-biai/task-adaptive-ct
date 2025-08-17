"""#### Install Libraries"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import radon, iradon
import bm3d
from scipy.ndimage import zoom
from matplotlib.widgets import RectangleSelector
from datetime import datetime
import json

"""#### CONFIG"""

CONFIG = {
    # Inference parameters
    'batch_size': 1,
    'num_workers': 64,
    'train_test_split': 0.8,  # 80% training, 20% testing
    'default_index': 5,  # Default sample index for visualization
    
    # BM3D parameters
    'sigma_psd': 0.075,
    
    # FBP parameters
    'theta': None,  # None for auto-calculation
    
    # Dataset directories
    'low_dose_dir': '/YEREL/dataset/low_dose',
    'full_dose_dir': '/YEREL/dataset/full_dose',
    
    # Model weights paths
    'task_adaptive_model_path': '.../model-weights/task-adaptive/...',
    'base_unet_model_path': '.../model-weights/base-unet/...',
    
    # Output directory for plots and logs
    'output_dir': '.../inference-results/bounding-box'
}

"""#### Output Directory and Logging Setup"""

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Create unique run directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(CONFIG['output_dir'], f"inference_run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

print(f"Inference outputs will be saved to: {run_dir}")

# Save configuration to run directory
config_path = os.path.join(run_dir, "config.json")
config_to_save = CONFIG.copy()
with open(config_path, 'w') as f:
    json.dump(config_to_save, f, indent=2)

# Initialize log file
log_file_path = os.path.join(run_dir, "inference_log.txt")
log_file = open(log_file_path, 'w')
log_file.write(f"Inference Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

# Dataset for Inference
class TaskAdaptiveDataset(Dataset):
    def __init__(self, low_dose_dir, full_dose_dir, transform_recon=None):
        self.transform_recon = transform_recon

        # Preload all data (already normalized to [0, 1] during dataset creation)
        self.low_dose = load_npy_from_folder(low_dose_dir)    # Shape: (N, H, W)
        self.full_dose = load_npy_from_folder(full_dose_dir)  # Shape: (N, H, W)

        # Verify that datasets have the same number of samples
        assert len(self.low_dose) == len(self.full_dose), \
            "Mismatch in number of files across directories"

    def __len__(self):
        return len(self.low_dose)

    def __getitem__(self, idx):
        low_dose_img = self.low_dose[idx]   # Shape: (H, W)
        full_dose_img = self.full_dose[idx] # Shape: (H, W)

        # Apply transformations to images (e.g., ToTensor)
        if self.transform_recon:
            low_dose_img = self.transform_recon(low_dose_img)   # Shape: (1, H, W)
            full_dose_img = self.transform_recon(full_dose_img) # Shape: (1, H, W)

        return low_dose_img, full_dose_img

# Define transform for images
transform_recon = transforms.ToTensor()  # Images are already normalized to [0, 1] during dataset creation

# Instantiate the dataset
dataset = TaskAdaptiveDataset(
    CONFIG['low_dose_dir'],
    CONFIG['full_dose_dir'],
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

"""#### Circular Mask"""

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(w, h) / 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.float32)

"""#### PSNR & SSIM Metrics"""

# PSNR and SSIM metrics
def psnr_metric(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    psnr_values = []
    for i in range(pred_np.shape[0]):
        p = peak_signal_noise_ratio(target_np[i, 0], pred_np[i, 0], data_range=target_np[i, 0].max() - target_np[i, 0].min())
        psnr_values.append(p)
    return psnr_values

def ssim_metric(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_values = []
    for i in range(pred_np.shape[0]):
        s = structural_similarity(target_np[i, 0], pred_np[i, 0], data_range=target_np[i, 0].max() - target_np[i, 0].min())
        ssim_values.append(s)
    return ssim_values

def masked_psnr_metric(pred, target, mask):
    mask_tensor = torch.from_numpy(mask).to(pred.device).view(1, 1, 512, 512)
    masked_pred = pred * mask_tensor
    masked_target = target * mask_tensor
    return psnr_metric(masked_pred, masked_target)

def masked_ssim_metric(pred, target, mask):
    mask_tensor = torch.from_numpy(mask).to(pred.device).view(1, 1, 512, 512)
    masked_pred = pred * mask_tensor
    masked_target = target * mask_tensor
    return ssim_metric(masked_pred, masked_target)

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

        return torch.sigmoid(out)  # Output normalized to [0, 1] for reconstruction

"""#### Visualization Function"""

def visualize_reconstruction_results(task_adaptive_model, ld_model, test_dataset, mask, index=CONFIG['default_index'], device='cuda', sigma_psd=CONFIG['sigma_psd']):
    task_adaptive_model.eval()
    ld_model.eval()

    # Get sample from test dataset
    low_dose, full_dose = test_dataset[index]
    low_dose = low_dose.unsqueeze(0).to(device)  # Add batch dimension
    full_dose = full_dose.unsqueeze(0).to(device)

    # Perform inferences
    with torch.no_grad():
        task_adaptive_output = task_adaptive_model(low_dose)
        ld_output = ld_model(low_dose)

    # FBP Reconstruction
    theta = CONFIG['theta']
    if theta is None:
        theta = np.linspace(0., 180., max(512, 512), endpoint=False)
    noisy_np = low_dose.squeeze(0).cpu().numpy()  # Shape: (1, H, W)
    fbp_reconstructions = []
    for j in range(noisy_np.shape[0]):
        sinogram = radon(noisy_np[j], theta=theta, circle=False)
        reconstruction = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
        fbp_reconstructions.append(reconstruction)
    fbp_reconstructions = np.stack(fbp_reconstructions, axis=0)  # Shape: (1, H, W)

    # Resize FBP if necessary
    if fbp_reconstructions.shape[-2:] != full_dose.shape[-2:]:
        scale_h = full_dose.shape[-2] / fbp_reconstructions.shape[-2]
        scale_w = full_dose.shape[-1] / fbp_reconstructions.shape[-1]
        fbp_reconstructions = zoom(fbp_reconstructions, (1, scale_h, scale_w), order=1)

    # BM3D Denoising
    bm3d_denoised = []
    for j in range(noisy_np.shape[0]):
        denoised = bm3d.bm3d(noisy_np[j], sigma_psd=sigma_psd)
        bm3d_denoised.append(denoised)
    bm3d_denoised = np.stack(bm3d_denoised, axis=0)  # Shape: (1, H, W)

    # Convert NumPy arrays to tensors for consistent metric computation
    fbp_reconstructions_tensor = torch.from_numpy(fbp_reconstructions).unsqueeze(0).to(device).float()
    bm3d_denoised_tensor = torch.from_numpy(bm3d_denoised.astype(np.float32)).unsqueeze(0).to(device)    
    low_dose_tensor = low_dose
    full_dose_tensor = full_dose
    ld_output_tensor = ld_output
    task_adaptive_output_tensor = task_adaptive_output

    # Compute metrics
    low_dose_psnr = masked_psnr_metric(low_dose_tensor, full_dose_tensor, mask)[0]
    low_dose_ssim = masked_ssim_metric(low_dose_tensor, full_dose_tensor, mask)[0]
    fbp_psnr = masked_psnr_metric(fbp_reconstructions_tensor, full_dose_tensor, mask)[0]
    fbp_ssim = masked_ssim_metric(fbp_reconstructions_tensor, full_dose_tensor, mask)[0]
    bm3d_psnr = masked_psnr_metric(bm3d_denoised_tensor, full_dose_tensor, mask)[0]
    bm3d_ssim = masked_ssim_metric(bm3d_denoised_tensor, full_dose_tensor, mask)[0]
    ld_psnr = masked_psnr_metric(ld_output_tensor, full_dose_tensor, mask)[0]
    ld_ssim = masked_ssim_metric(ld_output_tensor, full_dose_tensor, mask)[0]
    task_adaptive_psnr = masked_psnr_metric(task_adaptive_output_tensor, full_dose_tensor, mask)[0]
    task_adaptive_ssim = masked_ssim_metric(task_adaptive_output_tensor, full_dose_tensor, mask)[0]

    # Convert to numpy and apply circular mask for plotting
    mask_np = mask  # Shape: (512, 512)
    low_dose_np = low_dose_tensor.squeeze(0).cpu().numpy().squeeze(0) * mask_np  # [512, 512], masked
    fbp_np = fbp_reconstructions[0] * mask_np  # [512, 512], masked
    bm3d_np = bm3d_denoised[0] * mask_np  # [512, 512], masked
    ld_np = ld_output_tensor.squeeze(0).cpu().numpy().squeeze(0) * mask_np  # [512, 512], masked
    task_adaptive_np = task_adaptive_output_tensor.squeeze(0).cpu().numpy().squeeze(0) * mask_np  # [512, 512], masked
    full_dose_np = full_dose_tensor.squeeze(0).cpu().numpy().squeeze(0) * mask_np  # [512, 512], masked

    # Interactive ROI selection with RectangleSelector
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Click and drag to draw ROI on Low-Dose image')
    ax.imshow(low_dose_np, cmap='gray')
    ax.axis('on')  # Show axes for reference

    # Store ROI coordinates
    roi_coords = []

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        roi_coords.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y2, y2)))
        plt.close(fig)  # Close the figure after selection

    selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='data', interactive=True)
    plt.show()

    # Wait until ROI is selected
    while not roi_coords:
        plt.pause(0.1)

    # Extract ROI coordinates
    x1, y1, x2, y2 = roi_coords[0]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(512, x2)), int(min(512, y2))
    if x2 <= x1 or y2 <= y1:  # Ensure valid rectangle
        raise ValueError("Invalid ROI selected. Please select a valid rectangle.")

    # Define ROI slices
    x_slice = slice(x1, x2)
    y_slice = slice(y1, y2)

    # Extract ROI patches
    low_dose_roi = low_dose_np[y_slice, x_slice]
    fbp_roi = fbp_np[y_slice, x_slice]
    bm3d_roi = bm3d_np[y_slice, x_slice]
    ld_roi = ld_np[y_slice, x_slice]
    task_adaptive_roi = task_adaptive_np[y_slice, x_slice]
    full_dose_roi = full_dose_np[y_slice, x_slice]

    # Plot original results in a 3x2 grid
    plt.figure(figsize=(8, 12))

    plt.subplot(3, 2, 1)
    plt.title(f'Low-Dose\nPSNR: {low_dose_psnr:.2f}, SSIM: {low_dose_ssim:.3f}', fontsize=10)
    plt.imshow(low_dose_np, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.title(f'FBP\nPSNR: {fbp_psnr:.2f}, SSIM: {fbp_ssim:.3f}', fontsize=10)
    plt.imshow(fbp_np, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.title(f'BM3D\nPSNR: {bm3d_psnr:.2f}, SSIM: {bm3d_ssim:.3f}', fontsize=10)
    plt.imshow(bm3d_np, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title(f'U-Net\nPSNR: {ld_psnr:.2f}, SSIM: {ld_ssim:.3f}', fontsize=10)
    plt.imshow(ld_np, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.title(f'Task-Adaptive\nPSNR: {task_adaptive_psnr:.2f}, SSIM: {task_adaptive_ssim:.3f}', fontsize=10)
    plt.imshow(task_adaptive_np, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title(f'Full-Dose', fontsize=10)
    plt.imshow(full_dose_np, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot ROI boundary on full-sized low-dose scan
    plt.figure(figsize=(6, 6))
    plt.title('Low-Dose with ROI Boundary')
    plt.imshow(low_dose_np, cmap='gray')
    # Draw rectangle for ROI
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Plot ROI results in a 3x2 grid
    plt.figure(figsize=(8, 12))

    plt.subplot(3, 2, 1)
    plt.title(f'Low-Dose ROI\nPSNR: {low_dose_psnr:.2f}, SSIM: {low_dose_ssim:.3f}', fontsize=10)
    plt.imshow(low_dose_roi, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.title(f'FBP ROI\nPSNR: {fbp_psnr:.2f}, SSIM: {fbp_ssim:.3f}', fontsize=10)
    plt.imshow(fbp_roi, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.title(f'BM3D ROI\nPSNR: {bm3d_psnr:.2f}, SSIM: {bm3d_ssim:.3f}', fontsize=10)
    plt.imshow(bm3d_roi, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title(f'U-Net ROI\nPSNR: {ld_psnr:.2f}, SSIM: {ld_ssim:.3f}', fontsize=10)
    plt.imshow(ld_roi, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.title(f'Task-Adaptive ROI\nPSNR: {task_adaptive_psnr:.2f}, SSIM: {task_adaptive_ssim:.3f}', fontsize=10)
    plt.imshow(task_adaptive_roi, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title(f'Full-Dose ROI', fontsize=10)
    plt.imshow(full_dose_roi, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

"""#### Setup"""

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_and_print(f"Using device: {device}")

# Create circular mask
mask = create_circular_mask(512, 512)

# Load pre-trained models
log_and_print("Loading pre-trained models...")

# Load Task-Adaptive model
task_adaptive_model = U_Net(in_channels=1, out_channels=1).to(device)
task_adaptive_model.load_state_dict(torch.load(CONFIG['task_adaptive_model_path'], map_location=device))
task_adaptive_model.eval()
log_and_print(f"Task-Adaptive model loaded from: {CONFIG['task_adaptive_model_path']}")

# Load Base U-Net model
ld_model = U_Net(in_channels=1, out_channels=1).to(device)
ld_model.load_state_dict(torch.load(CONFIG['base_unet_model_path'], map_location=device))
ld_model.eval()
log_and_print(f"Base U-Net model loaded from: {CONFIG['base_unet_model_path']}")

# Set BM3D sigma_psd
sigma_psd = CONFIG['sigma_psd']
log_and_print(f"BM3D sigma_psd set to: {sigma_psd}")

"""#### Results Visualization and Analysis"""

# Visualize results for a sample
sample_index = CONFIG['default_index']
log_and_print(f"Starting visualization for sample index: {sample_index}")

# Create save path for the visualization
visualization_save_path = os.path.join(run_dir, f"bounding_box_analysis_sample_{sample_index}.png")

# Run visualization with results saving
try:
    visualize_reconstruction_results(
        task_adaptive_model, 
        ld_model, 
        test_dataset, 
        mask, 
        index=sample_index, 
        device=device, 
        sigma_psd=sigma_psd
    )
    log_and_print("Visualization completed successfully")
    
    # Save completion message
    log_and_print(f"\nInference completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"All outputs saved to: {run_dir}")
    
except Exception as e:
    log_and_print(f"Error during visualization: {str(e)}")
    
finally:
    # Close log file
    log_file.close()
