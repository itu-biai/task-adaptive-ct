"""#### Install Libraries"""

import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import json
import copy

"""#### CONFIG"""

CONFIG = {
    # Training parameters
    'num_epochs': 200,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_workers': 64,
    'train_test_split': 0.8,  # 80% training, 20% testing
    
    # Optimizer parameters
    'optimizer': 'Adam',
    'weight_decay': 1e-4,  # L2 regularization to prevent overfitting
    'betas': (0.9, 0.999),  # Adam momentum parameters
    'eps': 1e-8,  # Adam epsilon for numerical stability
    
    # Learning rate scheduling
    'use_lr_scheduler': True,
    'scheduler_type': 'StepLR',  # Options: 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'
    'step_size': 15,  # For StepLR: reduce LR every 15 epochs
    'gamma': 0.5,  # For StepLR: multiply LR by 0.5
    'patience_lr': 8,  # For ReduceLROnPlateau: wait 7 epochs before reducing LR
    'factor': 0.5,  # For ReduceLROnPlateau: multiply LR by 0.5
    'min_lr': 1e-6,  # Minimum learning rate
    
    # Early stopping parameters
    'patience': 10,  # Increased for longer training - wait 10 epochs without improvement
    'min_epochs': 20,  # Enforce at least 20 epochs before early stopping
    
    # Dataset directories
    'noisy_dir': '.../dataset/low_dose',
    'clean_dir': '.../dataset/full_dose',
    
    # Output directory for plots and logs
    'output_dir': '.../logs/base-unet', 
    
    # Loss parameters
    'criterion': nn.MSELoss()
}

"""#### Output Directory and Logging Setup"""

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Use output directory directly (overwrites previous runs)
run_dir = CONFIG['output_dir']
os.makedirs(run_dir, exist_ok=True)

print(f"Training outputs will be saved to: {run_dir}")

# Save configuration to run directory
config_path = os.path.join(run_dir, "config.json")
config_to_save = CONFIG.copy()
# Convert non-serializable objects to strings
config_to_save['criterion'] = str(CONFIG['criterion'])
with open(config_path, 'w') as f:
    json.dump(config_to_save, f, indent=2)

# Initialize training tracking
training_history = {
    'epoch': [],
    'train_loss': [],
    'val_psnr': [],
    'val_ssim': [],
    'learning_rate': [],
    'best_epoch': 0,
    'best_ssim': -1.0,
    'best_psnr': -1.0,
    'best_ssim_std': 0.0,
    'best_psnr_std': 0.0
}

# Initialize log file
log_file_path = os.path.join(run_dir, "training_log.txt")
log_file = open(log_file_path, 'w')
log_file.write(f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

def save_training_plots():
    """Save training curves to plots"""
    if len(training_history['epoch']) < 2:
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = training_history['epoch']
    
    # Plot 1: Training Loss
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Validation PSNR
    ax2.plot(epochs, training_history['val_psnr'], 'g-', label='Validation PSNR', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Validation PSNR Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Validation SSIM
    ax3.plot(epochs, training_history['val_ssim'], 'r-', label='Validation SSIM', linewidth=2)
    # Mark best epoch
    if training_history['best_epoch'] > 0:
        best_idx = training_history['best_epoch'] - 1
        ax3.scatter(epochs[best_idx], training_history['val_ssim'][best_idx], 
                   color='red', s=100, marker='*', label=f'Best (Epoch {training_history["best_epoch"]})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.set_title('Validation SSIM Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Learning Rate
    ax4.plot(epochs, training_history['learning_rate'], 'm-', label='Learning Rate', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_and_print(f"Training curves saved to: {plot_path}")

"""#### Dataset"""

# Function to preload .npy files from a folder
def load_npy_from_folder(folder_path):
    images = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        img = np.load(file_path)  # Shape: (H, W), already normalized to [0, 1]
        images.append(img)
    return np.stack(images, axis=0)  # Shape: (N, H, W)

# Dataset for paired noisy-clean images
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.transform = transform
        self.noisy = load_npy_from_folder(noisy_dir)  # Shape: (N, H, W)
        self.clean = load_npy_from_folder(clean_dir)  # Shape: (N, H, W)
        assert len(self.noisy) == len(self.clean), "Mismatch between noisy and clean images"

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy_img = self.noisy[idx]  # Shape: (H, W)
        clean_img = self.clean[idx]  # Shape: (H, W)

        # Apply transforms if provided
        if self.transform:
            noisy_img = self.transform(noisy_img)  # Shape: (1, H, W)
            clean_img = self.transform(clean_img)  # Shape: (1, H, W)

        return noisy_img, clean_img

# Define transform (for both noisy and clean images)
transform = transforms.ToTensor()  # Images are already normalized to [0, 1] during dataset creation

# Create dataset with preloaded data
dataset = DenoisingDataset(CONFIG['noisy_dir'], CONFIG['clean_dir'], transform=transform)

# Set random seed for reproducibility
torch.manual_seed(1773)
generator = torch.Generator().manual_seed(1773)

# Split dataset
train_size = int(CONFIG['train_test_split'] * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True, 
    num_workers=CONFIG['num_workers']
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=CONFIG['num_workers']
)

# Extract noisy and clean images from the processed test_dataset
noisy_processed = []
clean_processed = []

# Iterate through the test_dataset to get processed data
for noisy_img, clean_img in dataset:
    # noisy_img and clean_img are tensors of shape (1, H, W) in range [0, 1]
    noisy_processed.append(noisy_img.numpy().squeeze())  # Shape: (H, W)
    clean_processed.append(clean_img.numpy().squeeze())  # Shape: (H, W)

# Convert lists to numpy arrays and flatten for histogram
noisy_processed = np.stack(noisy_processed, axis=0).flatten()  # Shape: (N*H*W,)
clean_processed = np.stack(clean_processed, axis=0).flatten()  # Shape: (N*H*W,)

# Check the range of values
print(f"Noisy images - Min: {noisy_processed.min():.4f}, Max: {noisy_processed.max():.4f}")
print(f"Clean images - Min: {clean_processed.min():.4f}, Max: {clean_processed.max():.4f}")

# Plot histograms of the processed data in [0, 1] range
plt.figure(figsize=(10, 6))
plt.hist(noisy_processed, bins=100, density=True, alpha=0.5, color='blue', label='Processed Noisy Images')
plt.hist(clean_processed, bins=100, density=True, alpha=0.5, color='green', label='Processed Clean Images')
plt.title('Histogram of Processed Noisy and Clean Image Pixel Intensities ')
plt.xlabel('Pixel Intensity (in [0, 1])')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Function to visualize noisy and clean image pairs
def visualize_test_samples(test_loader, num_samples=4):
    # Get one batch from the test loader
    noisy_images, clean_images = next(iter(test_loader))

    # Limit to num_samples
    num_samples = min(num_samples, noisy_images.size(0))

    # Create a figure with 2 rows (noisy, clean) and num_samples columns
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        # Get noisy and clean images (shape: [1, H, W])
        noisy_img = noisy_images[i].squeeze(0)  # Shape: [H, W]
        clean_img = clean_images[i].squeeze(0)  # Shape: [H, W]

        # Convert to NumPy for plotting
        noisy_img = noisy_img.cpu().numpy()
        clean_img = clean_img.cpu().numpy()

        # Plot noisy image
        if num_samples == 1:
            axes[0].imshow(noisy_img, cmap='gray')
            axes[0].set_title('Noisy Image')
            axes[0].axis('off')

            axes[1].imshow(clean_img, cmap='gray')
            axes[1].set_title('Clean Image')
            axes[1].axis('off')
        else:
            axes[0, i].imshow(noisy_img, cmap='gray')
            axes[0, i].set_title('Noisy Image')
            axes[0, i].axis('off')

            axes[1, i].imshow(clean_img, cmap='gray')
            axes[1, i].set_title('Clean Image')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize samples from test dataset
visualize_test_samples(test_loader, num_samples=5)

"""#### Circular Mask"""

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # Use the center of the image
        center = (w // 2, h // 2)
    if radius is None:  # Use half the image width
        radius = min(w, h) / 2

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.float32)  # Shape: (H, W)

def visualize_with_mask(test_loader, mask, num_samples=2):
    noisy_images, clean_images = next(iter(test_loader))
    num_samples = min(num_samples, noisy_images.size(0))

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        noisy_img = noisy_images[i].squeeze(0).cpu().numpy()  # Shape: (512, 512)
        clean_img = clean_images[i].squeeze(0).cpu().numpy()  # Shape: (512, 512)

        # Create a green mask overlay (RGBA: green where mask=1, transparent where mask=0)
        mask_rgba = np.zeros((512, 512, 4))
        mask_rgba[..., 0] = mask  # Green channel
        mask_rgba[..., 3] = mask * 0.3  # Semi-transparent where mask=1

        # Plot noisy image with mask
        if num_samples == 1:
            axes[0].imshow(noisy_img, cmap='gray')
            axes[0].imshow(mask_rgba)
            axes[0].set_title('Noisy Image with Mask')
            axes[0].axis('off')

            axes[1].imshow(clean_img, cmap='gray')
            axes[1].imshow(mask_rgba)
            axes[1].set_title('Clean Image with Mask')
            axes[1].axis('off')
        else:
            axes[0, i].imshow(noisy_img, cmap='gray')
            axes[0, i].imshow(mask_rgba)
            axes[0, i].set_title('Noisy Image with Mask')
            axes[0, i].axis('off')

            axes[1, i].imshow(clean_img, cmap='gray')
            axes[1, i].imshow(mask_rgba)
            axes[1, i].set_title('Clean Image with Mask')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)  # Shape: (512, 512)
visualize_with_mask(test_loader, mask, num_samples=5)

"""#### PSNR & SSIM Metrics"""

# PSNR metric calculation.
def psnr_metric(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    psnr_values = []
    for i in range(pred_np.shape[0]):
        # Single-channel images, index with [0] to get [H, W]
        p = peak_signal_noise_ratio(target_np[i, 0], pred_np[i, 0], data_range=target_np[i, 0].max() - target_np[i, 0].min())
        psnr_values.append(p)
    return psnr_values

# SSIM metric using skimage (compute per image in the batch).
def ssim_metric(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_values = []
    for i in range(pred_np.shape[0]):
        # Since images are single-channel, index with [0]
        s = structural_similarity(target_np[i, 0], pred_np[i, 0], data_range=target_np[i, 0].max() - target_np[i, 0].min())
        ssim_values.append(s)
    return ssim_values

"""#### Low-Dose Images"""

# Function to compute masked PSNR for a batch
def masked_psnr_metric(noisy, clean, mask):
    """
    Compute PSNR for each image in a batch, only within the masked region.
    noisy, clean: Torch tensors of shape (batch_size, 1, 512, 512)
    mask: Numpy array of shape (512, 512)
    Returns: List of PSNR scores
    """
    mask_tensor = torch.from_numpy(mask).to(noisy.device).view(1, 1, 512, 512)
    masked_noisy = noisy * mask_tensor
    masked_clean = clean * mask_tensor
    return psnr_metric(masked_noisy, masked_clean)

# Function to compute masked SSIM for a batch
def masked_ssim_metric(noisy, clean, mask):
    """
    Compute SSIM for each image in a batch, only within the masked region.
    noisy, clean: Torch tensors of shape (batch_size, 1, 512, 512)
    mask: Numpy array of shape (512, 512)
    Returns: List of SSIM scores
    """
    mask_tensor = torch.from_numpy(mask).to(noisy.device).view(1, 1, 512, 512)
    masked_noisy = noisy * mask_tensor
    masked_clean = clean * mask_tensor
    return ssim_metric(masked_noisy, masked_clean)

# Calculate metrics for low-dose images
lowdose_psnr_scores = []
lowdose_ssim_scores = []

# No need for model, evaluate noisy vs clean directly
with torch.no_grad():
    for noisy, clean in test_loader:
        lowdose_psnr_scores.extend(masked_psnr_metric(noisy, clean, mask))
        lowdose_ssim_scores.extend(masked_ssim_metric(noisy, clean, mask))

# Compute averages and standard deviations
lowdose_avg_psnr = np.mean(lowdose_psnr_scores)
lowdose_std_psnr = np.std(lowdose_psnr_scores)
lowdose_avg_ssim = np.mean(lowdose_ssim_scores)
lowdose_std_ssim = np.std(lowdose_ssim_scores)

# Print results
print(f"Low-Dose Images -> PSNR: {lowdose_avg_psnr:.4f} ± {lowdose_std_psnr:.4f}, SSIM: {lowdose_avg_ssim:.4f} ± {lowdose_std_ssim:.4f}")

"""#### U-Net Model"""

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
        out =  self.final_conv(c1)

        # Use sigmoid to ensure output in [0,1] (assuming normalized images)
        return torch.sigmoid(out)

# ----- Model, optimizer, loss -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U_Net().to(device)

# Setup optimizer based on CONFIG
if CONFIG['optimizer'] == 'Adam':
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=CONFIG['betas'],
        eps=CONFIG['eps']
    )
elif CONFIG['optimizer'] == 'SGD':
    optimizer = optim.SGD(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        momentum=0.9
    )
else:
    raise ValueError(f"Unsupported optimizer: {CONFIG['optimizer']}")

criterion = CONFIG['criterion']

# Setup learning rate scheduler
scheduler = None
if CONFIG['use_lr_scheduler']:
    if CONFIG['scheduler_type'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=CONFIG['step_size'], 
            gamma=CONFIG['gamma']
        )
    elif CONFIG['scheduler_type'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # We want to maximize SSIM
            factor=CONFIG['factor'],
            patience=CONFIG['patience_lr'],
            min_lr=CONFIG['min_lr'],
            verbose=True
        )
    elif CONFIG['scheduler_type'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=CONFIG['num_epochs'],
            eta_min=CONFIG['min_lr']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {CONFIG['scheduler_type']}")
    
    log_and_print(f"Using {CONFIG['scheduler_type']} learning rate scheduler")

# Convert mask to tensor for training
mask_tensor = torch.from_numpy(mask).to(device).view(1, 1, 512, 512)

# ----- Training loop with Early Stopping -----
# Early stopping variables
best_ssim = -1.0
epochs_without_improvement = 0
best_model_state = None

log_and_print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
log_and_print(f"Device: {device}")
log_and_print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(CONFIG['num_epochs']):
    model.train()
    train_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Evaluation on test set using masked metrics
    model.eval()
    psnr_scores = []
    ssim_scores = []
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            psnr_scores.extend(masked_psnr_metric(output, clean, mask))
            ssim_scores.extend(masked_ssim_metric(output, clean, mask))
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)

    # Get current learning rate for display
    current_lr = optimizer.param_groups[0]['lr']
    
    # Track metrics in history
    training_history['epoch'].append(epoch + 1)
    training_history['train_loss'].append(train_loss)
    training_history['val_psnr'].append(avg_psnr)
    training_history['val_ssim'].append(avg_ssim)
    training_history['learning_rate'].append(current_lr)
    
    # Early stopping logic
    if avg_ssim > best_ssim:
        best_ssim = avg_ssim
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        training_history['best_epoch'] = epoch + 1
        training_history['best_ssim'] = best_ssim
        training_history['best_psnr'] = avg_psnr
        training_history['best_ssim_std'] = std_ssim
        training_history['best_psnr_std'] = std_psnr
        log_and_print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {train_loss:.4f}, LR: {current_lr:.2e}, "
              f"U-Net Reconstructed -> Masked PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}, "
              f"Masked SSIM: {avg_ssim:.4f} ± {std_ssim:.4f} (NEW BEST!)")
    else:
        epochs_without_improvement += 1
        log_and_print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {train_loss:.4f}, LR: {current_lr:.2e}, "
              f"U-Net Reconstructed -> Masked PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}, "
              f"Masked SSIM: {avg_ssim:.4f} ± {std_ssim:.4f} (No improvement: {epochs_without_improvement}/{CONFIG['patience']})")
    
    # Update learning rate scheduler
    if scheduler is not None:
        if CONFIG['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler.step(avg_ssim)  # Step with SSIM for plateau detection
        else:
            scheduler.step()  # Step with epoch for other schedulers
    
    # Training curves will be saved only at the end
    
    # Check early stopping condition
    if epoch + 1 >= CONFIG['min_epochs'] and epochs_without_improvement >= CONFIG['patience']:
        log_and_print(f"\nEarly stopping triggered! No improvement in SSIM for {CONFIG['patience']} epochs.")
        log_and_print(f"Loading best model with SSIM: {best_ssim:.4f}")
        model.load_state_dict(best_model_state)
        break

# If training completed without early stopping, use the best model found
if best_model_state is not None and epochs_without_improvement < CONFIG['patience']:
    log_and_print(f"\nTraining completed. Loading best model with SSIM: {best_ssim:.4f}")
    model.load_state_dict(best_model_state)

# Save final plots
save_training_plots()

unet = model

# Save model weights with constant name
model_name = 'base_unet_reconstruction.pth'
original_model_path = f'.../model-weights/{model_name}'

torch.save(unet.state_dict(), original_model_path)

log_and_print(f"Model saved to: {original_model_path}")

# Calculate final metrics with the best model
log_and_print("\nCalculating final metrics with best model...")
model.eval()
final_psnr_scores = []
final_ssim_scores = []
with torch.no_grad():
    for noisy, clean in test_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        final_psnr_scores.extend(masked_psnr_metric(output, clean, mask))
        final_ssim_scores.extend(masked_ssim_metric(output, clean, mask))
final_avg_psnr = np.mean(final_psnr_scores)
final_avg_ssim = np.mean(final_ssim_scores)
final_std_psnr = np.std(final_psnr_scores)
final_std_ssim = np.std(final_ssim_scores)

# ----------------------------------------------------------------------------------
# Quick verification: reload the just-saved weights and confirm metrics are identical
# ----------------------------------------------------------------------------------
log_and_print("\nVerifying saved model by reloading from disk ...")
reloaded_model = U_Net().to(device)
reloaded_model.load_state_dict(torch.load(original_model_path, map_location=device))
reloaded_model.eval()
verify_psnr, verify_ssim = [], []
with torch.no_grad():
    for noisy, clean in test_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = reloaded_model(noisy)
        verify_psnr.extend(masked_psnr_metric(output, clean, mask))
        verify_ssim.extend(masked_ssim_metric(output, clean, mask))
verify_avg_psnr = np.mean(verify_psnr)
verify_avg_ssim = np.mean(verify_ssim)

log_and_print(f"Reloaded model -> Masked PSNR: {verify_avg_psnr:.4f}, SSIM: {verify_avg_ssim:.4f}")
if np.isclose(verify_avg_psnr, final_avg_psnr) and np.isclose(verify_avg_ssim, final_avg_ssim):
    log_and_print("Verification successful: saved model matches in-memory best model.")
else:
    log_and_print("WARNING: Reloaded model metrics differ! Check saving logic.")


# Save training history and final results
results_summary = {
    'training_history': training_history,
    'final_metrics': {
        'psnr_mean': float(final_avg_psnr),
        'psnr_std': float(final_std_psnr),
        'ssim_mean': float(final_avg_ssim),
        'ssim_std': float(final_std_ssim)
    },
    'lowdose_baseline': {
        'psnr_mean': float(lowdose_avg_psnr),
        'psnr_std': float(lowdose_std_psnr),
        'ssim_mean': float(lowdose_avg_ssim),
        'ssim_std': float(lowdose_std_ssim)
    },
    'best_epoch': training_history['best_epoch'],
    'total_epochs_trained': len(training_history['epoch']),
    'early_stopped': len(training_history['epoch']) < CONFIG['num_epochs']
}

# Save results to JSON
results_path = os.path.join(run_dir, 'training_results.json')
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

log_and_print(f"Training results saved to: {results_path}")

# Log training summary
log_and_print("\n" + "="*80)
log_and_print("TRAINING SUMMARY")
log_and_print("="*80)
log_and_print(f"Total epochs trained: {len(training_history['epoch'])}/{CONFIG['num_epochs']}")
log_and_print(f"Best epoch: {training_history['best_epoch']}")
log_and_print(f"Best SSIM: {training_history['best_ssim']:.4f} ± {training_history['best_ssim_std']:.4f}")
log_and_print(f"Best PSNR: {training_history['best_psnr']:.4f} ± {training_history['best_psnr_std']:.4f}")
log_and_print(f"Early stopping: {'Yes' if len(training_history['epoch']) < CONFIG['num_epochs'] else 'No'}")
log_and_print(f"Final learning rate: {training_history['learning_rate'][-1]:.2e}")
log_and_print("="*80)

# Close log file
log_and_print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print(f"All outputs saved to: {run_dir}")
log_file.close()

def show_reconstruction_comparison(dataset, index, model_unet, device='cuda', save_path=None):
    """
    Plots noisy, U-Net reconstructed, and full-dose images for a given dataset index,
    with masked PSNR and SSIM metrics displayed in the titles. Images are masked to show only the circular region.

    Parameters:
    - dataset: DenoisingDataset instance
    - index: Index of the image to display
    - model_unet: Trained U-Net model
    - device: Device to run models on ('cuda' or 'cpu')
    - save_path: Optional path to save the image. If None, image is displayed.
    """
    # Create the circular mask
    mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)  # Shape: (512, 512)

    # Get the noisy and clean images from the dataset
    noisy, clean = dataset[index]
    noisy = noisy.unsqueeze(0)  # Add batch dimension: [1, 1, 512, 512]
    clean = clean.unsqueeze(0)  # Add batch dimension: [1, 1, 512, 512]

    # Move to device
    noisy = noisy.to(device)
    clean = clean.to(device)

    # U-Net Reconstruction
    model_unet.eval()
    with torch.no_grad():
        unet_output = model_unet(noisy)  # Shape: [1, 1, 512, 512]

    # Compute masked PSNR and SSIM
    psnr_noisy = masked_psnr_metric(noisy, clean, mask)[0]  # Single image score
    ssim_noisy = masked_ssim_metric(noisy, clean, mask)[0]  # Single image score
    psnr_unet = masked_psnr_metric(unet_output, clean, mask)[0]  # Single image score
    ssim_unet = masked_ssim_metric(unet_output, clean, mask)[0]  # Single image score

    # Convert to numpy and apply mask for plotting
    mask_np = mask  # Shape: [512, 512]
    noisy_np = noisy.squeeze(0).cpu().numpy().squeeze(0) * mask_np  # [512, 512], masked
    clean_np = clean.squeeze(0).cpu().numpy().squeeze(0) * mask_np  # [512, 512], masked
    unet_np = unet_output.squeeze(0).cpu().numpy().squeeze(0) * mask_np  # [512, 512], masked

    # Plotting with masked PSNR and SSIM in titles
    methods = [
        (f"Noisy Image\nMasked PSNR: {psnr_noisy:.2f}, Masked SSIM: {ssim_noisy:.3f}", noisy_np),
        (f"U-Net\nMasked PSNR: {psnr_unet:.2f}, Masked SSIM: {ssim_unet:.3f}", unet_np),
        (f"Full-Dose (Ground Truth)", clean_np)
    ]

    fig, axes = plt.subplots(1, len(methods), figsize=(3 * len(methods), 3))
    for ax, (title, img) in zip(axes, methods):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction comparison saved to: {save_path}")
        plt.close()
    else:
        plt.show()

# Assuming your models and dataset are already defined and trained
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create save path for the reconstruction comparison image
comparison_save_path = os.path.join(run_dir, "reconstruction_comparison.png")

show_reconstruction_comparison(
    dataset=test_dataset,
    index=5,  # Change this to show a different image
    model_unet=unet,  # Trained U-Net model
    device=device,
    save_path=comparison_save_path
)
