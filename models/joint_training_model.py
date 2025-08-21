"""#### Install Libraries"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datetime import datetime
import json

"""#### CONFIG"""

CONFIG = {
    # Training parameters
    'num_epochs': 200,
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_workers': 64,
    'train_test_split': 0.8,  # 80% training, 20% testing
    'num_classes': 3,  # Number of segmentation classes
    
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
    'patience_lr': 8,  # For ReduceLROnPlateau: wait 8 epochs before reducing LR
    'factor': 0.5,  # For ReduceLROnPlateau: multiply LR by 0.5
    'min_lr': 1e-6,  # Minimum learning rate
    
    # Early stopping parameters
    'patience': 10,  # Wait 10 epochs without improvement
    'min_epochs': 25,  # Minimum epochs before early stopping can trigger
    
    # Dataset directories
    'low_dose_dir': '.../low_dose',
    'full_dose_dir': '.../full_dose',
    'seg_dir': '.../segmentation_slices',
    
    # Output directory for plots and logs
    'output_dir': '.../training-logs/joint-training',
    
    # Loss parameters
    'criterion_mse': nn.MSELoss(),
    'alpha': 0.5  # Weight for Dice loss (1-alpha for MSE loss)
}

"""#### Dice Loss"""

class DiceLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, pred_logits, target):
        """
        pred_logits: Logits from the model, shape [B, C, H, W]
        target: Ground truth labels, shape [B, H, W]
        """
        pred_probs = torch.softmax(pred_logits, dim=1)
        
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred_probs * target_one_hot).sum(dim=[2, 3])
        pred_sum = pred_probs.sum(dim=[2, 3])
        target_sum = target_one_hot.sum(dim=[2, 3])

        dice_score = (2. * intersection + self.eps) / (pred_sum + target_sum + self.eps)
        
        return 1 - dice_score.mean()

# Hardcode DiceLoss instantiation
criterion_dice = DiceLoss(num_classes=CONFIG['num_classes'])

"""#### Output Directory and Logging Setup"""

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Create unique run directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(CONFIG['output_dir'], f"training_run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

print(f"Training outputs will be saved to: {run_dir}")

# Save configuration to run directory
config_path = os.path.join(run_dir, "config.json")
config_to_save = CONFIG.copy()
# Convert non-serializable objects to strings
config_to_save['criterion_mse'] = str(CONFIG['criterion_mse'])
config_to_save['criterion_dice'] = str(criterion_dice)
with open(config_path, 'w') as f:
    json.dump(config_to_save, f, indent=2)

# Initialize training tracking
training_history = {
    'epoch': [],
    'train_loss': [],
    'train_mse_loss': [],
    'train_dice_loss': [],
    'val_psnr': [],
    'val_ssim': [],
    'val_dice': [],
    'learning_rate': [],
    'best_epoch': 0,
    'best_ssim': -1.0,
    'best_psnr': -1.0,
    'best_dice': -1.0,
    'best_ssim_std': 0.0,
    'best_psnr_std': 0.0,
    'best_dice_std': 0.0
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
    
    # Plot 1: Training Losses
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Total Loss', linewidth=2)
    ax1.plot(epochs, training_history['train_mse_loss'], 'g-', label='MSE Loss', linewidth=2)
    ax1.plot(epochs, training_history['train_dice_loss'], 'r-', label='Dice Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Validation PSNR
    ax2.plot(epochs, training_history['val_psnr'], 'g-', label='Validation PSNR', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Validation PSNR Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Validation SSIM and Dice
    ax3_twin = ax3.twinx()  # Create twin y-axis for Dice
    line1 = ax3.plot(epochs, training_history['val_ssim'], 'r-', label='Validation SSIM', linewidth=2)
    line2 = ax3_twin.plot(epochs, training_history['val_dice'], 'b-', label='Validation Dice', linewidth=2)
    # Mark best epoch
    if training_history['best_epoch'] > 0:
        best_idx = training_history['best_epoch'] - 1
        ax3.scatter(epochs[best_idx], training_history['val_ssim'][best_idx], 
                   color='red', s=100, marker='*', label=f'Best (Epoch {training_history["best_epoch"]})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM', color='r')
    ax3_twin.set_ylabel('Dice Score', color='b')
    ax3.set_title('Validation SSIM and Dice Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='r')
    ax3_twin.tick_params(axis='y', labelcolor='b')
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center left')
    
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
        img = np.load(file_path)  # Shape: (H, W), already normalized to [0, 1] for images
        images.append(img)
    return np.stack(images, axis=0)  # Shape: (N, H, W)

# Unified Dataset for Joint Training
class JointTrainingUnifiedDataset(Dataset):
    def __init__(self, low_dose_dir, full_dose_dir, seg_dir, transform_recon=None, transform_seg=None):
        self.transform_recon = transform_recon
        self.transform_seg = transform_seg

        # Preload all data
        self.low_dose = load_npy_from_folder(low_dose_dir)    # Shape: (N, H, W), already normalized to [0, 1]
        self.full_dose = load_npy_from_folder(full_dose_dir)  # Shape: (N, H, W), already normalized to [0, 1]
        self.seg_maps = load_npy_from_folder(seg_dir)         # Shape: (N, H, W)

        # Verify that all datasets have the same number of samples
        assert len(self.low_dose) == len(self.full_dose) == len(self.seg_maps), \
            "Mismatch in number of files across directories"

    def __len__(self):
        return len(self.low_dose)

    def __getitem__(self, idx):
        low_dose_img = self.low_dose[idx]   # Shape: (H, W)
        full_dose_img = self.full_dose[idx] # Shape: (H, W)
        seg_map = self.seg_maps[idx]        # Shape: (H, W)

        # Apply transformations to images
        if self.transform_recon:
            low_dose_img = self.transform_recon(low_dose_img)   # Shape: (1, H, W)
            full_dose_img = self.transform_recon(full_dose_img) # Shape: (1, H, W)

        seg_map = seg_map.astype(np.int64)  # For CrossEntropyLoss

        if self.transform_seg:
            seg_map = self.transform_seg(seg_map)  # Shape: (1, H, W)

        return low_dose_img, full_dose_img, seg_map

"""#### Circular Mask"""

# Circular mask
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

"""#### Dice Metric"""

# Dice calculation
def calculate_dice(pred, target, num_classes=3, threshold=0.5, eps=1e-6):
    if num_classes > 1:
        pred = torch.softmax(pred, dim=1)
    else:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    if target.dim() == 3:
        target_one_hot = torch.zeros_like(pred)
        for c in range(num_classes):
            target_one_hot[:, c, :, :] = (target == c).float()
        target = target_one_hot

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
            return torch.sigmoid(out)
        else:
            return out

"""#### Setup"""

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transform for images
transform_recon = transforms.ToTensor()  # Images are already normalized to [0, 1] during dataset creation
transform_seg = transforms.ToTensor()    # For segmentation masks

# Instantiate the unified dataset
dataset = JointTrainingUnifiedDataset(
    CONFIG['low_dose_dir'], 
    CONFIG['full_dose_dir'], 
    CONFIG['seg_dir'], 
    transform_recon=transform_recon, 
    transform_seg=transform_seg
)

# Set random seed for reproducibility
torch.manual_seed(1773)
generator = torch.Generator().manual_seed(1773)

# Split dataset into training and testing sets
train_size = int(CONFIG['train_test_split'] * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# Create DataLoaders
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

# Define both models for joint training
recon_model = U_Net(in_channels=1, out_channels=1).to(device)
seg_model = U_Net(in_channels=1, out_channels=CONFIG['num_classes']).to(device)

# Setup combined parameters for optimizer
combined_parameters = list(recon_model.parameters()) + list(seg_model.parameters())

# Setup optimizer based on CONFIG
if CONFIG['optimizer'] == 'Adam':
    optimizer = optim.Adam(
        combined_parameters, 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=CONFIG['betas'],
        eps=CONFIG['eps']
    )
elif CONFIG['optimizer'] == 'SGD':
    optimizer = optim.SGD(
        combined_parameters, 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        momentum=0.9
    )
else:
    raise ValueError(f"Unsupported optimizer: {CONFIG['optimizer']}")

criterion_mse = CONFIG['criterion_mse']
# criterion_dice is already defined above

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

# Create circular mask
mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)

"""#### Joint Training"""

# ----- Training loop with Early Stopping -----
# Early stopping variables
best_ssim = -1.0
epochs_without_improvement = 0
best_recon_state = None
best_seg_state = None

log_and_print(f"\nStarting joint training for {CONFIG['num_epochs']} epochs...")
log_and_print(f"Device: {device}")
log_and_print(f"Reconstruction model parameters: {sum(p.numel() for p in recon_model.parameters()):,}")
log_and_print(f"Segmentation model parameters: {sum(p.numel() for p in seg_model.parameters()):,}")

for epoch in range(CONFIG['num_epochs']):
    recon_model.train()
    seg_model.train()
    train_loss = 0.0
    train_mse_loss = 0.0
    train_dice_loss = 0.0
    
    for low_dose, full_dose, seg_map in train_loader:
        low_dose = low_dose.to(device)
        full_dose = full_dose.to(device)
        seg_map = seg_map.to(device).squeeze(1)

        optimizer.zero_grad()

        # Reconstruction
        reconstructed = recon_model(low_dose)  # [B, 1, H, W], in [0, 1]
        mse_loss = criterion_mse(reconstructed, full_dose)

        # Segmentation on reconstructed image
        predicted_seg = seg_model(reconstructed)  # [B, 3, H, W] logits
        dice_loss = criterion_dice(predicted_seg, seg_map)

        # Combined loss with alpha weighting
        total_loss = (1 - CONFIG['alpha']) * mse_loss + CONFIG['alpha'] * dice_loss
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_mse_loss += mse_loss.item()
        train_dice_loss += dice_loss.item()

    train_loss /= len(train_loader)
    train_mse_loss /= len(train_loader)
    train_dice_loss /= len(train_loader)

    # Evaluation on test set using masked metrics
    recon_model.eval()
    seg_model.eval()
    psnr_scores = []
    ssim_scores = []
    dice_per_case_scores = []
    with torch.no_grad():
        for low_dose, full_dose, seg_map in test_loader:
            low_dose = low_dose.to(device)
            full_dose = full_dose.to(device)
            seg_map = seg_map.to(device).squeeze(1)  # [B, H, W] for segmentation
            reconstructed = recon_model(low_dose)
            # Reconstruction metrics
            psnr_scores.extend(masked_psnr_metric(reconstructed, full_dose, mask))
            ssim_scores.extend(masked_ssim_metric(reconstructed, full_dose, mask))
            # Segmentation metrics
            predicted_seg = seg_model(reconstructed)  # [B, 3, H, W] logits
            dice_scores = calculate_dice(predicted_seg, seg_map, num_classes=CONFIG['num_classes'])
            dice_per_case_scores.extend(dice_scores.cpu().numpy())
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)
    avg_dice_per_case = np.mean(dice_per_case_scores)
    std_dice_per_case = np.std(dice_per_case_scores)

    # Get current learning rate for display
    current_lr = optimizer.param_groups[0]['lr']
    
    # Track metrics in history
    training_history['epoch'].append(epoch + 1)
    training_history['train_loss'].append(train_loss)
    training_history['train_mse_loss'].append(train_mse_loss)
    training_history['train_dice_loss'].append(train_dice_loss)
    training_history['val_psnr'].append(avg_psnr)
    training_history['val_ssim'].append(avg_ssim)
    training_history['val_dice'].append(avg_dice_per_case)
    training_history['learning_rate'].append(current_lr)
    
    # Early stopping logic based on combined metric (0.5 * SSIM + 0.4 * Dice)
    combined_metric = 0.5 * avg_ssim + 0.5 * avg_dice_per_case
    if combined_metric > (0.5 * best_ssim + 0.5 * training_history.get('best_dice', -1.0)):
        best_ssim = avg_ssim
        training_history['best_dice'] = avg_dice_per_case
        best_recon_state = recon_model.state_dict().copy()
        best_seg_state = seg_model.state_dict().copy()
        epochs_without_improvement = 0
        training_history['best_epoch'] = epoch + 1
        training_history['best_ssim'] = best_ssim
        training_history['best_psnr'] = avg_psnr
        training_history['best_ssim_std'] = std_ssim
        training_history['best_psnr_std'] = std_psnr
        training_history['best_dice_std'] = std_dice_per_case
        log_and_print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Total Loss: {train_loss:.4f}, MSE: {train_mse_loss:.4f}, "
              f"Dice: {train_dice_loss:.4f}, LR: {current_lr:.2e}, "
              f"Joint Training -> Masked PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}, "
              f"Masked SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}, Val Dice: {avg_dice_per_case:.4f} ± {std_dice_per_case:.4f} (NEW BEST!)")
    else:
        epochs_without_improvement += 1
        log_and_print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Total Loss: {train_loss:.4f}, MSE: {train_mse_loss:.4f}, "
              f"Dice: {train_dice_loss:.4f}, LR: {current_lr:.2e}, "
              f"Joint Training -> Masked PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}, "
              f"Masked SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}, Val Dice: {avg_dice_per_case:.4f} ± {std_dice_per_case:.4f} "
              f"(No improvement: {epochs_without_improvement}/{CONFIG['patience']})")
    
    # Update learning rate scheduler
    if scheduler is not None:
        if CONFIG['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler.step(combined_metric)  # Step with combined metric for plateau detection
        else:
            scheduler.step()  # Step with epoch for other schedulers
    
    # Check early stopping condition (only after minimum epochs)
    if epoch + 1 >= CONFIG['min_epochs'] and epochs_without_improvement >= CONFIG['patience']:
        log_and_print(f"\nEarly stopping triggered! No improvement in combined metric for {CONFIG['patience']} epochs after minimum {CONFIG['min_epochs']} epochs.")
        log_and_print(f"Loading best models with SSIM: {best_ssim:.4f}, Dice: {training_history['best_dice']:.4f}")
        recon_model.load_state_dict(best_recon_state)
        seg_model.load_state_dict(best_seg_state)
        break

# If training completed without early stopping, use the best models found
if best_recon_state is not None and best_seg_state is not None and epochs_without_improvement < CONFIG['patience']:
    log_and_print(f"\nTraining completed. Loading best models with SSIM: {best_ssim:.4f}, Dice: {training_history['best_dice']:.4f}")
    recon_model.load_state_dict(best_recon_state)
    seg_model.load_state_dict(best_seg_state)

# Save final plots
save_training_plots()

"""#### Final Results"""

# Save model weights with timestamp
timestamped_recon_name = f'joint_recon_unet_{timestamp}.pth'
timestamped_seg_name = f'joint_seg_unet_{timestamp}.pth'
recon_model_path = f'.../model-weights/joint-training/recon_unet/{timestamped_recon_name}'
seg_model_path = f'.../model-weights/joint-training/seg_unet/{timestamped_seg_name}'

# Create model weights directory if it doesn't exist
os.makedirs(os.path.dirname(recon_model_path), exist_ok=True)
torch.save(recon_model.state_dict(), recon_model_path)
torch.save(seg_model.state_dict(), seg_model_path)

torch.save(recon_model.state_dict(), '.../model-weights/latest-weights/joint_training.pth')

log_and_print(f"Reconstruction model saved to: {recon_model_path}")
log_and_print(f"Segmentation model saved to: {seg_model_path}")

# Calculate final metrics with the best models
log_and_print("\nCalculating final metrics with best models...")
recon_model.eval()
seg_model.eval()
final_psnr_scores = []
final_ssim_scores = []
final_dice_scores = []
with torch.no_grad():
    for low_dose, full_dose, seg_map in test_loader:
        low_dose, full_dose = low_dose.to(device), full_dose.to(device)
        seg_map = seg_map.to(device).squeeze(1)  # [B, H, W] for segmentation
        reconstructed = recon_model(low_dose)
        # Reconstruction metrics
        final_psnr_scores.extend(masked_psnr_metric(reconstructed, full_dose, mask))
        final_ssim_scores.extend(masked_ssim_metric(reconstructed, full_dose, mask))
        # Segmentation metrics
        predicted_seg = seg_model(reconstructed)  # [B, 3, H, W] logits
        dice_scores = calculate_dice(predicted_seg, seg_map, num_classes=CONFIG['num_classes'])
        final_dice_scores.extend(dice_scores.cpu().numpy())
final_avg_psnr = np.mean(final_psnr_scores)
final_avg_ssim = np.mean(final_ssim_scores)
final_avg_dice = np.mean(final_dice_scores)
final_std_psnr = np.std(final_psnr_scores)
final_std_ssim = np.std(final_ssim_scores)
final_std_dice = np.std(final_dice_scores)

# Calculate baseline low-dose metrics
log_and_print("Calculating baseline low-dose metrics...")
lowdose_psnr_scores = []
lowdose_ssim_scores = []
with torch.no_grad():
    for low_dose, full_dose, seg_map in test_loader:
        low_dose, full_dose = low_dose.to(device), full_dose.to(device)
        lowdose_psnr_scores.extend(masked_psnr_metric(low_dose, full_dose, mask))
        lowdose_ssim_scores.extend(masked_ssim_metric(low_dose, full_dose, mask))
lowdose_avg_psnr = np.mean(lowdose_psnr_scores)
lowdose_avg_ssim = np.mean(lowdose_ssim_scores)
lowdose_std_psnr = np.std(lowdose_psnr_scores)
lowdose_std_ssim = np.std(lowdose_ssim_scores)

# Convert training history to JSON-serializable format
json_training_history = {
    'epoch': [int(x) for x in training_history['epoch']],
    'train_loss': [float(x) for x in training_history['train_loss']],
    'train_mse_loss': [float(x) for x in training_history['train_mse_loss']],
    'train_dice_loss': [float(x) for x in training_history['train_dice_loss']],
    'val_psnr': [float(x) for x in training_history['val_psnr']],
    'val_ssim': [float(x) for x in training_history['val_ssim']],
    'val_dice': [float(x) for x in training_history['val_dice']],
    'learning_rate': [float(x) for x in training_history['learning_rate']],
    'best_epoch': int(training_history['best_epoch']),
    'best_ssim': float(training_history['best_ssim']),
    'best_psnr': float(training_history['best_psnr']),
    'best_dice': float(training_history['best_dice']),
    'best_ssim_std': float(training_history['best_ssim_std']),
    'best_psnr_std': float(training_history['best_psnr_std']),
    'best_dice_std': float(training_history['best_dice_std'])
}

# Save training history and final results
results_summary = {
    'training_history': json_training_history,
    'final_metrics': {
        'psnr_mean': float(final_avg_psnr),
        'psnr_std': float(final_std_psnr),
        'ssim_mean': float(final_avg_ssim),
        'ssim_std': float(final_std_ssim),
        'dice_mean': float(final_avg_dice),
        'dice_std': float(final_std_dice)
    },
    'lowdose_baseline': {
        'psnr_mean': float(lowdose_avg_psnr),
        'psnr_std': float(lowdose_std_psnr),
        'ssim_mean': float(lowdose_avg_ssim),
        'ssim_std': float(lowdose_std_ssim)
    },
    'best_epoch': int(json_training_history['best_epoch']),
    'total_epochs_trained': len(json_training_history['epoch']),
    'early_stopped': len(json_training_history['epoch']) < CONFIG['num_epochs'],
    'alpha_weight': CONFIG['alpha'],
    'model_paths': {
        'reconstruction': recon_model_path,
        'segmentation': seg_model_path
    }
}

# Save results to JSON
results_path = os.path.join(run_dir, 'training_results.json')
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

log_and_print(f"Training results saved to: {results_path}")

# Log training summary with final metrics comparison
log_and_print("\n" + "="*80)
log_and_print("JOINT TRAINING SUMMARY")
log_and_print("="*80)
log_and_print(f"Total epochs trained: {len(training_history['epoch'])}/{CONFIG['num_epochs']}")
log_and_print(f"Best epoch: {training_history['best_epoch']}")
log_and_print(f"Best SSIM: {training_history['best_ssim']:.4f} ± {training_history['best_ssim_std']:.4f}")
log_and_print(f"Best PSNR: {training_history['best_psnr']:.4f} ± {training_history['best_psnr_std']:.4f}")
log_and_print(f"Best Dice: {training_history['best_dice']:.4f} ± {training_history['best_dice_std']:.4f}")
log_and_print(f"Early stopping: {'Yes' if len(training_history['epoch']) < CONFIG['num_epochs'] else 'No'}")
log_and_print(f"Final learning rate: {training_history['learning_rate'][-1]:.2e}")
log_and_print(f"Alpha weight (MSE vs Dice): {CONFIG['alpha']:.2f}")
log_and_print("="*80)

"""#### Results Visualization"""

def visualize_joint_training_results(recon_model, seg_model, test_dataset, mask, index=0, device='cuda', save_path=None):
    # Set models to evaluation mode
    recon_model.eval()
    seg_model.eval()

    # Fetch data from test_dataset
    low_dose, full_dose, gt_seg = test_dataset[index]

    # Reshape tensors to add batch dimension
    low_dose = low_dose.unsqueeze(0).to(device)    # [1, 1, H, W]
    full_dose = full_dose.unsqueeze(0).to(device)  # [1, 1, H, W]
    gt_seg = gt_seg.squeeze(0).to(device)                     # [H, W]

    # Generate reconstructed image and segmentation using both models
    with torch.no_grad():
        reconstructed = recon_model(low_dose)  # [1, 1, H, W]
        pred_seg_logits = seg_model(reconstructed)  # [1, 3, H, W]
        pred_seg = torch.argmax(pred_seg_logits, dim=1)   # [1, H, W]

    # Compute metrics
    psnr_lowdose = masked_psnr_metric(low_dose, full_dose, mask)[0]
    ssim_lowdose = masked_ssim_metric(low_dose, full_dose, mask)[0]
    psnr_recon = masked_psnr_metric(reconstructed, full_dose, mask)[0]
    ssim_recon = masked_ssim_metric(reconstructed, full_dose, mask)[0]
    dice_score = calculate_dice(pred_seg_logits, gt_seg.unsqueeze(0), num_classes=3)[0].item()

    # Convert tensors to numpy arrays for plotting
    low_dose_np = low_dose.squeeze(0).cpu().numpy().squeeze(0)      # [H, W]
    full_dose_np = full_dose.squeeze(0).cpu().numpy().squeeze(0)    # [H, W]
    gt_seg_np = gt_seg.squeeze(0).cpu().numpy()                     # [H, W]
    reconstructed_np = reconstructed.squeeze(0).cpu().numpy().squeeze(0)  # [H, W]
    pred_seg_np = pred_seg.squeeze(0).cpu().numpy()                 # [H, W]

    # Create a 3x3 subplot grid (we'll use only 7 slots)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # First Row: Low-Dose CT, Joint Training Reconstructed, Ground Truth Full-Dose CT
    # Plot 1: Low-Dose CT
    axes[0, 0].imshow(low_dose_np, cmap='gray')
    axes[0, 0].set_title(f"Low-Dose CT\nPSNR: {psnr_lowdose:.2f}, SSIM: {ssim_lowdose:.3f}")
    axes[0, 0].axis('off')

    # Plot 2: Joint Training Reconstructed
    axes[0, 1].imshow(reconstructed_np, cmap='gray')
    axes[0, 1].set_title(f"Joint Training Reconstructed\nPSNR: {psnr_recon:.2f}, SSIM: {ssim_recon:.3f}")
    axes[0, 1].axis('off')

    # Plot 3: Ground Truth Full-Dose CT
    axes[0, 2].imshow(full_dose_np, cmap='gray')
    axes[0, 2].set_title("Ground Truth Full-Dose CT")
    axes[0, 2].axis('off')

    # Second Row: Segmentation on Reconstruction, Segmentation on Reconstruction with Overlay
    # Plot 4: Segmentation on Reconstruction
    axes[1, 0].imshow(pred_seg_np, cmap='jet', vmin=0, vmax=2)
    axes[1, 0].set_title(f"Segmentation on Reconstructed\nDice: {dice_score:.3f}")
    axes[1, 0].axis('off')

    # Plot 5: Segmentation on Reconstruction with Overlay
    axes[1, 1].imshow(reconstructed_np, cmap='gray')
    axes[1, 1].imshow(pred_seg_np, cmap='jet', alpha=0.5, vmin=0, vmax=2)
    axes[1, 1].set_title(f"Reconstructed with Segmentation\nDice: {dice_score:.3f}")
    axes[1, 1].axis('off')

    # Hide unused subplot in second row
    axes[1, 2].axis('off')

    # Third Row: Ground Truth Segmentation, Ground Truth with Segmentation Overlay
    # Plot 6: Ground Truth Segmentation
    axes[2, 0].imshow(gt_seg_np, cmap='jet', vmin=0, vmax=2)
    axes[2, 0].set_title("Ground Truth Segmentation")
    axes[2, 0].axis('off')

    # Plot 7: Ground Truth with Segmentation Overlay
    axes[2, 1].imshow(full_dose_np, cmap='gray')
    axes[2, 1].imshow(gt_seg_np, cmap='jet', alpha=0.5, vmin=0, vmax=2)
    axes[2, 1].set_title("Ground Truth with Segmentation")
    axes[2, 1].axis('off')

    # Hide unused subplot in third row
    axes[2, 2].axis('off')

    # Adjust layout and display
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_and_print(f"Joint training visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()

# Create save path for the joint training visualization
joint_training_save_path = os.path.join(run_dir, "joint_training_visualization_sample_5.png")

# Call the visualization function with the trained models
visualize_joint_training_results(
    recon_model=recon_model, 
    seg_model=seg_model, 
    test_dataset=test_dataset, 
    mask=mask, 
    index=3, 
    device=device,
    save_path=joint_training_save_path
)

# Log completion and close log file
log_and_print(f"\nJoint training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print(f"All outputs saved to: {run_dir}")
log_file.close()
