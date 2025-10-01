"""#### Install Libraries"""

import torch
import numpy as np
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
    'min_epochs': 20,  # Minimum epochs before early stopping can trigger
    
    # Dataset directories
    'image_dir': '.../dataset/full_dose',
    'mask_dir': '.../dataset/segmentation_slices',
    
    # Output directory for plots and logs
    'output_dir': '.../logs/segmentation-unet', 
    
    # Loss parameters
    'criterion': nn.CrossEntropyLoss()
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
    'val_dice': [],
    'learning_rate': [],
    'best_epoch': 0,
    'best_dice': -1.0,
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
    
    # Plot 1: Training Loss
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Validation Dice
    ax2.plot(epochs, training_history['val_dice'], 'g-', label='Validation Dice', linewidth=2)
    # Mark best epoch
    if training_history['best_epoch'] > 0:
        best_idx = training_history['best_epoch'] - 1
        ax2.scatter(epochs[best_idx], training_history['val_dice'][best_idx], 
                   color='red', s=100, marker='*', label=f'Best (Epoch {training_history["best_epoch"]})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Learning Rate
    ax3.plot(epochs, training_history['learning_rate'], 'm-', label='Learning Rate', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Keep empty or add additional metrics if needed
    ax4.axis('off')
    ax4.text(0.5, 0.5, 'Additional metrics\ncan be added here', 
             ha='center', va='center', fontsize=12, style='italic')
    
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
    # Sorting ensures pairing with corresponding masks
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        img = np.load(file_path)  # Shape: (H, W), already normalized to [0, 1] for images
        images.append(img)
    return np.stack(images, axis=0)  # Shape: (N, H, W)

# Dataset for paired image-mask data with preloading
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.transform = transform

        # Preload all images and masks into memory
        self.images = load_npy_from_folder(image_dir)  # Shape: (N, H, W), already normalized to [0, 1]
        self.masks = load_npy_from_folder(mask_dir)    # Shape: (N, H, W)

        # Ensure the number of images and masks match
        assert len(self.images) == len(self.masks), "Mismatch between images and masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (H, W)
        mask = self.masks[idx]    # Shape: (H, W), integer labels

        # Ensure correct data types
        image = image.astype(np.float32)  # Keep float32 for PyTorch
        mask = mask.astype(np.int64)  # For CrossEntropyLoss

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)  # Transform pipeline includes conversion to tensor
            mask = torch.from_numpy(mask)  # (H, W) -> (H, W) tensor

        return image, mask

# Define transform pipeline for images
transform = transforms.ToTensor()  # Images are already normalized to [0, 1] during dataset creation

# Create dataset with preloaded data
dataset = SegmentationDataset(CONFIG['image_dir'], CONFIG['mask_dir'], transform=transform)

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

# Create circular mask
mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)

"""#### Dice Metric"""

def calculate_dice(pred, target, num_classes=3, threshold=0.5, eps=1e-6, mask=None):
    """Argmax-based Dice for multi-class segmentation; thresholded for binary."""
    if num_classes > 1:
        # Convert logits to class labels via argmax
        pred_labels = torch.argmax(pred, dim=1)  # [B, H, W]
        if target.dim() == 4 and target.shape[1] == num_classes:
            target_labels = torch.argmax(target, dim=1)
        else:
            target_labels = target
        pred = torch.nn.functional.one_hot(pred_labels, num_classes).permute(0,3,1,2).float()
        target = torch.nn.functional.one_hot(target_labels.long(), num_classes).permute(0,3,1,2).float()
    else:
        # Binary case retains thresholding behaviour
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

    intersection = (pred * target).sum(dim=[2, 3])
    pred_sum = pred.sum(dim=[2, 3])
    target_sum = target.sum(dim=[2, 3])
    dice_per_case = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    mean_dice_per_case = dice_per_case.mean(dim=1)
    return mean_dice_per_case

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
    def __init__(self, in_channels=1, out_channels=3):
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

        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U_Net(in_channels=1, out_channels=CONFIG['num_classes']).to(device)

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
            mode='max',  # We want to maximize Dice
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

# ----- Training loop with Early Stopping -----
# Early stopping variables
best_dice = -1.0
epochs_without_improvement = 0
best_model_state = copy.deepcopy(model.state_dict())

log_and_print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
log_and_print(f"Device: {device}")
log_and_print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(CONFIG['num_epochs']):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)  # masks: [B, H, W] with integer labels
        optimizer.zero_grad()
        output = model(images)  # [B, C, H, W] logits
        loss = criterion(output, masks)  # CrossEntropyLoss expects [B, C, H, W] logits, [B, H, W] labels
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Evaluation on test set
    model.eval()
    dice_per_case_scores = []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)  # [B, C, H, W] logits
            dice_scores = calculate_dice(output, masks, num_classes=CONFIG['num_classes'], mask=mask)  # Compute Dice
            dice_per_case_scores.extend(dice_scores.cpu().numpy())  # Store per-case Dice

    # Compute statistics
    avg_dice_per_case = np.mean(dice_per_case_scores)
    std_dice_per_case = np.std(dice_per_case_scores)

    # Get current learning rate for display
    current_lr = optimizer.param_groups[0]['lr']
    
    # Track metrics in history
    training_history['epoch'].append(epoch + 1)
    training_history['train_loss'].append(train_loss)
    training_history['val_dice'].append(avg_dice_per_case)
    training_history['learning_rate'].append(current_lr)
    
    # Early stopping logic
    if avg_dice_per_case > best_dice:
        best_dice = avg_dice_per_case
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        training_history['best_epoch'] = epoch + 1
        training_history['best_dice'] = best_dice
        training_history['best_dice_std'] = std_dice_per_case
        log_and_print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {train_loss:.4f}, LR: {current_lr:.2e}, "
              f"U-Net Segmentation -> Dice Per Case: {avg_dice_per_case:.4f} ± {std_dice_per_case:.4f} (NEW BEST!)")
    else:
        epochs_without_improvement += 1
        log_and_print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {train_loss:.4f}, LR: {current_lr:.2e}, "
              f"U-Net Segmentation -> Dice Per Case: {avg_dice_per_case:.4f} ± {std_dice_per_case:.4f} "
              f"(No improvement: {epochs_without_improvement}/{CONFIG['patience']})")
    
    # Update learning rate scheduler
    if scheduler is not None:
        if CONFIG['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler.step(avg_dice_per_case)  # Step with Dice for plateau detection
        else:
            scheduler.step()  # Step with epoch for other schedulers
    
    # Check early stopping condition
    if epoch + 1 >= CONFIG['min_epochs'] and epochs_without_improvement >= CONFIG['patience']:
        log_and_print(f"\nEarly stopping triggered! No improvement in Dice for {CONFIG['patience']} epochs.")
        log_and_print(f"Loading best model with Dice: {best_dice:.4f}")
        model.load_state_dict(best_model_state)
        break

# If training completed without early stopping, use the best model found
if best_model_state is not None and epochs_without_improvement < CONFIG['patience']:
    log_and_print(f"\nTraining completed. Loading best model with Dice: {best_dice:.4f}")
    model.load_state_dict(best_model_state)

# Save final plots
save_training_plots()

trained_Unet = model

# Save model weights with constant name
model_name = 'seg_unet.pth'
original_model_path = f'.../model-weights/{model_name}'

torch.save(trained_Unet.state_dict(), original_model_path)

log_and_print(f"Model saved to: {original_model_path}")

# Calculate final metrics with the best model
log_and_print("\nCalculating final metrics with best model...")
model.eval()
final_dice_scores = []
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        output = model(images)
        dice_scores = calculate_dice(output, masks, num_classes=CONFIG['num_classes'], mask=mask)
        final_dice_scores.extend(dice_scores.cpu().numpy())
final_avg_dice = np.mean(final_dice_scores)
# ------------------------------------------------------------------------------
# Quick verification: reload saved segmentation weights and recompute Dice metric
# ------------------------------------------------------------------------------
log_and_print("\nVerifying saved segmentation model by reloading from disk ...")
reloaded_model = U_Net(in_channels=1, out_channels=CONFIG['num_classes']).to(device)
reloaded_model.load_state_dict(torch.load(original_model_path, map_location=device))
reloaded_model.eval()

verify_dice_scores = []
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        output = reloaded_model(images)
        dice_scores = calculate_dice(output, masks, num_classes=CONFIG['num_classes'], mask=mask)
        verify_dice_scores.extend(dice_scores.cpu().numpy())

verify_avg_dice = np.mean(verify_dice_scores)
log_and_print(f"Reloaded model -> Dice Per Case: {verify_avg_dice:.4f}")
if np.isclose(verify_avg_dice, final_avg_dice):
    log_and_print("Verification successful: saved model matches in-memory best model.")
else:
    log_and_print("WARNING: Reloaded model metrics differ! Check saving logic.")

final_std_dice = np.std(final_dice_scores)

# Convert training history to JSON-serializable format
json_training_history = {
    'epoch': [int(x) for x in training_history['epoch']],
    'train_loss': [float(x) for x in training_history['train_loss']],
    'val_dice': [float(x) for x in training_history['val_dice']],
    'learning_rate': [float(x) for x in training_history['learning_rate']],
    'best_epoch': int(training_history['best_epoch']),
    'best_dice': float(training_history['best_dice']),
    'best_dice_std': float(training_history['best_dice_std'])
}

# Save training history and final results
results_summary = {
    'training_history': json_training_history,
    'final_metrics': {
        'dice_mean': float(final_avg_dice),
        'dice_std': float(final_std_dice)
    },
    'best_epoch': int(json_training_history['best_epoch']),
    'total_epochs_trained': len(json_training_history['epoch']),
    'early_stopped': len(json_training_history['epoch']) < CONFIG['num_epochs']
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
log_and_print(f"Best Dice: {training_history['best_dice']:.4f} ± {training_history['best_dice_std']:.4f}")
log_and_print(f"Early stopping: {'Yes' if len(training_history['epoch']) < CONFIG['num_epochs'] else 'No'}")
log_and_print(f"Final learning rate: {training_history['learning_rate'][-1]:.2e}")
log_and_print("="*80)

# Close log file
log_and_print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print(f"All outputs saved to: {run_dir}")

"""#### Final Results"""

def plot_segmentation_inference(model, dataset, sample_idx=0, num_classes=3, device='cuda' if torch.cuda.is_available() else 'cpu', save_path=None):
    model.eval()
    model.to(device)
    image, ground_truth = dataset[sample_idx]
    image = image.unsqueeze(0).to(device)  # [1, C_in, H, W]

    # Handle ground_truth: if it's a tensor, use it directly; if it's a numpy array, convert it
    if isinstance(ground_truth, torch.Tensor):
        ground_truth_tensor = ground_truth.unsqueeze(0).to(device)  # [1, H, W] for Dice calculation
        ground_truth_np = ground_truth.cpu().numpy()  # Convert to numpy for plotting
    else:
        ground_truth_tensor = torch.from_numpy(ground_truth).unsqueeze(0).to(device)  # [1, H, W]
        ground_truth_np = ground_truth  # Keep as numpy for plotting

    with torch.no_grad():
        output = model(image)  # [1, C_out, H, W] logits

    # Compute Dice coefficient
    dice_metrics = calculate_dice(output, ground_truth_tensor, num_classes=num_classes, mask=mask)

    # Extract the Dice score for the single sample
    dice_per_case = dice_metrics[0].item()  # mean_dice_per_case is [B], so take the first element

    # Get predicted segmentation
    if num_classes > 1:
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]
    else:
        pred = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy()  # [H, W]

    image_np = image.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

    # First Row
    # Plot 1: Original Image
    axes[0, 0].imshow(image_np, cmap='gray')
    axes[0, 0].set_title('Original CT')
    axes[0, 0].axis('off')

    # Plot 2: Standalone Predicted Segmentation
    axes[0, 1].imshow(pred, cmap='jet', vmin=0, vmax=num_classes-1)
    axes[0, 1].set_title('Predicted Segmentation')
    axes[0, 1].axis('off')

    # Plot 3: Standalone Ground Truth
    axes[0, 2].imshow(ground_truth_np, cmap='jet', vmin=0, vmax=num_classes-1)
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')

    # Second Row
    # Plot 4: Original Image (repeated for reference)
    axes[1, 0].imshow(image_np, cmap='gray')
    axes[1, 0].set_title('Original CT')
    axes[1, 0].axis('off')

    # Plot 5: Predicted Segmentation Overlay with Dice
    axes[1, 1].imshow(image_np, cmap='gray')  # Base CT image
    axes[1, 1].imshow(pred, cmap='jet', alpha=0.5, vmin=0, vmax=num_classes-1)  # Overlay Predicted Segmentation
    axes[1, 1].set_title(f'Predicted Overlay (Dice: {dice_per_case:.4f})')
    axes[1, 1].axis('off')

    # Plot 6: Ground Truth Overlay
    axes[1, 2].imshow(image_np, cmap='gray')  # Base CT image
    axes[1, 2].imshow(ground_truth_np, cmap='jet', alpha=0.5, vmin=0, vmax=num_classes-1)  # Overlay Ground Truth
    axes[1, 2].set_title('Ground Truth Overlay')
    axes[1, 2].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_and_print(f"Segmentation visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()

# Create save path for the segmentation visualization
segmentation_save_path = os.path.join(run_dir, "segmentation_visualization.png")

# Call the function with your trained U-Net and test dataset
plot_segmentation_inference(
    model=trained_Unet, 
    dataset=test_dataset, 
    sample_idx=5, 
    num_classes=CONFIG['num_classes'], 
    device=device,
    save_path=segmentation_save_path
)

log_file.close()
