"""#### Install Libraries"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import json
from tqdm import tqdm

"""#### CONFIG"""

CONFIG = {
    # Inference parameters
    'batch_size': 64,
    'num_workers': 64,
    'train_test_split': 0.8,  # 80% training, 20% testing
    'num_classes': 3,  # Number of segmentation classes
    
    # Dataset directories
    'low_dose_dir': '/YEREL/dataset/low_dose',
    'full_dose_dir': '/YEREL/dataset/full_dose',
    'seg_dir': '/YEREL/dataset/segmentation_slices',
    
    # Model weights paths
    'seg_model_path': '/ari/users/nsefercioglu/model-weights/seg-unet/seg_unet_20250815_112913.pth',
    'task_adaptive_model_path': '/ari/users/nsefercioglu/model-weights/task-adaptive/task_adaptive_unet_20250815_150041.pth',
    
    # Output directory for plots and logs
    'output_dir': '/ari/users/nsefercioglu/inference-results/task-adaptive'
}

"""#### Output Directory and Logging Setup"""

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Create unique run directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(CONFIG['output_dir'], f"task_adaptive_inference_run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

print(f"Task adaptive segmentation inference outputs will be saved to: {run_dir}")

# Save configuration to run directory
config_path = os.path.join(run_dir, "config.json")
config_to_save = CONFIG.copy()
with open(config_path, 'w') as f:
    json.dump(config_to_save, f, indent=2)

# Initialize log file
log_file_path = os.path.join(run_dir, "task_adaptive_inference_log.txt")
log_file = open(log_file_path, 'w')
log_file.write(f"Task Adaptive Segmentation Inference Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    """Calculate Dice coefficient for segmentation evaluation"""
    # If pred is logits, apply softmax or sigmoid
    if num_classes > 1:
        pred = torch.softmax(pred, dim=1)  # For multi-class, softmax over channel dimension
    else:
        pred = torch.sigmoid(pred)  # For binary, sigmoid

    # Convert predictions to binary masks
    pred = (pred > threshold).float()  # Threshold to get binary predictions

    # If target is not one-hot encoded (e.g., [B, H, W] with integer labels), convert it
    if target.dim() == 3:  # [B, H, W]
        target_one_hot = torch.zeros_like(pred)  # [B, C, H, W]
        for c in range(num_classes):
            target_one_hot[:, c, :, :] = (target == c).float()
        target = target_one_hot
    elif target.dim() == 4 and target.shape[1] != num_classes:
        raise ValueError("Target shape must match [B, C, H, W] or [B, H, W] with integer labels")

    # Calculate intersection and sum of predictions and targets
    intersection = (pred * target).sum(dim=[2, 3])  # Sum over H and W dimensions
    pred_sum = pred.sum(dim=[2, 3])
    target_sum = target.sum(dim=[2, 3])

    # Dice per case (per sample in batch) for each class
    dice_per_case = (2 * intersection + eps) / (pred_sum + target_sum + eps)  # [B, C]

    # Mean Dice per case across classes
    mean_dice_per_case = dice_per_case.mean(dim=1)  # [B]

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

# Load Task-adaptive reconstruction model
task_adaptive_model = U_Net(in_channels=1, out_channels=1).to(device)
task_adaptive_model.load_state_dict(torch.load(CONFIG['task_adaptive_model_path'], map_location=device))
task_adaptive_model.eval()
log_and_print(f"Task-adaptive reconstruction model loaded from: {CONFIG['task_adaptive_model_path']}")

"""#### Task Adaptive Segmentation Inference"""

def evaluate_task_adaptive_segmentation():
    """Evaluate segmentation performance on task adaptive reconstructed images"""
    
    log_and_print(f"\nStarting task adaptive segmentation inference...")
    log_and_print(f"Processing {len(test_loader)} batches with batch size {CONFIG['batch_size']} ({len(test_loader.dataset)} total samples)")
    
    # Initialize result storage
    results = []
    
    # Process test dataset
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Task Adaptive Segmentation", unit="batch")
        for batch_idx, (low_dose, full_dose, seg_map) in enumerate(progress_bar):
            low_dose = low_dose.to(device)
            full_dose = full_dose.to(device)
            seg_map = seg_map.to(device)
            
            # Task-adaptive reconstructed images segmentation
            task_adaptive_reconstructed = task_adaptive_model(low_dose)
            seg_pred_task_adaptive = seg_model(task_adaptive_reconstructed)
            dice_task_adaptive = calculate_dice(seg_pred_task_adaptive, seg_map, num_classes=CONFIG['num_classes'])
            results.extend(dice_task_adaptive.cpu().numpy())
            
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
    log_and_print("TASK ADAPTIVE SEGMENTATION INFERENCE RESULTS")
    log_and_print("="*80)
    
    # Calculate statistics
    dice_scores = np.array(results)
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    statistics = {
        'mean': float(mean_dice),
        'std': float(std_dice),
        'count': len(dice_scores)
    }
    
    # Display results
    log_and_print(f"Task Adaptive Segmentation:")
    log_and_print(f"  Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
    log_and_print(f"  Sample Count: {len(dice_scores)}")
    log_and_print("")
    
    # Save detailed results to JSON
    results_summary = {
        'statistics': statistics,
        'raw_results': [float(x) for x in results],
        'config': config_to_save,
        'timestamp': timestamp
    }
    
    results_path = os.path.join(run_dir, 'task_adaptive_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    log_and_print(f"Detailed results saved to: {results_path}")
    
    # Save summary results to text file
    summary_path = os.path.join(run_dir, 'task_adaptive_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Task Adaptive Segmentation Results Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Task Adaptive Segmentation:\n")
        f.write(f"  Dice Score: {statistics['mean']:.4f} ± {statistics['std']:.4f}\n")
        f.write(f"  Sample Count: {statistics['count']}\n")
    
    log_and_print(f"Summary results saved to: {summary_path}")
    
    return statistics



"""#### Main Execution"""

if __name__ == "__main__":
    try:
        # Run task adaptive segmentation inference
        results = evaluate_task_adaptive_segmentation()
        
        # Analyze and save results
        statistics = analyze_and_save_results(results)
        
        # Log completion
        log_and_print(f"\nTask adaptive segmentation inference completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_and_print(f"All outputs saved to: {run_dir}")
        
    except Exception as e:
        log_and_print(f"Error during task adaptive segmentation inference: {str(e)}")
        import traceback
        log_and_print(f"Traceback: {traceback.format_exc()}")
        
    finally:
        # Close log file
        log_file.close()
