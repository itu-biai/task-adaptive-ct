"""#### Install Libraries"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

"""#### Dataset"""

# Function to preload .npy files from a folder
def load_npy_from_folder(folder_path):
    images = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        img = np.load(file_path)  # Shape: (H, W)
        images.append(img)
    return np.stack(images, axis=0)  # Shape: (N, H, W)

# Unified Dataset for Task-Adaptive Training
class TaskAdaptiveUnifiedDataset(Dataset):
    def __init__(self, low_dose_dir, full_dose_dir, seg_dir, transform_recon=None, transform_seg=None):
        self.transform_recon = transform_recon
        self.transform_seg = transform_seg

        # Preload all data
        self.low_dose = load_npy_from_folder(low_dose_dir)    # Shape: (N, H, W)
        self.full_dose = load_npy_from_folder(full_dose_dir)  # Shape: (N, H, W)
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

        seg_map = seg_map.astype(np.int64)

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

"""#### Dice Loss"""

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

# Define directories
low_dose_dir = 'low_dose'
full_dose_dir = 'full_dose'
seg_dir = 'segmentation_slices'

# Define transform for images
transform_recon = transforms.ToTensor()  # Images are already normalized to [0, 1] during dataset creation
transform_seg = transforms.ToTensor()    # For segmentation masks

# Instantiate the unified dataset
dataset = TaskAdaptiveUnifiedDataset(low_dose_dir, full_dose_dir, seg_dir, transform_recon=transform_recon, transform_seg=transform_seg)

# Split dataset into training and testing sets (80-20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pre-trained segmentation model
seg_model = U_Net(in_channels=1, out_channels=3).to(device)
seg_model.load_state_dict(torch.load('unet_segmentation_state.pth'))
seg_model.eval()
for param in seg_model.parameters():
    param.requires_grad = False

"""#### Task-Adapted Training"""

# Define reconstruction model
recon_model = U_Net(in_channels=1, out_channels=1).to(device)

# Optimizer and Loss
optimizer = optim.Adam(recon_model.parameters(), lr=1e-3)
criterion_mse = nn.MSELoss()

# Loss weights
w1 = 1.0  # Weight for MSE loss
w2 = 1.0  # Weight for Dice loss

# Create circular mask
mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)

# Training loop
num_epochs = 35
for epoch in range(num_epochs):
    recon_model.train()
    train_loss = 0.0
    train_mse_loss = 0.0
    train_dice_loss = 0.0
    for low_dose, full_dose, seg_map in train_loader:
        low_dose = low_dose.to(device)
        full_dose = full_dose.to(device)
        seg_map = seg_map.to(device)

        optimizer.zero_grad()

        # Reconstruction
        reconstructed = recon_model(low_dose)  # [B, 1, H, W], in [0, 1]
        mse_loss = criterion_mse(reconstructed, full_dose)

        # Segmentation regularization
        predicted_seg = seg_model(reconstructed)  # [B, 3, H, W] logits
        dice_per_case = calculate_dice(predicted_seg, seg_map, num_classes=3)
        dice_loss = 1 - dice_per_case.mean()

        # Combined loss
        total_loss = w1 * mse_loss + w2 * dice_loss
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_mse_loss += mse_loss.item()
        train_dice_loss += dice_loss.item()

    train_loss /= len(train_loader)
    train_mse_loss /= len(train_loader)
    train_dice_loss /= len(train_loader)

    # Evaluation
    recon_model.eval()
    psnr_scores = []
    ssim_scores = []
    with torch.no_grad():
        for low_dose, full_dose, seg_map in test_loader:
            low_dose = low_dose.to(device)
            full_dose = full_dose.to(device)
            reconstructed = recon_model(low_dose)
            psnr_values = masked_psnr_metric(reconstructed, full_dose, mask)
            ssim_values = masked_ssim_metric(reconstructed, full_dose, mask)
            psnr_scores.extend(psnr_values)
            ssim_scores.extend(ssim_values)
    avg_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {train_loss:.4f}, MSE Loss: {train_mse_loss:.4f}, Dice Loss: {train_dice_loss:.4f}, "
          f"PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}, SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

# Save the trained reconstruction model
torch.save(recon_model.state_dict(), 'task_adaptive_model.pth')

"""#### Final Results"""

# Final metrics for low-dose images
lowdose_psnr_scores = []
lowdose_ssim_scores = []
with torch.no_grad():
    for low_dose, full_dose, seg_map in test_loader:
        low_dose = low_dose.to(device)
        full_dose = full_dose.to(device)
        psnr_values = masked_psnr_metric(low_dose, full_dose, mask)
        ssim_values = masked_ssim_metric(low_dose, full_dose, mask)
        lowdose_psnr_scores.extend(psnr_values)
        lowdose_ssim_scores.extend(ssim_values)
lowdose_avg_psnr = np.mean(lowdose_psnr_scores)
lowdose_std_psnr = np.std(lowdose_psnr_scores)
lowdose_avg_ssim = np.mean(lowdose_ssim_scores)
lowdose_std_ssim = np.std(lowdose_ssim_scores)

# Print final metrics with improved alignment
print("\nFinal Metrics Comparison")
print("-" * 80)
# Define column widths for better alignment
col1_width = 30  # Method
col2_width = 25  # PSNR (Mean ± Std)
col3_width = 25  # SSIM (Mean ± Std)

# Header
header = f"{'Method':<{col1_width}} {'PSNR (Mean ± Std)':<{col2_width}} {'SSIM (Mean ± Std)':<{col3_width}}"
print(header)
print("-" * 80)

# Row 1: Low-Dose Images
psnr_str1 = f"{lowdose_avg_psnr:.4f} ± {lowdose_std_psnr:.4f}"
ssim_str1 = f"{lowdose_avg_ssim:.4f} ± {lowdose_std_ssim:.4f}"
row1 = f"{'Low-Dose Images':<{col1_width}} {psnr_str1:<{col2_width}} {ssim_str1:<{col3_width}}"
print(row1)

# Row 2: Task Adaptive Reconstructed
psnr_str2 = f"{avg_psnr:.4f} ± {std_psnr:.4f}"
ssim_str2 = f"{avg_ssim:.4f} ± {std_ssim:.4f}"
row2 = f"{'Task Adaptive Reconstructed':<{col1_width}} {psnr_str2:<{col2_width}} {ssim_str2:<{col3_width}}"
print(row2)

print("-" * 80)

print("Training completed and model saved as 'task_adaptive_recon_model.pth'.")

"""#### Results Visualization"""

def visualize_task_adaptive_results(recon_model, seg_model, test_dataset, mask, index=0, device='cuda'):
    # Set models to evaluation mode
    recon_model.eval()
    seg_model.eval()

    # Fetch data from test_dataset
    low_dose, full_dose, gt_seg = test_dataset[index]

    # Reshape tensors to add batch dimension
    low_dose = low_dose.unsqueeze(0).to(device)    # [1, 1, H, W]
    full_dose = full_dose.unsqueeze(0).to(device)  # [1, 1, H, W]
    gt_seg = gt_seg.to(device)                     # [1, H, W]

    # Generate reconstructed image using the reconstruction model
    with torch.no_grad():
        reconstructed = recon_model(low_dose)  # [1, 1, H, W]
        pred_seg_logits = seg_model(reconstructed)  # [1, 3, H, W]
        pred_seg = torch.argmax(pred_seg_logits, dim=1)   # [1, H, W]

    # Compute metrics
    psnr_lowdose = masked_psnr_metric(low_dose, full_dose, mask)[0]
    ssim_lowdose = masked_ssim_metric(low_dose, full_dose, mask)[0]
    psnr_recon = masked_psnr_metric(reconstructed, full_dose, mask)[0]
    ssim_recon = masked_ssim_metric(reconstructed, full_dose, mask)[0]
    dice_score = calculate_dice(pred_seg_logits, gt_seg, num_classes=3)[0].item()

    # Convert tensors to numpy arrays for plotting
    low_dose_np = low_dose.squeeze(0).cpu().numpy().squeeze(0)      # [H, W]
    full_dose_np = full_dose.squeeze(0).cpu().numpy().squeeze(0)    # [H, W]
    gt_seg_np = gt_seg.squeeze(0).cpu().numpy()                     # [H, W]
    reconstructed_np = reconstructed.squeeze(0).cpu().numpy().squeeze(0)  # [H, W]
    pred_seg_np = pred_seg.squeeze(0).cpu().numpy()                 # [H, W]

    # Create a 3x3 subplot grid (we'll use only 7 slots)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # First Row: Low-Dose CT, Task Adaptive Reconstructed, Ground Truth Full-Dose CT
    # Plot 1: Low-Dose CT
    axes[0, 0].imshow(low_dose_np, cmap='gray')
    axes[0, 0].set_title(f"Low-Dose CT\nPSNR: {psnr_lowdose:.2f}, SSIM: {ssim_lowdose:.3f}")
    axes[0, 0].axis('off')

    # Plot 2: Task Adaptive Reconstructed
    axes[0, 1].imshow(reconstructed_np, cmap='gray')
    axes[0, 1].set_title(f"Task Adaptive Reconstructed\nPSNR: {psnr_recon:.2f}, SSIM: {ssim_recon:.3f}")
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
    plt.show()

visualize_task_adaptive_results(recon_model, seg_model, test_dataset, mask, index=120, device='cuda')