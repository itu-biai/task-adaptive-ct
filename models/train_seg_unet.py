"""#### Install Libraries"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
import os

"""#### Dataset"""

# Function to preload .npy files from a folder
def load_npy_from_folder(folder_path):
    images = []
    # Sorting ensures pairing with corresponding masks
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        img = np.load(file_path)  # Shape: (H, W)
        images.append(img)
    return np.stack(images, axis=0)  # Shape: (N, H, W)

# Dataset for paired image-mask data with preloading
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.transform = transform

        # Preload all images and masks into memory
        self.images = load_npy_from_folder(image_dir)  # Shape: (N, H, W)
        self.masks = load_npy_from_folder(mask_dir)    # Shape: (N, H, W)

        # Ensure the number of images and masks match
        assert len(self.images) == len(self.masks), "Mismatch between images and masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (H, W)
        mask = self.masks[idx]    # Shape: (H, W), integer labels

        # Ensure correct data types
        image = image.astype(np.float32)
        mask = mask.astype(np.int64)  # For CrossEntropyLoss

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)  # Transform pipeline includes normalization and conversion to tensor
            mask = torch.from_numpy(mask)  # (H, W) -> (H, W) tensor

        return image, mask

# ----- Prepare dataset and dataloaders -----
image_dir = 'full_dose'       # Folder with .npy image files
mask_dir = 'segmentation_slices'  # Folder with .npy mask files

# Define transform pipeline for images
transform = transforms.ToTensor()  # Images are already normalized to [0, 1] during dataset creation

# Create dataset with preloaded data
dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)

# Split dataset 80-20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

"""#### Dice Metric"""

def calculate_dice(pred, target, num_classes=2, threshold=0.5, eps=1e-6):
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
num_classes = 3  # Adjust based on your segmentation task
model = U_Net(in_channels=1, out_channels=num_classes).to(device)  # Update in_channels if needed
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # For multi-class segmentation (logits input)

# ----- Training loop -----
num_epochs = 30
for epoch in range(num_epochs):
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
            dice_scores = calculate_dice(output, masks, num_classes=num_classes)  # Compute Dice
            dice_per_case_scores.extend(dice_scores.cpu().numpy())  # Store per-case Dice

    # Compute statistics
    avg_dice_per_case = np.mean(dice_per_case_scores)
    std_dice_per_case = np.std(dice_per_case_scores)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, "f"U-Net Segmentation -> Dice Per Case: {avg_dice_per_case:.4f} ± {std_dice_per_case:.4f}")

trained_Unet = model

torch.save(trained_Unet.state_dict(), 'unet_segmentation_state.pth')

"""#### Final Results"""

def plot_segmentation_inference(model, dataset, sample_idx=0, num_classes=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    dice_metrics = calculate_dice(output, ground_truth_tensor, num_classes=num_classes)

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
    plt.show()

# Call the function with your trained U-Net and test dataset
plot_segmentation_inference(trained_Unet, test_dataset, sample_idx=98, num_classes=num_classes)