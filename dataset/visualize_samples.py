import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
from matplotlib import colors

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
    def __init__(self, low_dose_dir, full_dose_dir, seg_dir):
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

        # Convert segmentation map to int64
        seg_map = seg_map.astype(np.int64)

        # Return min and max of full_dose_img as tensors
        full_dose_min = float(full_dose_img.min())
        full_dose_max = float(full_dose_img.max())

        return (low_dose_img, full_dose_img, seg_map,
                torch.tensor(full_dose_min, dtype=torch.float32),
                torch.tensor(full_dose_max, dtype=torch.float32))

# Set up dataset
low_dose_dir = "..."  
full_dose_dir = "..."  
seg_dir = "..."

# Instantiate dataset
dataset = TaskAdaptiveUnifiedDataset(
    low_dose_dir=low_dose_dir,
    full_dose_dir=full_dose_dir,
    seg_dir=seg_dir
)

# Select random samples
num_samples = 3
indices = random.sample(range(len(dataset)), num_samples)

# Create 3x3 subplot with adjusted figure size
fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(14, 16))

# Titles for columns
col_titles = ["Low Dose", "Full Dose", "Segmentation Map"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=12)

# Define class labels for segmentation map
class_labels = {0: "Background", 1: "Liver", 2: "Tumor"}
num_classes = len(class_labels)

# Create a discrete colormap for segmentation maps
cmap = plt.get_cmap('viridis', num_classes)
norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, num_classes, 1), ncolors=num_classes)

# Plot each sample
for row, idx in enumerate(indices):
    # Get sample
    low_dose_img, full_dose_img, seg_map, _, _ = dataset[idx]

    # Plot low-dose image
    axes[row, 0].imshow(low_dose_img, cmap='gray')
    axes[row, 0].axis('off')

    # Plot full-dose image
    axes[row, 1].imshow(full_dose_img, cmap='gray')
    axes[row, 1].axis('off')

    # Plot segmentation map with discrete colormap
    im = axes[row, 2].imshow(seg_map, cmap=cmap, norm=norm)
    axes[row, 2].axis('off')

# Add a horizontal colorbar below the figure
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', location='bottom', fraction=0.05, pad=0.1, aspect=40)
cbar.set_ticks(np.arange(num_classes))  # Set ticks at class boundaries
cbar.set_label("Segmentation Classes")
cbar.ax.set_xticklabels([class_labels[i] for i in range(num_classes)])  # Set class labels

# Adjust layout to prevent overlap and ensure alignment
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.20, wspace=0.05, hspace=0.05)

plt.show()