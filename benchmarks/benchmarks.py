"""#### Install Libraries"""

import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import transforms
import torch.nn.functional as F
from skimage.transform import radon, iradon
import bm3d
import matplotlib.pyplot as plt
import os

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

# Custom transform to normalize NumPy array to [0, 1]
class NormalizeTo01:
    def __call__(self, img):
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max != img_min:  # Avoid division by zero
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)  # If image is constant, set to 0
        return img

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

# ----- Prepare dataset and dataloaders -----
noisy_dir = 'low_dose'    # Folder with .npy noisy image files
clean_dir = 'full_dose'   # Folder with .npy clean image files

# Define transform (for both noisy and clean images)
transform = transforms.Compose([
    NormalizeTo01(),          # Normalize to [0, 1]
    transforms.ToTensor()     # Convert to tensor (1, H, W)
])

# Create dataset with preloaded data
dataset = DenoisingDataset(noisy_dir, clean_dir, transform=transform)

# Split dataset 80-20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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

# Calculate metrics for low-dose images
lowdose_psnr_scores = []
lowdose_ssim_scores = []

# No need for model, just evaluate noisy vs clean directly
with torch.no_grad():
    for noisy, clean in test_loader:
        lowdose_psnr_scores.extend(psnr_metric(noisy, clean))
        lowdose_ssim_scores.extend(ssim_metric(noisy, clean))

# Compute averages and standard deviations
lowdose_avg_psnr = np.mean(lowdose_psnr_scores)
lowdose_std_psnr = np.std(lowdose_psnr_scores)
lowdose_avg_ssim = np.mean(lowdose_ssim_scores)
lowdose_std_ssim = np.std(lowdose_ssim_scores)

# Print results
print(f"Low-Dose Images ->    SNR: {lowdose_avg_psnr:.4f} ± {lowdose_std_psnr:.4f},     SSIM: {lowdose_avg_ssim:.4f} ± {lowdose_std_ssim:.4f}")

"""#### Filtered Back Projection"""

fbp_psnr_scores = []
fbp_ssim_scores = []

# Define angles for Radon transform (adjust as needed based on your data)
theta = np.linspace(0., 180., 180, endpoint=False)

with torch.no_grad():
    for noisy, clean in test_loader:
        # Convert noisy batch to numpy for scikit-image (shape: [batch_size, 1, H, W] -> [batch_size, H, W])
        noisy_np = noisy.cpu().numpy().squeeze(1)  # Remove channel dim
        clean_np = clean.cpu().numpy()  # Keep as tensor for metrics later

        # Process each image in the batch
        fbp_reconstructions = []
        for i in range(noisy_np.shape[0]):
            # Forward projection (sinogram)
            sinogram = radon(noisy_np[i], theta=theta, circle=False)
            # FBP reconstruction
            reconstruction = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
            fbp_reconstructions.append(reconstruction)

        # Stack reconstructions and convert back to tensor
        fbp_reconstructions = np.stack(fbp_reconstructions, axis=0)  # [batch_size, H, W]
        fbp_reconstructions = torch.from_numpy(fbp_reconstructions).unsqueeze(1)  # [batch_size, 1, H, W]

        # Ensure same size as clean images (if Radon/iradon changes dimensions)
        if fbp_reconstructions.shape[-2:] != clean.shape[-2:]:
            fbp_reconstructions = F.interpolate(fbp_reconstructions, size=clean.shape[-2:], mode='bilinear', align_corners=False)

        # Compute metrics
        fbp_psnr_scores.extend(psnr_metric(fbp_reconstructions, clean))
        fbp_ssim_scores.extend(ssim_metric(fbp_reconstructions, clean))

# Compute averages and standard deviations for FBP
fbp_avg_psnr = np.mean(fbp_psnr_scores)
fbp_std_psnr = np.std(fbp_psnr_scores)
fbp_avg_ssim = np.mean(fbp_ssim_scores)
fbp_std_ssim = np.std(fbp_ssim_scores)

# Print results
print(f"FBP Reconstructed ->    PSNR: {fbp_avg_psnr:.4f} ± {fbp_std_psnr:.4f},     SSIM: {fbp_avg_ssim:.4f} ± {fbp_std_ssim:.4f}")

"""#### BM3D"""

# BM3D Denoising and Metrics
bm3d_psnr_scores = []
bm3d_ssim_scores = []

# Noise standard deviation estimate (adjust this based on your data)
sigma_psd = 0.1

with torch.no_grad():
    for noisy, clean in test_loader:
        # Convert noisy batch to numpy (shape: [batch_size, 1, H, W] -> [batch_size, H, W])
        noisy_np = noisy.cpu().numpy().squeeze(1)  # Remove channel dim

        # Process each image in the batch with BM3D
        bm3d_denoised = []
        for i in range(noisy_np.shape[0]):
            # BM3D expects image in [0, 1] range, which your data already is from transforms.ToTensor()
            denoised = bm3d.bm3d(noisy_np[i], sigma_psd=sigma_psd)
            bm3d_denoised.append(denoised)

        # Stack and convert back to tensor
        bm3d_denoised = np.stack(bm3d_denoised, axis=0)  # [batch_size, H, W]
        bm3d_denoised = torch.from_numpy(bm3d_denoised).unsqueeze(1)  # [batch_size, 1, H, W]

        # Compute metrics
        bm3d_psnr_scores.extend(psnr_metric(bm3d_denoised, clean))
        bm3d_ssim_scores.extend(ssim_metric(bm3d_denoised, clean))

# Compute averages and standard deviations for BM3D
bm3d_avg_psnr = np.mean(bm3d_psnr_scores)
bm3d_std_psnr = np.std(bm3d_psnr_scores)
bm3d_avg_ssim = np.mean(bm3d_ssim_scores)
bm3d_std_ssim = np.std(bm3d_ssim_scores)

# Print results
print(f"BM3D Denoised ->    PSNR: {bm3d_avg_psnr:.4f} ± {bm3d_std_psnr:.4f},     SSIM: {bm3d_avg_ssim:.4f} ± {bm3d_std_ssim:.4f}")

"""#### Final Results"""

# Function to print the metrics cleanly
def print_metrics_table():
    # List of methods and their corresponding metrics
    methods = [
        ("Low-Dose Images", lowdose_avg_psnr, lowdose_std_psnr, lowdose_avg_ssim, lowdose_std_ssim),
        ("FBP Reconstructed", fbp_avg_psnr, fbp_std_psnr, fbp_avg_ssim, fbp_std_ssim),
        ("BM3D Denoised", bm3d_avg_psnr, bm3d_std_psnr, bm3d_avg_ssim, bm3d_std_ssim),
    ]

    # Print table header
    print("\n Metrics Comparison")
    print("-" * 80)
    print(f"{'Method':<25} {'PSNR (Mean ± Std)':<25} {'SSIM (Mean ± Std)':<25}")
    print("-" * 80)

    # Print each row
    for method, psnr_mean, psnr_std, ssim_mean, ssim_std in methods:
        psnr_str = f"{psnr_mean:.4f} ± {psnr_std:.4f}"
        ssim_str = f"{ssim_mean:.4f} ± {ssim_std:.4f}"
        print(f"{method:<25} {psnr_str:<25} {ssim_str:<25}")

    print("-" * 80)

# Call the function to display the table
print_metrics_table()

def show_reconstruction_comparison(dataset, index, fbp_theta, sigma_psd=0.1):
    """
    Plots noisy, reconstructed (from each method), and full-dose images for a given dataset index,
    with PSNR and SSIM metrics displayed in the titles.

    Parameters:
    - dataset: DenoisingDataset instance
    - index: Index of the image to display
    - fbp_theta: Angles for Radon transform (used in FBP)
    - sigma_psd: Noise standard deviation for BM3D
    """
    # Get the noisy and clean images from the dataset
    noisy, clean = dataset[index]
    noisy = noisy.unsqueeze(0)  # Add batch dimension: [1, 1, H, W]
    clean = clean.unsqueeze(0)  # Add batch dimension: [1, 1, H, W]

    # Convert to numpy for processing and plotting
    noisy_np = noisy.squeeze(0).cpu().numpy().squeeze(0)  # [H, W]
    clean_np = clean.squeeze(0).cpu().numpy().squeeze(0)  # [H, W]

    # 1. FBP Reconstruction
    sinogram = radon(noisy_np, theta=fbp_theta, circle=False)
    fbp_reconstruction = iradon(sinogram, theta=fbp_theta, filter_name='ramp', circle=False)

    # 2. BM3D Denoising
    bm3d_denoised = bm3d.bm3d(noisy_np, sigma_psd=sigma_psd)

    # Compute PSNR and SSIM for each method
    data_range = clean_np.max() - clean_np.min()

    psnr_noisy = peak_signal_noise_ratio(clean_np, noisy_np, data_range=data_range)
    ssim_noisy = structural_similarity(clean_np, noisy_np, data_range=data_range)

    psnr_fbp = peak_signal_noise_ratio(clean_np, fbp_reconstruction, data_range=data_range)
    ssim_fbp = structural_similarity(clean_np, fbp_reconstruction, data_range=data_range)

    psnr_bm3d = peak_signal_noise_ratio(clean_np, bm3d_denoised, data_range=data_range)
    ssim_bm3d = structural_similarity(clean_np, bm3d_denoised, data_range=data_range)

    # Plotting with PSNR and SSIM in titles
    methods = [
        (f"Noisy Image\nPSNR: {psnr_noisy:.2f}, SSIM: {ssim_noisy:.3f}", noisy_np),
        (f"FBP\nPSNR: {psnr_fbp:.2f}, SSIM: {ssim_fbp:.3f}", fbp_reconstruction),
        (f"BM3D\nPSNR: {psnr_bm3d:.2f}, SSIM: {ssim_bm3d:.3f}", bm3d_denoised),
        (f"Full-Dose (Ground Truth)", clean_np)
    ]

    fig, axes = plt.subplots(1, len(methods), figsize=(3 * len(methods), 3))
    for ax, (title, img) in zip(axes, methods):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Define theta
theta = np.linspace(0., 180., 180, endpoint=False)

# Assuming your models and dataset are already defined and trained
show_reconstruction_comparison(
    dataset=dataset,
    index=1,  # Change this to show a different image
    fbp_theta=theta,
    sigma_psd=0.1,  # Adjust based on your noise level
)