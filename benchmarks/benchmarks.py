import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import radon, iradon
import bm3d
import torch
from torch.utils.data import Dataset, random_split

"""#### CONFIG"""

CONFIG = {
    # Benchmark parameters
    'batch_size': 1,
    'device': 'cuda',  # or 'cpu'
    'train_test_split': 0.8,  # 80% training, 20% testing
    
    # BM3D parameters
    'sigma_psd': 0.075,
    
    # FBP parameters
    'theta': None,  # None for auto-calculation
    
    # Directory paths
    'low_dose_dir': "...",
    'full_dose_dir': "...",
    'seg_dir': "..."
}

# Function to preload .npy files from a folder
def load_npy_from_folder(folder_path):
    images = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        img = np.load(file_path)  # Shape: (H, W), already normalized to [0, 1] during dataset creation
        images.append(img)
    return np.stack(images, axis=0)  # Shape: (N, H, W)

# Simplified Dataset for loading low-dose, full-dose, and segmentation maps
class TaskAdaptiveUnifiedDataset(Dataset):
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

        return low_dose_img, full_dose_img, seg_map

# Create dataset and split into train/test
dataset = TaskAdaptiveUnifiedDataset(
    CONFIG['low_dose_dir'],
    CONFIG['full_dose_dir'],
    CONFIG['seg_dir']
)

# Set random seed for reproducibility
torch.manual_seed(1773)
generator = torch.Generator().manual_seed(1773)

# Split dataset
train_size = int(CONFIG['train_test_split'] * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

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

# PSNR and SSIM metrics for NumPy arrays
def psnr_metric(pred, target):
    psnr_values = []
    for i in range(pred.shape[0]):
        p = peak_signal_noise_ratio(target[i], pred[i], 
                                  data_range=target[i].max() - target[i].min())
        psnr_values.append(p)
    return psnr_values

def ssim_metric(pred, target):
    ssim_values = []
    for i in range(pred.shape[0]):
        s = structural_similarity(target[i], pred[i], 
                                data_range=target[i].max() - target[i].min())
        ssim_values.append(s)
    return ssim_values

def masked_psnr_metric(pred, target, mask):
    masked_pred = pred * mask[np.newaxis, ...]
    masked_target = target * mask[np.newaxis, ...]
    return psnr_metric(masked_pred, masked_target)

def masked_ssim_metric(pred, target, mask):
    masked_pred = pred * mask[np.newaxis, ...]
    masked_target = target * mask[np.newaxis, ...]
    return ssim_metric(masked_pred, masked_target)

# Calculate metrics for low-dose, FBP, and BM3D images (regular and masked)
def evaluate_metrics(dataset, theta, sigma_psd, image_size=(512, 512)):
    # Initialize lists for metrics
    lowdose_psnr_scores = []
    lowdose_ssim_scores = []
    lowdose_masked_psnr_scores = []
    lowdose_masked_ssim_scores = []
    
    fbp_psnr_scores = []
    fbp_ssim_scores = []
    fbp_masked_psnr_scores = []
    fbp_masked_ssim_scores = []
    
    bm3d_psnr_scores = []
    bm3d_ssim_scores = []
    bm3d_masked_psnr_scores = []
    bm3d_masked_ssim_scores = []

    # Create circular mask
    mask = create_circular_mask(image_size[0], image_size[1])

    # Process each sample in the dataset
    for i in range(len(dataset)):
        noisy, clean, _ = dataset[i]  # low_dose_img, full_dose_img, seg_map
        noisy = noisy[np.newaxis, ...]  # Shape: (1, H, W)
        clean = clean[np.newaxis, ...]  # Shape: (1, H, W)

        # --- Low-Dose Metrics ---
        lowdose_psnr_scores.extend(psnr_metric(noisy, clean))
        lowdose_ssim_scores.extend(ssim_metric(noisy, clean))
        lowdose_masked_psnr_scores.extend(masked_psnr_metric(noisy, clean, mask))
        lowdose_masked_ssim_scores.extend(masked_ssim_metric(noisy, clean, mask))

        # --- FBP Reconstruction ---
        noisy_np = noisy  # Shape: (1, H, W)
        fbp_reconstructions = []
        for j in range(noisy_np.shape[0]):
            sinogram = radon(noisy_np[j], theta=theta, circle=False)
            reconstruction = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
            fbp_reconstructions.append(reconstruction)

        fbp_reconstructions = np.stack(fbp_reconstructions, axis=0)  # Shape: (1, H, W)

        # Resize if necessary (using simple interpolation)
        if fbp_reconstructions.shape[-2:] != clean.shape[-2:]:
            from scipy.ndimage import zoom
            scale_h = clean.shape[-2] / fbp_reconstructions.shape[-2]
            scale_w = clean.shape[-1] / fbp_reconstructions.shape[-1]
            fbp_reconstructions = zoom(fbp_reconstructions, (1, scale_h, scale_w), order=1)

        fbp_psnr_scores.extend(psnr_metric(fbp_reconstructions, clean))
        fbp_ssim_scores.extend(ssim_metric(fbp_reconstructions, clean))
        fbp_masked_psnr_scores.extend(masked_psnr_metric(fbp_reconstructions, clean, mask))
        fbp_masked_ssim_scores.extend(masked_ssim_metric(fbp_reconstructions, clean, mask))

        # --- BM3D Denoising ---
        bm3d_denoised = []
        for j in range(noisy_np.shape[0]):
            denoised = bm3d.bm3d(noisy_np[j], sigma_psd=CONFIG['sigma_psd'])
            bm3d_denoised.append(denoised)

        bm3d_denoised = np.stack(bm3d_denoised, axis=0)  # Shape: (1, H, W)

        bm3d_psnr_scores.extend(psnr_metric(bm3d_denoised, clean))
        bm3d_ssim_scores.extend(ssim_metric(bm3d_denoised, clean))
        bm3d_masked_psnr_scores.extend(masked_psnr_metric(bm3d_denoised, clean, mask))
        bm3d_masked_ssim_scores.extend(masked_ssim_metric(bm3d_denoised, clean, mask))

    # Compute averages and standard deviations
    metrics = {
        'lowdose': {
            'psnr': (np.mean(lowdose_psnr_scores), np.std(lowdose_psnr_scores)),
            'ssim': (np.mean(lowdose_ssim_scores), np.std(lowdose_ssim_scores)),
            'masked_psnr': (np.mean(lowdose_masked_psnr_scores), np.std(lowdose_masked_psnr_scores)),
            'masked_ssim': (np.mean(lowdose_masked_ssim_scores), np.std(lowdose_masked_ssim_scores))
        },
        'fbp': {
            'psnr': (np.mean(fbp_psnr_scores), np.std(fbp_psnr_scores)),
            'ssim': (np.mean(fbp_ssim_scores), np.std(fbp_ssim_scores)),
            'masked_psnr': (np.mean(fbp_masked_psnr_scores), np.std(fbp_masked_psnr_scores)),
            'masked_ssim': (np.mean(fbp_masked_ssim_scores), np.std(fbp_masked_ssim_scores))
        },
        'bm3d': {
            'psnr': (np.mean(bm3d_psnr_scores), np.std(bm3d_psnr_scores)),
            'ssim': (np.mean(bm3d_ssim_scores), np.std(bm3d_ssim_scores)),
            'masked_psnr': (np.mean(bm3d_masked_psnr_scores), np.std(bm3d_masked_psnr_scores)),
            'masked_ssim': (np.mean(bm3d_masked_ssim_scores), np.std(bm3d_masked_ssim_scores))
        }
    }

    # Print results
    print("Low-Dose Images:")
    print(f"  PSNR: {metrics['lowdose']['psnr'][0]:.4f} ± {metrics['lowdose']['psnr'][1]:.4f}")
    print(f"  SSIM: {metrics['lowdose']['ssim'][0]:.4f} ± {metrics['lowdose']['ssim'][1]:.4f}")
    print(f"  Masked PSNR: {metrics['lowdose']['masked_psnr'][0]:.4f} ± {metrics['lowdose']['masked_psnr'][1]:.4f}")
    print(f"  Masked SSIM: {metrics['lowdose']['masked_ssim'][0]:.4f} ± {metrics['lowdose']['masked_ssim'][1]:.4f}")
    
    print("\nFBP Reconstructed:")
    print(f"  PSNR: {metrics['fbp']['psnr'][0]:.4f} ± {metrics['fbp']['psnr'][1]:.4f}")
    print(f"  SSIM: {metrics['fbp']['ssim'][0]:.4f} ± {metrics['fbp']['ssim'][1]:.4f}")
    print(f"  Masked PSNR: {metrics['fbp']['masked_psnr'][0]:.4f} ± {metrics['fbp']['masked_psnr'][1]:.4f}")
    print(f"  Masked SSIM: {metrics['fbp']['masked_ssim'][0]:.4f} ± {metrics['fbp']['masked_ssim'][1]:.4f}")
    
    print("\nBM3D Denoised:")
    print(f"  PSNR: {metrics['bm3d']['psnr'][0]:.4f} ± {metrics['bm3d']['psnr'][1]:.4f}")
    print(f"  SSIM: {metrics['bm3d']['ssim'][0]:.4f} ± {metrics['bm3d']['ssim'][1]:.4f}")
    print(f"  Masked PSNR: {metrics['bm3d']['masked_psnr'][0]:.4f} ± {metrics['bm3d']['masked_psnr'][1]:.4f}")
    print(f"  Masked SSIM: {metrics['bm3d']['masked_ssim'][0]:.4f} ± {metrics['bm3d']['masked_ssim'][1]:.4f}")

    return metrics

# Example usage
if __name__ == "__main__":
    # Initialize dataset (no need for normalization as data is already normalized)
    dataset = TaskAdaptiveUnifiedDataset(
        low_dose_dir=CONFIG['low_dose_dir'],
        full_dose_dir=CONFIG['full_dose_dir'],
        seg_dir=CONFIG['seg_dir'],
        transform_recon=None  # Only add transforms if needed for tensor conversion, etc.
    )

    # Evaluate metrics
    metrics = evaluate_metrics(test_dataset, CONFIG['theta'], CONFIG['sigma_psd'], image_size=(512, 512))