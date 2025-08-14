import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import radon, iradon
import bm3d
import torch
from torch.utils.data import Dataset, random_split
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import time

# Import the dataset and other utilities from benchmarks.py
from benchmarks import (
    CONFIG, TaskAdaptiveUnifiedDataset, create_circular_mask,
    psnr_metric, ssim_metric, masked_psnr_metric, masked_ssim_metric
)

def process_batch(batch_data, theta, sigma_psd, mask):
    """Process a single batch of images and compute metrics."""
    noisy, clean, _ = batch_data  # low_dose_img, full_dose_img, seg_map
    noisy = noisy[np.newaxis, ...]  # Shape: (1, H, W)
    clean = clean[np.newaxis, ...]  # Shape: (1, H, W)

    metrics = {
        'lowdose': {},
        'fbp': {},
        'bm3d': {}
    }

    # --- Low-Dose Metrics ---
    metrics['lowdose']['psnr'] = psnr_metric(noisy, clean)
    metrics['lowdose']['ssim'] = ssim_metric(noisy, clean)
    metrics['lowdose']['masked_psnr'] = masked_psnr_metric(noisy, clean, mask)
    metrics['lowdose']['masked_ssim'] = masked_ssim_metric(noisy, clean, mask)

    # --- FBP Reconstruction ---
    noisy_np = noisy  # Shape: (1, H, W)
    fbp_reconstructions = []
    for j in range(noisy_np.shape[0]):
        sinogram = radon(noisy_np[j], theta=theta, circle=False)
        reconstruction = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
        fbp_reconstructions.append(reconstruction)

    fbp_reconstructions = np.stack(fbp_reconstructions, axis=0)  # Shape: (1, H, W)

    # Resize if necessary
    if fbp_reconstructions.shape[-2:] != clean.shape[-2:]:
        from scipy.ndimage import zoom
        scale_h = clean.shape[-2] / fbp_reconstructions.shape[-2]
        scale_w = clean.shape[-1] / fbp_reconstructions.shape[-1]
        fbp_reconstructions = zoom(fbp_reconstructions, (1, scale_h, scale_w), order=1)

    metrics['fbp']['psnr'] = psnr_metric(fbp_reconstructions, clean)
    metrics['fbp']['ssim'] = ssim_metric(fbp_reconstructions, clean)
    metrics['fbp']['masked_psnr'] = masked_psnr_metric(fbp_reconstructions, clean, mask)
    metrics['fbp']['masked_ssim'] = masked_ssim_metric(fbp_reconstructions, clean, mask)

    # --- BM3D Denoising ---
    bm3d_denoised = []
    for j in range(noisy_np.shape[0]):
        denoised = bm3d.bm3d(noisy_np[j], sigma_psd=sigma_psd)
        bm3d_denoised.append(denoised)

    bm3d_denoised = np.stack(bm3d_denoised, axis=0)  # Shape: (1, H, W)

    metrics['bm3d']['psnr'] = psnr_metric(bm3d_denoised, clean)
    metrics['bm3d']['ssim'] = ssim_metric(bm3d_denoised, clean)
    metrics['bm3d']['masked_psnr'] = masked_psnr_metric(bm3d_denoised, clean, mask)
    metrics['bm3d']['masked_ssim'] = masked_ssim_metric(bm3d_denoised, clean, mask)

    return metrics

def evaluate_metrics_parallel(dataset, theta, sigma_psd, image_size=(512, 512), num_processes=None):
    """Evaluate metrics in parallel using multiple CPU cores."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Create circular mask
    mask = create_circular_mask(image_size[0], image_size[1])

    # Create a pool of workers
    pool = mp.Pool(processes=num_processes)

    # Prepare the partial function with fixed arguments
    process_func = partial(process_batch, theta=theta, sigma_psd=sigma_psd, mask=mask)

    # Process all samples in parallel with progress bar
    print(f"\nProcessing {len(dataset)} samples using {num_processes} CPU cores...")
    start_time = time.time()
    
    results = list(tqdm(
        pool.imap(process_func, [dataset[i] for i in range(len(dataset))]),
        total=len(dataset),
        desc="Computing metrics"
    ))

    # Close the pool
    pool.close()
    pool.join()

    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")

    # Aggregate results
    metrics = {
        'lowdose': {
            'psnr': [],
            'ssim': [],
            'masked_psnr': [],
            'masked_ssim': []
        },
        'fbp': {
            'psnr': [],
            'ssim': [],
            'masked_psnr': [],
            'masked_ssim': []
        },
        'bm3d': {
            'psnr': [],
            'ssim': [],
            'masked_psnr': [],
            'masked_ssim': []
        }
    }

    # Collect all results
    for result in results:
        for method in result:
            for metric_name in result[method]:
                metrics[method][metric_name].extend(result[method][metric_name])

    # Compute final statistics
    final_metrics = {}
    for method in metrics:
        final_metrics[method] = {}
        for metric_name in metrics[method]:
            values = metrics[method][metric_name]
            final_metrics[method][metric_name] = (np.mean(values), np.std(values))

    # Print results
    print("\nResults:")
    for method in ['Low-Dose Images', 'FBP Reconstructed', 'BM3D Denoised']:
        method_key = method.lower().replace(' images', '').replace(' reconstructed', '').replace(' denoised', '')
        print(f"\n{method}:")
        print(f"  PSNR: {final_metrics[method_key]['psnr'][0]:.4f} ± {final_metrics[method_key]['psnr'][1]:.4f}")
        print(f"  SSIM: {final_metrics[method_key]['ssim'][0]:.4f} ± {final_metrics[method_key]['ssim'][1]:.4f}")
        print(f"  Masked PSNR: {final_metrics[method_key]['masked_psnr'][0]:.4f} ± {final_metrics[method_key]['masked_psnr'][1]:.4f}")
        print(f"  Masked SSIM: {final_metrics[method_key]['masked_ssim'][0]:.4f} ± {final_metrics[method_key]['masked_ssim'][1]:.4f}")

    return final_metrics

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(1773)
    generator = torch.Generator().manual_seed(1773)

    # Initialize dataset
    dataset = TaskAdaptiveUnifiedDataset(
        low_dose_dir=CONFIG['low_dose_dir'],
        full_dose_dir=CONFIG['full_dose_dir'],
        seg_dir=CONFIG['seg_dir'],
        transform_recon=None
    )

    # Split dataset
    train_size = int(CONFIG['train_test_split'] * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    # Evaluate metrics in parallel
    metrics = evaluate_metrics_parallel(
        test_dataset,
        CONFIG['theta'],
        CONFIG['sigma_psd'],
        image_size=(512, 512)
    )
