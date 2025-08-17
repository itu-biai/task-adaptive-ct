import nibabel as nib
import numpy as np
import os
from skimage.transform import radon, iradon, resize
from skimage.util import img_as_float

"""#### CONFIG"""

CONFIG = {
    # Dataset parameters
    'target_size': (512, 512),  # Target size for resizing
    
    # Low-dose simulation parameters
    'dose_fraction': 0.75,  # Fraction of original dose
    'theta': None,  # Projection angles, None for auto
    
    # Directory paths
<<<<<<< HEAD
    'volume_folder': ".../nifti-data/Volumes",
    'segmentation_folder': ".../nifti-data/Segmentations",
    'output_dir': ".../preprocessed_data",
=======
    'volume_folder': "...",
    'segmentation_folder': "...",
    'output_dir': "...",
>>>>>>> d6247a0d702814980d577acbf34af3e9cee52ee2
    
    # Output subdirectories
    'low_dose_subdir': "low_dose",
    'full_dose_subdir': "full_dose",
    'seg_slice_subdir': "segmentation_slices"
}

# Create output directories
os.makedirs(CONFIG['output_dir'], exist_ok=True)
full_dose_dir = os.path.join(CONFIG['output_dir'], CONFIG['full_dose_subdir'])
low_dose_dir = os.path.join(CONFIG['output_dir'], CONFIG['low_dose_subdir'])
seg_slice_dir = os.path.join(CONFIG['output_dir'], CONFIG['seg_slice_subdir'])
os.makedirs(full_dose_dir, exist_ok=True)
os.makedirs(low_dose_dir, exist_ok=True)
os.makedirs(seg_slice_dir, exist_ok=True)

def normalize_to_01(img):
    """Normalize image to [0, 1] range"""
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max != img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return img

# Function to get a slice based on index and axis
def get_slice(data, slice_idx, axis):
    if axis == 0:
        return data[slice_idx, :, :]
    elif axis == 1:
        return data[:, slice_idx, :]
    else:  # axis=2
        return data[:, :, slice_idx]

# Function to simulate low-dose CT by adding Poisson noise in Radon (sinogram) domain
def simulate_low_dose_ct(image, dose_fraction=CONFIG['dose_fraction'], theta=CONFIG['theta']):
    """
    Simulates low-dose CT by adding Poisson noise in Radon (sinogram) domain.
    
    Parameters:
    - image: 2D numpy array (grayscale image)
    - dose_fraction: float, fraction of original dose (e.g., 0.25 for 25%)
    - theta: projection angles in degrees (default: 180 angles)
    
    Returns:
    - reconstructed noisy image
    """
    # Normalize image to [0, 1] range for Radon transform
    image = normalize_to_01(image)
    image = img_as_float(image)

    # Ensure image is square (required for radon/iradon)
    size = max(image.shape)
    image = resize(image, (size, size), anti_aliasing=True, preserve_range=True)

    # Projection angles
    if theta is None:
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    # Apply Radon Transform to get the sinogram (projection data)
    sinogram = radon(image, theta=theta, circle=True)

    # Simulate X-ray counts and add Poisson noise
    sinogram_max = sinogram.max()
    if sinogram_max == 0:  # Avoid division by zero
        counts = sinogram
    else:
        counts = sinogram / sinogram_max * 1e4  # Scale to simulate detector counts
    noisy_counts = np.random.poisson(counts * dose_fraction)
    noisy_sinogram = noisy_counts / (1e4 * dose_fraction) * sinogram_max

    # Reconstruct image using Inverse Radon Transform
    reconstruction = iradon(noisy_sinogram, theta=theta, circle=True)

    # Resize back to target size
    reconstruction = resize(reconstruction, CONFIG['target_size'], anti_aliasing=True, preserve_range=True)
    reconstruction = reconstruction.astype(np.float32)

    return reconstruction

# Function to match volume and segmentation files (assuming similar naming)
def get_matched_files(vol_folder, seg_folder):
    vol_files = sorted([f for f in os.listdir(vol_folder) if f.endswith('.nii')])
    seg_files = sorted([f for f in os.listdir(seg_folder) if f.endswith('.nii')])
    
    matched_pairs = []
    for vol_file in vol_files:
        identifier = vol_file.replace("volume-", "").replace(".nii", "")
        expected_seg_file = f"segmentation-{identifier}.nii"
        if expected_seg_file in seg_files:
            matched_pairs.append((vol_file, expected_seg_file))
        else:
            print(f"Warning: No matching segmentation file found for {vol_file}")
    
    return matched_pairs

# Get matched file pairs
matched_pairs = get_matched_files(CONFIG['volume_folder'], CONFIG['segmentation_folder'])

# Initialize counters for summary
total_slices_saved = 0
slices_per_volume = {}

# Process each pair of files
for vol_file, seg_file in matched_pairs:
    print(f"Processing {vol_file} and {seg_file}...")
    slices_per_volume[vol_file] = 0
    
    # Load the .nii files
    volume_path = os.path.join(CONFIG['volume_folder'], vol_file)
    segmentation_path = os.path.join(CONFIG['segmentation_folder'], seg_file)
    volume_img = nib.load(volume_path)
    segmentation_img = nib.load(segmentation_path)
    
    # Get the data as numpy arrays
    volume_data = volume_img.get_fdata()
    segmentation_data = segmentation_img.get_fdata()
    
    # Ensure the shapes match
    assert volume_data.shape == segmentation_data.shape, f"Shapes do not match for {vol_file} and {seg_file}!"
    
    # Number of slices along the chosen axis
    num_slices = volume_data.shape[2]  # Using axial slices (axis=2)
    
    # Extract, resize, simulate low-dose, and save slices only if segmentation has non-zero values
    for slice_idx in range(num_slices):
        vol_slice = get_slice(volume_data, slice_idx, 2)  # Using axial slices (axis=2)
        seg_slice = get_slice(segmentation_data, slice_idx, 2)  # Using axial slices (axis=2)
        
        # Define identifier outside the conditional block
        identifier = vol_file.replace("volume-", "").replace(".nii", "")
        
        # Check if segmentation slice contains any non-zero values
        if np.any(seg_slice != 0):
            # Resize the slices to target size
            vol_slice_resized = resize(vol_slice, CONFIG['target_size'], anti_aliasing=True, preserve_range=True)
            seg_slice_resized = resize(seg_slice, CONFIG['target_size'], anti_aliasing=False, preserve_range=True, order=0)  # Nearest-neighbor for segmentation
            
            # Normalize full-dose image to [0, 1]
            vol_slice_resized = normalize_to_01(vol_slice_resized)
            
            # Simulate low-dose from the resized full-dose slice
            low_dose_slice = simulate_low_dose_ct(vol_slice_resized)
            
            # Define save paths
            full_dose_save_path = os.path.join(full_dose_dir, f"full_dose_{identifier}_slice_{slice_idx:03d}.npy")
            low_dose_save_path = os.path.join(low_dose_dir, f"low_dose_{identifier}_slice_{slice_idx:03d}.npy")
            seg_save_path = os.path.join(seg_slice_dir, f"segmentation_{identifier}_slice_{slice_idx:03d}.npy")
            
            # Save as NumPy arrays
            full_dose_final = vol_slice_resized.astype(np.float32)
            low_dose_final = low_dose_slice.astype(np.float32)
            seg_final = seg_slice_resized.astype(np.uint8)
            
            np.save(full_dose_save_path, full_dose_final)  # normalized [0, 1]
            np.save(low_dose_save_path, low_dose_final)    # normalized [0, 1]
            np.save(seg_save_path, seg_final)
            
            # Update counters
            total_slices_saved += 1
            slices_per_volume[vol_file] += 1
            
            print(f"  Saved slice {slice_idx}/{num_slices-1} for {identifier}")
        else:
            print(f"  Skipped slice {slice_idx}/{num_slices-1} for {identifier} (all zeros in segmentation)")

<<<<<<< HEAD
# Create summary file
summary_path = os.path.join(CONFIG['output_dir'], 'dataset_summary.txt')
with open(summary_path, 'w') as f:
    f.write(f"Dataset Summary\n")
    f.write(f"==============\n\n")
    f.write(f"Total slices saved: {total_slices_saved}\n\n")
    f.write(f"Slices per volume:\n")
    f.write(f"----------------\n")
    for vol_file, count in slices_per_volume.items():
        f.write(f"{vol_file}: {count} slices\n")

print(f"All relevant slices saved successfully in {CONFIG['output_dir']} with size {CONFIG['target_size']}, including normalized low-dose, full-dose and segmentation versions!")
print(f"Summary saved to {summary_path}")
=======
print(f"All relevant slices saved successfully in {CONFIG['output_dir']} with size {CONFIG['target_size']}, including normalized low-dose, full-dose and segmentation versions!")
>>>>>>> d6247a0d702814980d577acbf34af3e9cee52ee2
