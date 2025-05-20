import numpy as np
import matplotlib.pyplot as plt

# Paths to your .npy files (adjust paths based on your previous code's output structure)
npy_file_path_full_dose = "..."
npy_file_path_low_dose = "..."
npy_file_path_segment = "..."

# Load the NumPy files
full_dose_data = np.load(npy_file_path_full_dose)
low_dose_data = np.load(npy_file_path_low_dose)
segment_data = np.load(npy_file_path_segment)

# Check the shapes of the data
print(f"Full-dose data shape: {full_dose_data.shape}")
print(f"Low-dose data shape: {low_dose_data.shape}")
print(f"Segmentation data shape: {segment_data.shape}")
image_size = full_dose_data.shape  # Assuming all have the same size

# Create a figure with three subplots side by side
plt.figure(figsize=(15, 5))  # Adjusted figure size for three images

# Subplot 1: Full-dose image
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st position
plt.imshow(full_dose_data, cmap="gray")
plt.title("Full-Dose Image")
plt.axis("off")  # Hide axes

# Subplot 2: Low-dose image
plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd position
plt.imshow(low_dose_data, cmap="gray")
plt.title("Low-Dose Image (Quarter-Dose)")
plt.axis("off")  # Hide axes

# Subplot 3: Segmentation map
plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd position
plt.imshow(segment_data, cmap="gray")
plt.title("Segmentation Map")
plt.axis("off")  # Hide axes

# Add a super title for the entire figure
plt.suptitle(f"Comparison of Full-Dose, Low-Dose, and Segmentation\nSize: {image_size[0]} x {image_size[1]} pixels", fontsize=14)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle

# Show the plot
plt.show()