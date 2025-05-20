import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

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
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Extract noisy and clean images from the processed test_dataset
noisy_processed = []
clean_processed = []

# Iterate through the test_dataset to get processed data
for noisy_img, clean_img in dataset:
    noisy_processed.append(noisy_img.numpy().squeeze())  # Shape: (H, W)
    clean_processed.append(clean_img.numpy().squeeze())  # Shape: (H, W)

# Convert lists to numpy arrays and flatten for histogram
noisy_processed = np.stack(noisy_processed, axis=0).flatten()  # Shape: (N*H*W,)
clean_processed = np.stack(clean_processed, axis=0).flatten()  # Shape: (N*H*W,)

# Check the range of values
print(f"Noisy images - Min: {noisy_processed.min():.4f}, Max: {noisy_processed.max():.4f}")
print(f"Clean images - Min: {clean_processed.min():.4f}, Max: {clean_processed.max():.4f}")

# Plot histograms of the processed data in [0, 1] range
plt.figure(figsize=(10, 6))
plt.hist(noisy_processed, bins=100, density=True, alpha=0.5, color='blue', label='Processed Noisy Images')
plt.hist(clean_processed, bins=100, density=True, alpha=0.5, color='green', label='Processed Clean Images')
plt.title('Histogram of Processed Noisy and Clean Image Pixel Intensities ')
plt.xlabel('Pixel Intensity (in [0, 1])')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Function to visualize noisy and clean image pairs
def visualize_test_samples(test_loader, num_samples=4):
    noisy_images, clean_images = next(iter(test_loader))
    num_samples = min(num_samples, noisy_images.size(0))
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        noisy_img = noisy_images[i].squeeze(0).cpu().numpy()
        clean_img = clean_images[i].squeeze(0).cpu().numpy()

        if num_samples == 1:
            axes[0].imshow(noisy_img, cmap='gray')
            axes[0].set_title('Noisy Image')
            axes[0].axis('off')
            axes[1].imshow(clean_img, cmap='gray')
            axes[1].set_title('Clean Image')
            axes[1].axis('off')
        else:
            axes[0, i].imshow(noisy_img, cmap='gray')
            axes[0, i].set_title('Noisy Image')
            axes[0, i].axis('off')
            axes[1, i].imshow(clean_img, cmap='gray')
            axes[1, i].set_title('Clean Image')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

visualize_test_samples(test_loader, num_samples=5)

"""#### Circular Mask"""

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(w, h) / 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.float32)  # Shape: (H, W)

def visualize_with_mask(test_loader, mask, num_samples=2):
    noisy_images, clean_images = next(iter(test_loader))
    num_samples = min(num_samples, noisy_images.size(0))
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        noisy_img = noisy_images[i].squeeze(0).cpu().numpy()
        clean_img = clean_images[i].squeeze(0).cpu().numpy()
        mask_rgba = np.zeros((512, 512, 4))
        mask_rgba[..., 0] = mask
        mask_rgba[..., 3] = mask * 0.3

        if num_samples == 1:
            axes[0].imshow(noisy_img, cmap='gray')
            axes[0].imshow(mask_rgba)
            axes[0].set_title('Noisy Image with Mask')
            axes[0].axis('off')
            axes[1].imshow(clean_img, cmap='gray')
            axes[1].imshow(mask_rgba)
            axes[1].set_title('Clean Image with Mask')
            axes[1].axis('off')
        else:
            axes[0, i].imshow(noisy_img, cmap='gray')
            axes[0, i].imshow(mask_rgba)
            axes[0, i].set_title('Noisy Image with Mask')
            axes[0, i].axis('off')
            axes[1, i].imshow(clean_img, cmap='gray')
            axes[1, i].imshow(mask_rgba)
            axes[1, i].set_title('Clean Image with Mask')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)
visualize_with_mask(test_loader, mask, num_samples=5)

"""#### PSNR & SSIM Metrics"""

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

"""#### Low-Dose Images"""

def masked_psnr_metric(noisy, clean, mask):
    mask_tensor = torch.from_numpy(mask).to(noisy.device).view(1, 1, 512, 512)
    masked_noisy = noisy * mask_tensor
    masked_clean = clean * mask_tensor
    return psnr_metric(masked_noisy, masked_clean)

def masked_ssim_metric(noisy, clean, mask):
    mask_tensor = torch.from_numpy(mask).to(noisy.device).view(1, 1, 512, 512)
    masked_noisy = noisy * mask_tensor
    masked_clean = clean * mask_tensor
    return ssim_metric(masked_noisy, masked_clean)

lowdose_psnr_scores = []
lowdose_ssim_scores = []

with torch.no_grad():
    for noisy, clean in test_loader:
        lowdose_psnr_scores.extend(masked_psnr_metric(noisy, clean, mask))
        lowdose_ssim_scores.extend(masked_ssim_metric(noisy, clean, mask))

lowdose_avg_psnr = np.mean(lowdose_psnr_scores)
lowdose_std_psnr = np.std(lowdose_psnr_scores)
lowdose_avg_ssim = np.mean(lowdose_ssim_scores)
lowdose_std_ssim = np.std(lowdose_ssim_scores)

print(f"Low-Dose Images -> PSNR: {lowdose_avg_psnr:.4f} ± {lowdose_std_psnr:.4f}, SSIM: {lowdose_avg_ssim:.4f} ± {lowdose_std_ssim:.4f}")

"""#### RED-CNN Model"""

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=2)  # Padding=2 to maintain size
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        
        # Decoder
        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=2)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        residual_1 = x  # Shape: (batch, 1, 512, 512)
        out = self.relu(self.conv1(x))  # Shape: (batch, out_ch, 512, 512)
        out = self.relu(self.conv2(out))  # Shape: (batch, out_ch, 512, 512)
        residual_2 = out
        out = self.relu(self.conv3(out))  # Shape: (batch, out_ch, 512, 512)
        out = self.relu(self.conv4(out))  # Shape: (batch, out_ch, 512, 512)
        residual_3 = out
        out = self.relu(self.conv5(out))  # Shape: (batch, out_ch, 512, 512)
        
        # Decoder
        out = self.tconv1(out)  # Shape: (batch, out_ch, 512, 512)
        out = out + residual_3  # Skip connection
        out = self.relu(out)
        out = self.tconv2(out)  # Shape: (batch, out_ch, 512, 512)
        out = self.tconv3(out)  # Shape: (batch, out_ch, 512, 512)
        out = out + residual_2  # Skip connection
        out = self.relu(out)
        out = self.tconv4(out)  # Shape: (batch, out_ch, 512, 512)
        out = self.tconv5(out)  # Shape: (batch, 1, 512, 512)
        out = out + residual_1  # Skip connection
        out = torch.sigmoid(out)  # Ensure output in [0,1]
        return out

# ----- Model, optimizer, loss -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RED_CNN(out_ch=96).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Convert mask to tensor for training
mask_tensor = torch.from_numpy(mask).to(device).view(1, 1, 512, 512)

# ----- Training loop -----
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        masked_output = output * mask_tensor
        masked_clean = clean * mask_tensor
        loss = criterion(masked_output, masked_clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Evaluation on test set using masked metrics
    model.eval()
    psnr_scores = []
    ssim_scores = []
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            psnr_scores.extend(masked_psnr_metric(output, clean, mask))
            ssim_scores.extend(masked_ssim_metric(output, clean, mask))
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, "
          f"RED-CNN Reconstructed -> Masked PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}, "
          f"Masked SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

redcnn = model
torch.save(redcnn.state_dict(), 'redcnn_LD_reconstruction_state.pth')

"""#### Final Results"""

def print_metrics_table():
    methods = [
        ("Low-Dose Images", lowdose_avg_psnr, lowdose_std_psnr, lowdose_avg_ssim, lowdose_std_ssim),
        ("RED-CNN Reconstructed", avg_psnr, std_psnr, avg_ssim, std_ssim),
    ]
    print("\n Metrics Comparison")
    print("-" * 80)
    print(f"{'Method':<25} {'PSNR (Mean ± Std)':<25} {'SSIM (Mean ± Std)':<25}")
    print("-" * 80)
    for method, psnr_mean, psnr_std, ssim_mean, ssim_std in methods:
        psnr_str = f"{psnr_mean:.4f} ± {psnr_std:.4f}"
        ssim_str = f"{ssim_mean:.4f} ± {ssim_std:.4f}"
        print(f"{method:<25} {psnr_str:<25} {ssim_str:<25}")
    print("-" * 80)

print_metrics_table()

def show_reconstruction_comparison(dataset, index, model_redcnn, device='cuda'):
    mask = create_circular_mask(h=512, w=512, center=(256, 256), radius=256)
    noisy, clean = dataset[index]
    noisy = noisy.unsqueeze(0)
    clean = clean.unsqueeze(0)
    noisy = noisy.to(device)
    clean = clean.to(device)

    model_redcnn.eval()
    with torch.no_grad():
        redcnn_output = model_redcnn(noisy)

    psnr_noisy = masked_psnr_metric(noisy, clean, mask)[0]
    ssim_noisy = masked_ssim_metric(noisy, clean, mask)[0]
    psnr_redcnn = masked_psnr_metric(redcnn_output, clean, mask)[0]
    ssim_redcnn = masked_ssim_metric(redcnn_output, clean, mask)[0]

    mask_np = mask
    noisy_np = noisy.squeeze(0).cpu().numpy().squeeze(0) * mask_np
    clean_np = clean.squeeze(0).cpu().numpy().squeeze(0) * mask_np
    redcnn_np = redcnn_output.squeeze(0).cpu().numpy().squeeze(0) * mask_np

    methods = [
        (f"Noisy Image\nMasked PSNR: {psnr_noisy:.2f}, Masked SSIM: {ssim_noisy:.3f}", noisy_np),
        (f"RED-CNN\nMasked PSNR: {psnr_redcnn:.2f}, Masked SSIM: {ssim_redcnn:.3f}", redcnn_np),
        (f"Full-Dose (Ground Truth)", clean_np)
    ]

    fig, axes = plt.subplots(1, len(methods), figsize=(3 * len(methods), 3))
    for ax, (title, img) in zip(axes, methods):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
show_reconstruction_comparison(
    dataset=test_dataset,
    index=160,
    model_redcnn=redcnn,
    device=device
)