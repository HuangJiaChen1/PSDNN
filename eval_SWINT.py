import glob
import os
import random
import numpy as np
from mat4py import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Basic transforms
to_tensor = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


# =============================================================================
# Dataset Class
# =============================================================================
# This dataset always returns the full image (448x672) for evaluation.
class ShanghaiTechADataset_ViT(Dataset):
    def __init__(self, root_dir, patch_size=8, mode="eval"):
        """
        For evaluation, each image is resized to 448×672.
        The ground truth counts grid is computed on the resized image using patch_size=8,
        which yields a grid of 56x84.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size  # 8
        self.mode = mode  # For test, mode must be "eval"
        self.samples = []
        self.full_size = (448, 672)  # (height, width)
        self._prepare_samples()

    def _prepare_samples(self):
        images_dir = os.path.join(self.root_dir, "images")
        gt_dir = os.path.join(self.root_dir, "ground_truth")
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        target_h, target_w = self.full_size  # 448, 672
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            gt_file = "GT_" + os.path.splitext(img_file)[0] + ".mat"
            gt_path = os.path.join(gt_dir, gt_file)
            gt_data = loadmat(gt_path)
            points = [(float(p[0]), float(p[1])) for p in gt_data['image_info']['location']]
            with Image.open(img_path) as img:
                orig_width, orig_height = img.size
            scale_x = target_w / orig_width
            scale_y = target_h / orig_height
            scaled_points = [(x * scale_x, y * scale_y) for (x, y) in points]
            n_patches_x = target_w // self.patch_size  # 672/8 = 84
            n_patches_y = target_h // self.patch_size  # 448/8 = 56
            counts = np.zeros((n_patches_y, n_patches_x), dtype=int)
            for (x, y) in scaled_points:
                px = min(int(x // self.patch_size), n_patches_x - 1)
                py = min(int(y // self.patch_size), n_patches_y - 1)
                counts[py, px] = min(counts[py, px] + 1, 4)
            self.samples.append({'img_path': img_path, 'counts': counts})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert("RGB")
        image = image.resize((self.full_size[1], self.full_size[0]), Image.BILINEAR)
        counts_full = sample['counts']  # shape: (56,84)
        n_patches_y, n_patches_x = counts_full.shape
        labels = np.zeros((n_patches_y * n_patches_x, 5), dtype=np.float32)
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                count = counts_full[i, j]
                labels[i * n_patches_x + j, count] = 1
        image = to_tensor(image)
        image = normalize(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


# =============================================================================
# Model Definition
# =============================================================================
class LearnableUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Upsample from 7x7 to 28x28 using ConvTranspose2d.
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=4, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CrowdCountingSwin(nn.Module):
    def __init__(self):
        super().__init__()
        # Model is trained on 224x224 patches.
        self.base_model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            img_size=(224, 224)
        )
        self.head = nn.Linear(self.base_model.num_features, 5)
        self.upsample = LearnableUpsample(in_channels=self.base_model.num_features)

    def forward(self, x):
        B = x.size(0)
        # Expected input: [B, 3, 224, 224]. Output from backbone: [B, 7, 7, 1024]
        x = self.base_model.forward_features(x)
        x = x.permute(0, 3, 1, 2)  # [B, 1024, 7, 7]
        x = self.upsample(x)  # [B, 1024, 28, 28]
        x = x.permute(0, 2, 3, 1)  # [B, 28, 28, 1024]
        x = x.reshape(B, 28 * 28, -1)  # [B, 784, 1024]
        logits = self.head(x)  # [B, 784, 5]
        return logits


# =============================================================================
# Helper Function: predict_full_image
# =============================================================================
def predict_full_image(model, image, device):
    """
    Splits a full image (448x672) into 6 fixed 224x224 patches, runs each patch
    through the model, and stitches the 28x28 token outputs into a full prediction.
    Returns a tensor of shape [56*84, 5].
    """
    patches = []
    for i in range(2):  # 2 rows
        for j in range(3):  # 3 columns
            patch = image[:, i * 224:(i + 1) * 224, j * 224:(j + 1) * 224]
            patches.append(patch)
    patch_batch = torch.stack(patches, dim=0).to(device)  # [6, 3, 224, 224]
    patch_logits = model(patch_batch)  # [6, 784, 5]
    num_classes = patch_logits.shape[-1]
    # Reshape each patch's output to [28, 28, num_classes]
    patch_outputs = patch_logits.reshape(6, 28, 28, num_classes)  # [6, 28, 28, 5]
    top_row = torch.cat((patch_outputs[0], patch_outputs[1], patch_outputs[2]), dim=1)  # [28, 84, 5]
    bottom_row = torch.cat((patch_outputs[3], patch_outputs[4], patch_outputs[5]), dim=1)  # [28, 84, 5]
    full_output = torch.cat((top_row, bottom_row), dim=0)  # [56, 84, 5]
    full_output = full_output.reshape(-1, num_classes)  # [56*84, 5]
    return full_output


# =============================================================================
# Evaluation Function
# =============================================================================
def evaluate_model(model, dataloader, device):
    model.eval()
    mae_total = 0.0
    mape_total = 0.0
    n_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)
            batch_outputs = []
            for i in range(B):
                full_output = predict_full_image(model, images[i], device)  # [56*84, 5]
                batch_outputs.append(full_output)
            logits = torch.stack(batch_outputs, dim=0)  # [B, 56*84, 5]
            probs = torch.softmax(logits, dim=-1)
            counts_range = torch.arange(5, dtype=torch.float32, device=device)
            expected_counts = (probs * counts_range).sum(dim=-1).sum(dim=-1)
            gt_counts = labels.argmax(dim=-1).sum(dim=-1).float()
            mae = torch.abs(expected_counts - gt_counts)
            mape = torch.where(gt_counts > 0, mae / gt_counts, torch.zeros_like(mae))
            mae_total += mae.sum().item()
            mape_total += mape.sum().item()
            n_samples += B
    return mae_total / n_samples, mape_total / n_samples


# =============================================================================
# Visualization Function
# =============================================================================
def visualize_sample(model, dataset, device):
    model.eval()
    # For visualization, always use full image evaluation.
    idx = random.randint(0, len(dataset)-1)
    image, labels = dataset[idx]  # image: [3, 448, 672], labels: [56*84, 5]
    image = image.to(device)
    # Get full image prediction.
    logits = predict_full_image(model, image, device)  # [56*84, 5]
    counts_range = torch.arange(5, dtype=torch.float32, device=device)
    pred_density = (torch.softmax(logits, dim=-1) * counts_range).sum(dim=-1)
    pred_density = pred_density.cpu().detach().numpy().reshape(56,84)
    gt_density = labels.argmax(dim=-1).cpu().numpy().reshape(56,84)
    pred_total = pred_density.sum()
    gt_total = gt_density.sum()
    # Denormalize image.
    img_np = image.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img_np = img_np * std + mean
    img_np = np.transpose(img_np, (1,2,0))
    # Create a figure.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Panel 1: Full image with grid overlay and predicted count in each 8x8 cell.
    axs[0].imshow(img_np)
    H, W = 448, 672
    cell_h, cell_w = 8, 8
    for x in range(0, W + 1, cell_w):
        axs[0].plot([x, x], [0, H], color='red', linewidth=0.5)
    for y in range(0, H + 1, cell_h):
        axs[0].plot([0, W], [y, y], color='red', linewidth=0.5)
    for i in range(56):
        for j in range(84):
            cnt = pred_density[i, j]
            axs[0].text(j * cell_w + cell_w / 2, i * cell_h + cell_h / 2, f"{cnt:.1f}",
                        color='yellow', fontsize=3, ha='center', va='center')
    axs[0].set_title(f"Full Image with Grid\nPred Total: {pred_total:.1f}, GT Total: {gt_total:.1f}")
    axs[0].axis("off")
    # Panel 2: Ground truth density map.
    axs[1].imshow(gt_density, cmap="viridis")
    axs[1].set_title("Ground Truth Density Map")
    axs[1].axis("off")
    # Panel 3: Predicted density map.
    axs[2].imshow(pred_density, cmap="viridis")
    axs[2].set_title("Predicted Density Map")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Test Evaluation
# =============================================================================
if __name__ == "__main__":
    test_root = "datasets/partA/test_data"
    test_dataset = ShanghaiTechADataset_ViT(root_dir=test_root, patch_size=8, mode="eval")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = CrowdCountingSwin().to(device)
    ckpt_files = glob.glob("best_vit_model_*_*.pth")
    if ckpt_files:
        ckpt = ckpt_files[0]
        print(f"Loading checkpoint from {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print("No checkpoint found!")

    mae, mape = evaluate_model(model, test_loader, device)
    print(f"Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}")

    # Visualize a few test samples.
    for _ in range(3):
        visualize_sample(model, test_dataset, device)
