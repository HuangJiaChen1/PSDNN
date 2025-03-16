import glob
import os
import random
import re

import numpy as np
from mat4py import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Basic transforms for conversion and normalization.
to_tensor = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

# -----------------------------------------------------------------------------
# Dataset Class:
# - Each image is resized to 448x672.
# - In training mode, each image is split into 6 fixed 224x224 patches (2 rows x 3 cols)
#   and each patch (and its corresponding 28x28 counts grid) is returned as one sample,
#   enlarging the dataset by a factor of 6.
# - In evaluation, the full 448x672 image is returned.
# - The ground truth counts grid is computed on the resized image using patch_size=8.
# -----------------------------------------------------------------------------
class ShanghaiTechADataset_ViT(Dataset):
    def __init__(self, root_dir, patch_size=8, mode="train"):
        """
        mode: "train" or "eval"
        For training, each image is resized to 448×672 then split into 6 patches of 224×224.
        For evaluation, the full 448×672 image is returned.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size  # 8
        self.mode = mode
        self.samples = []
        self.full_size = (448, 672)  # (H, W)
        self.crop_size = (224, 224)  # for patches in training
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
            # Load ground truth points.
            gt_data = loadmat(gt_path)
            points = [(float(p[0]), float(p[1])) for p in gt_data['image_info']['location']]
            with Image.open(img_path) as img:
                orig_width, orig_height = img.size
            # Scale factors for full size.
            scale_x = target_w / orig_width
            scale_y = target_h / orig_height
            scaled_points = [(x * scale_x, y * scale_y) for (x, y) in points]
            # Compute counts grid on full image.
            n_patches_x = target_w // self.patch_size  # 672/8 = 84
            n_patches_y = target_h // self.patch_size  # 448/8 = 56
            counts = np.zeros((n_patches_y, n_patches_x), dtype=int)
            for (x, y) in scaled_points:
                px = min(int(x // self.patch_size), n_patches_x - 1)
                py = min(int(y // self.patch_size), n_patches_y - 1)
                counts[py, px] = min(counts[py, px] + 1, 4)
            if self.mode == "train":
                # In training, load and resize image.
                image = Image.open(img_path).convert("RGB")
                image = image.resize((target_w, target_h), Image.BILINEAR)
                # Split deterministically into 6 nonoverlapping patches (2 rows, 3 cols).
                for i in range(2):
                    for j in range(3):
                        top = i * 224
                        left = j * 224
                        patch = TF.crop(image, top, left, 224, 224)
                        # For the counts grid: 224/8 = 28.
                        grid_top = i * 28
                        grid_left = j * 28
                        label_patch = counts[grid_top:grid_top+28, grid_left:grid_left+28]
                        labels = np.zeros((28 * 28, 5), dtype=np.float32)
                        for m in range(28):
                            for n in range(28):
                                count = label_patch[m, n]
                                labels[m * 28 + n, count] = 1
                        self.samples.append({'image': patch, 'labels': labels})
            else:
                # Evaluation: store one sample per image.
                self.samples.append({'img_path': img_path, 'counts': counts})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "train":
            sample = self.samples[idx]
            image = sample['image']
            labels = sample['labels']
            image = to_tensor(image)
            image = normalize(image)
            labels = torch.tensor(labels, dtype=torch.float32)
            return image, labels
        else:
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

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------
class LearnableUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # ConvTranspose2d upsampling from 7x7 to 28x28.
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
        # We train the model on 224×224 patches.
        self.base_model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            img_size=(224, 224)
        )
        # Prediction head: map features to 5 classes.
        self.head = nn.Linear(self.base_model.num_features, 5)
        # Learnable upsample layer from 7x7 to 28x28.
        self.upsample = LearnableUpsample(in_channels=self.base_model.num_features)
        # Add a dropout layer with probability 0.5 (you can adjust this value)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        B = x.size(0)
        # Obtain feature map from the backbone.
        # Expected shape: [B, 7, 7, 1024] for a 224x224 input.
        x = self.base_model.forward_features(x)
        x = x.permute(0, 3, 1, 2)  # [B, 1024, 7, 7]
        x = self.upsample(x)        # [B, 1024, 28, 28]
        x = x.permute(0, 2, 3, 1)    # [B, 28, 28, 1024]
        x = x.reshape(B, 28 * 28, -1)  # [B, 784, 1024]
        # Apply dropout before the prediction head.
        x = self.dropout(x)
        logits = self.head(x)       # [B, 784, 5]
        return logits

# -----------------------------------------------------------------------------
# Helper function for evaluation:
# Given a full 448x672 image, split it into 6 patches (2 rows x 3 cols) and stitch predictions.
# -----------------------------------------------------------------------------
def predict_full_image(model, image, device):
    """
    image: tensor of shape [3, 448, 672]
    Returns:
       full_output: tensor of shape [56*84, 5] (i.e. 56 rows, 84 cols)
    """
    patches = []
    for i in range(2):
        for j in range(3):
            patch = image[:, i*224:(i+1)*224, j*224:(j+1)*224]
            patches.append(patch)
    patch_batch = torch.stack(patches, dim=0).to(device)  # [6, 3, 224, 224]
    patch_logits = model(patch_batch)  # [6, 784, 5]
    num_classes = patch_logits.shape[-1]
    # Reshape each patch's output to [28, 28, num_classes]
    patch_outputs = patch_logits.reshape(6, 28, 28, num_classes)  # [6, 28, 28, 5]
    top_row = torch.cat((patch_outputs[0], patch_outputs[1], patch_outputs[2]), dim=1)    # [28, 84, 5]
    bottom_row = torch.cat((patch_outputs[3], patch_outputs[4], patch_outputs[5]), dim=1)  # [28, 84, 5]
    full_output = torch.cat((top_row, bottom_row), dim=0)  # [56, 84, 5]
    full_output = full_output.reshape(-1, num_classes)     # [56*84, 5]
    return full_output

# -----------------------------------------------------------------------------
# Modified Evaluation Function (always uses full images)
# -----------------------------------------------------------------------------
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
            # Here, images are full images of shape [B, 3, 448, 672].
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

# -----------------------------------------------------------------------------
# Loss, Training, and Visualization Functions
# -----------------------------------------------------------------------------
cross_entropy_loss = nn.CrossEntropyLoss()

def wasserstein_distance(pred, target):
    p = torch.softmax(pred, dim=-1)
    q = target.float()
    Fp = torch.cumsum(p, dim=-1)
    Fq = torch.cumsum(q, dim=-1)
    wd = torch.sum(torch.abs(Fp - Fq), dim=-1)
    return wd.mean()

def total_loss(logits, labels, lambda_w=1.0):
    ce_loss = cross_entropy_loss(logits.view(-1, 5), labels.argmax(dim=-1).view(-1))
    w_loss = wasserstein_distance(logits, labels)
    return ce_loss + lambda_w * w_loss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)  # images are 224x224 patches in training.
        loss = total_loss(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

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
    cell_h, cell_w = 8, 8  # each patch in counts grid corresponds to 8x8 pixels.
    for x in range(0, W+1, cell_w):
        axs[0].plot([x, x], [0, H], color='red', linewidth=0.5)
    for y in range(0, H+1, cell_h):
        axs[0].plot([0, W], [y, y], color='red', linewidth=0.5)
    # Overlay predicted count in each cell.
    for i in range(56):
        for j in range(84):
            count_val = pred_density[i, j]
            axs[0].text(j*cell_w+cell_w/2, i*cell_h+cell_h/2, f"{count_val:.1f}",
                        color='yellow', fontsize=3, ha='center', va='center')
    axs[0].set_title(f"Full Image with Grid\nPredicted Total: {pred_total:.1f}, GT Total: {gt_total}")
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
    plt.savefig("VITEBC_VIS.png")

def train_and_validate(train_root, val_split=0.2, num_epochs=50, batch_size=8, patience=5):
    # For training, use mode "train" (dataset is enlarged 6×).
    full_dataset = ShanghaiTechADataset_ViT(root_dir=train_root, patch_size=8, mode="train")
    total_samples = len(full_dataset)
    val_size = int(val_split * total_samples)
    train_size = total_samples - val_size

    train_dataset, _ = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # For validation, we want full images. So create a separate eval dataset.
    eval_dataset = ShanghaiTechADataset_ViT(root_dir=train_root, patch_size=8, mode="eval")
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CrowdCountingSwin().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Look for an existing checkpoint with pattern: best_vit_model_{mape}_{epoch}.pth
    ckpt_files = glob.glob("best_vit_model_*_*.pth")
    start_epoch = 1
    best_val_mape = float('inf')
    if ckpt_files:
        # For simplicity, use the first one found (or pick the one with the lowest MAPE)
        ckpt_file = ckpt_files[0]
        pattern = r"best_vit_model_([0-9]+\.[0-9]+)_([0-9]+)\.pth"
        m = re.search(pattern, ckpt_file)
        if m:
            best_val_mape = float(m.group(1))
            start_epoch = int(m.group(2)) + 1
            print(f"Loading checkpoint from {ckpt_file} with best MAPE = {best_val_mape:.3f} at epoch {start_epoch-1}")
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
        else:
            print("Checkpoint found but pattern did not match; starting from scratch.")
    else:
        print("No checkpoint found; starting from scratch.")

    for epoch in range(start_epoch, num_epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()  # update LR per epoch (if using an epoch-level scheduler)
        mae, mape = evaluate_model(model, eval_loader, device)
        print(f"Epoch {epoch}/{num_epochs}: Train Loss = {train_loss:.4f} | Val MAE = {mae:.4f}, Val MAPE = {mape:.4f}")
        if mape < best_val_mape:
            best_val_mape = mape
            # Remove previous checkpoint(s)
            for f in glob.glob("best_vit_model_*_*.pth"):
                os.remove(f)
            checkpoint_path = f"best_vit_model_{best_val_mape:.3f}_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model found and saved with MAPE = {best_val_mape:.4f} at epoch {epoch}")
        else:
            print("No improvement in validation MAPE.")
        print("Visualizing a random sample from the validation set...")
        visualize_sample(model, eval_dataset, device)
    return model, eval_dataset

if __name__ == "__main__":
    train_root = "datasets/partA/train_data"
    model, val_dataset = train_and_validate(train_root, val_split=0.2, num_epochs=300, batch_size=8, patience=5)
