import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
from mat4py import loadmat
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ---------------------------
# Global Settings
# ---------------------------
# Block (patch) size in pixels (each training sample is a 32x32 patch)
BLOCK_SIZE = 32

# Define the discrete classes for count prediction.
# These represent the number of people in a patch. Note that class 11 (value 15) means “> 10”
CLASS_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
NUM_CLASSES = len(CLASS_VALUES)
# Lambda weight for the regression (count) loss component.
LAMBDA_COUNT = 1.0


# ---------------------------
# Utility Functions for Labeling
# ---------------------------
def get_patch_targets(point_count):
    """
    Given the number of annotated points in a patch,
    return:
      - a classification label (integer index between 0 and 11) and
      - the ground-truth count (as a float, clamped to 15).
    If count > 10, we assign the index corresponding to value 15.
    """
    if point_count > 10:
        return NUM_CLASSES - 1, 15.0
    else:
        # For counts 0 to 10, the class label is the count itself.
        return int(point_count), float(point_count)


# ---------------------------
# Dataset: ShanghaiTech A for VIT-EBC
# ---------------------------
class ShanghaiTechADataset_ViT(Dataset):
    def __init__(self, root_dir, block_size, transform):
        """
        Args:
            root_dir (str): Path to dataset split (e.g., "datasets/partA/train_data")
            block_size (int): Patch size (e.g., 32)
            transform: Transform to apply (e.g., resize/normalize for your model)
        """
        self.root_dir = root_dir
        self.block_size = block_size
        self.transform = transform
        self.samples = []  # each sample is a dict with keys: 'img_path', 'bbox', 'class_label', 'gt_count'
        self._prepare_samples()

    def _prepare_samples(self):
        images_dir = os.path.join(self.root_dir, "images")
        gt_dir = os.path.join(self.root_dir, "ground_truth")
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            # Assume GT file naming: "GT_" + image file base + ".mat"
            gt_file = "GT_" + os.path.splitext(img_file)[0] + ".mat"
            gt_path = os.path.join(gt_dir, gt_file)
            # Load GT points
            try:
                gt_data = loadmat(gt_path)
                points = gt_data['image_info']['location']
                points = [(float(p[0]), float(p[1])) for p in points]
            except Exception as e:
                print(f"Error loading {gt_path}: {e}")
                points = []
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue
            width, height = image.size
            n_blocks_x = width // self.block_size
            n_blocks_y = height // self.block_size
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):
                    left = bx * self.block_size
                    upper = by * self.block_size
                    right = left + self.block_size
                    lower = upper + self.block_size
                    # Count points in this patch
                    count = sum(1 for (x, y) in points if left <= x < right and upper <= y < lower)
                    class_label, gt_count = get_patch_targets(count)
                    self.samples.append({
                        'img_path': img_path,
                        'bbox': (left, upper, right, lower),
                        'class_label': class_label,
                        'gt_count': gt_count
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert("RGB")
        patch = image.crop(sample['bbox'])
        patch = self.transform(patch)
        return patch, sample['class_label'], sample['gt_count']


# ---------------------------
# Minimal Vision Transformer for EBC
# ---------------------------
class ViT_EBC(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256, depth=8, num_heads=8,
                 num_classes=NUM_CLASSES):
        """
        A larger Vision Transformer for EBC.
          - img_size: size of the input patch (e.g., 32)
          - patch_size: sub-patch size (e.g., 8). So 32x32 becomes 4x4 tokens.
          - embed_dim: embedding dimension (increased to 256)
          - depth: number of transformer layers (increased to 8)
          - num_heads: number of attention heads (increased to 8)
          - num_classes: output classes (12)
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Project patches into embeddings using a conv layer.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)  # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # Use the [CLS] token.
        logits = self.head(cls_out)  # [B, num_classes]
        return logits


# ---------------------------
# Loss Functions
# ---------------------------
# We use CrossEntropyLoss for classification and L1Loss for count regression.
# The expected count is computed as the weighted sum over class probabilities.
cross_entropy_loss = nn.CrossEntropyLoss()
l1_loss = nn.L1Loss()


# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for patches, class_labels, gt_counts in tqdm(dataloader, desc="Training", leave=False):
        patches = patches.to(device)
        class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
        gt_counts = torch.tensor(gt_counts, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        logits = model(patches)  # shape: [batch_size, NUM_CLASSES]
        # Classification loss
        loss_class = cross_entropy_loss(logits, class_labels)
        # Compute probabilities and expected count
        probs = torch.softmax(logits, dim=-1)  # [B, NUM_CLASSES]
        rep_tensor = torch.tensor(CLASS_VALUES, dtype=torch.float32, device=device)  # [NUM_CLASSES]
        expected_count = (probs * rep_tensor).sum(dim=-1)  # [B]
        # Count loss: L1 loss between expected count and ground truth (clamped to 15)
        loss_count = l1_loss(expected_count, gt_counts)
        loss = loss_class + LAMBDA_COUNT * loss_count
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for patches, class_labels, gt_counts in tqdm(dataloader, desc="Evaluating", leave=False):
            patches = patches.to(device)
            class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
            gt_counts = torch.tensor(gt_counts, dtype=torch.float32, device=device)
            logits = model(patches)
            loss_class = cross_entropy_loss(logits, class_labels)
            probs = torch.softmax(logits, dim=-1)
            rep_tensor = torch.tensor(CLASS_VALUES, dtype=torch.float32, device=device)
            expected_count = (probs * rep_tensor).sum(dim=-1)
            loss_count = l1_loss(expected_count, gt_counts)
            loss = loss_class + LAMBDA_COUNT * loss_count
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ---------------------------
# Testing/Visualization on a Full Image
# ---------------------------
def predict_image(model, image_path, transform, device):
    """
    For a test image, divide it into 32x32 patches,
    predict the count class for each patch, compute the expected count,
    and assemble a density map.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    n_blocks_x = width // BLOCK_SIZE
    n_blocks_y = height // BLOCK_SIZE
    density_map = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for by in tqdm(range(n_blocks_y), desc="Processing blocks", leave=False):
            for bx in range(n_blocks_x):
                left = bx * BLOCK_SIZE
                upper = by * BLOCK_SIZE
                right = left + BLOCK_SIZE
                lower = upper + BLOCK_SIZE
                patch = image.crop((left, upper, right, lower))
                patch_tensor = transform(patch).unsqueeze(0).to(device)
                logits = model(patch_tensor)
                probs = torch.softmax(logits, dim=-1)
                rep_tensor = torch.tensor(CLASS_VALUES, dtype=torch.float32, device=device)
                expected_count = (probs * rep_tensor).sum().item()
                density_map[by, bx] = expected_count
    total_count = density_map.sum()
    return image, density_map, total_count


def visualize_image_prediction(image, density_map, total_count):
    """
    Visualizes the image with block-level predictions and the density heatmap.
    """
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except IOError:
        font = ImageFont.load_default()
    n_blocks_y, n_blocks_x = density_map.shape
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            left = bx * BLOCK_SIZE
            upper = by * BLOCK_SIZE
            right = left + BLOCK_SIZE
            lower = upper + BLOCK_SIZE
            count = density_map[by, bx]
            draw.rectangle([left, upper, right, lower], outline="red", width=1)
            draw.text((left + 2, upper + 2), f"{count:.1f}", fill="yellow", font=font)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_with_boxes)
    plt.title(f"Predicted Total Count: {total_count:.1f}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(density_map, interpolation="nearest", cmap="hot")
    plt.colorbar(label="Predicted count per patch")
    plt.title("Density Map")
    plt.xlabel("Block X")
    plt.ylabel("Block Y")
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main Script: Training and Testing
# ---------------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop((BLOCK_SIZE, BLOCK_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # train_root = "datasets/partA/train_data"
    # full_dataset = ShanghaiTechADataset_ViT(root_dir=train_root, block_size=BLOCK_SIZE, transform=transform)
    # dataset_size = len(full_dataset)
    # val_size = int(0.2 * dataset_size)
    # train_size = dataset_size - val_size
    # train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    #
    # # Create dataloaders for each split.
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    train_root = "datasets/partA/train_data"  # adjust path as needed
    train_dataset = ShanghaiTechADataset_ViT(root_dir=train_root, block_size=BLOCK_SIZE, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    # Initialize the ViT-EBC model.
    model = ViT_EBC(img_size=BLOCK_SIZE, patch_size=4, in_chans=3, embed_dim=256, depth=8, num_heads=8,
                    num_classes=NUM_CLASSES)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        # val_loss = evaluate(model, val_loader, device)
        # print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}")

    # Save the trained model.
    torch.save(model.state_dict(), "vit_ebc_shtechA.pth")
    print("Training complete and model saved.")

    # ---------------------------
    # Testing on a Full Image from PartA Test Data
    # ---------------------------
    test_image_path = "datasets/partA/test_data/images/IMG_3.jpg"  # adjust as needed
    image, density_map, total_count = predict_image(model, test_image_path, transform, device)
    print(f"Predicted total count for test image: {total_count:.1f}")

    # (Optionally) Load ground truth points from the GT file for comparison.
    gt_path = "datasets/partA/test_data/ground_truth/GT_IMG_3.mat"
    try:
        gt_data = loadmat(gt_path)
        points = gt_data['image_info']['location']
        gt_total = len(points)
        print(f"Ground truth total count: {gt_total}")
    except Exception as e:
        print(f"Error loading GT: {e}")

    visualize_image_prediction(image, density_map, total_count)
