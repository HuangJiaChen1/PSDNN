
import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
from mat4py import loadmat
import matplotlib.pyplot as plt
import torchvision.transforms as T


# ---------------------------
# Global Settings
# ---------------------------
BLOCK_SIZE = 32
# These are the discrete count values (classes) used by your model.
# For counts 0 to 10, and 15 representing “more than 10”
CLASS_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
NUM_CLASSES = len(CLASS_VALUES)


# ---------------------------
# Bigger ViT-EBC Model Definition
# ---------------------------
class ViT_EBC_Big(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256, depth=8, num_heads=8,
                 num_classes=NUM_CLASSES):
        """
        A larger Vision Transformer model for Enhanced Blockwise Classification.
          - img_size: size of the input patch (32x32)
          - patch_size: sub-patch size (e.g., 8) so that 32x32 becomes 4x4 tokens.
          - embed_dim: embedding dimension (set to 256)
          - depth: number of transformer layers (set to 8)
          - num_heads: number of attention heads (set to 8)
          - num_classes: number of discrete classes (12)
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Project image patches into embeddings.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

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
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # Use [CLS] token output.
        logits = self.head(cls_out)  # [B, num_classes]
        return logits


# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(model, image_path, transform, device):
    """
    Divides a test image into BLOCK_SIZE x BLOCK_SIZE patches,
    runs the model on each patch, computes the expected count per patch,
    and returns the full density map and total predicted count.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    n_blocks_x = width // BLOCK_SIZE
    n_blocks_y = height // BLOCK_SIZE
    density_map = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                left = bx * BLOCK_SIZE
                upper = by * BLOCK_SIZE
                right = left + BLOCK_SIZE
                lower = upper + BLOCK_SIZE
                patch = image.crop((left, upper, right, lower))
                patch_tensor = transform(patch).unsqueeze(0).to(device)
                logits = model(patch_tensor)  # [1, NUM_CLASSES]
                probs = torch.softmax(logits, dim=-1)
                rep_tensor = torch.tensor(CLASS_VALUES, dtype=torch.float32, device=device)
                expected_count = (probs * rep_tensor).sum().item()
                density_map[by, bx] = expected_count
    total_count = density_map.sum()
    return image, density_map, total_count


# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, test_images_dir, test_gt_dir, transform, device):
    """
    Loops over all test images, computes predicted total counts and ground truth counts,
    and then calculates MAE and MAPE.
    """
    predicted_counts = []
    gt_counts = []
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.jpg')])

    for img_file in tqdm(image_files, desc="Evaluating Test Images"):
        image_path = os.path.join(test_images_dir, img_file)
        gt_file = "GT_" + os.path.splitext(img_file)[0] + ".mat"
        gt_path = os.path.join(test_gt_dir, gt_file)

        # Predict using the model.
        i, d, pred_total = predict_image(model, image_path, transform, device)
        predicted_counts.append(pred_total)

        # Load ground truth from MAT file.
        try:
            gt_data = loadmat(gt_path)
            points = gt_data['image_info']['location']
            gt_total = len(points)
        except Exception as e:
            print(f"Error loading GT for {img_file}: {e}")
            gt_total = 0
        gt_counts.append(gt_total)
        # comment here
        visualize_image_prediction(i, d, pred_total)
        print(f"{img_file}: Predicted = {pred_total:.1f}, Ground Truth = {gt_total}")

    predicted_counts = np.array(predicted_counts)
    gt_counts = np.array(gt_counts)
    mae = np.mean(np.abs(predicted_counts - gt_counts))
    eps = 1e-6
    mape = np.mean(np.abs(predicted_counts - gt_counts) / (gt_counts + eps)) * 100.0

    return mae, mape, predicted_counts, gt_counts

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
# Main Evaluation Script
# ---------------------------
if __name__ == "__main__":
    # Set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transform = T.Compose([
    #     T.Resize((BLOCK_SIZE, BLOCK_SIZE)),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])
    # ])

    # more advanced transform
    transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop((BLOCK_SIZE, BLOCK_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    test_images_dir = "datasets/partB/test_data/images"  # Adjust as needed.
    test_gt_dir = "datasets/partB/test_data/ground_truth"  # Adjust as needed.

    model = ViT_EBC_Big(img_size=BLOCK_SIZE, patch_size=4, in_chans=3, embed_dim=256, depth=8, num_heads=8,
                        num_classes=NUM_CLASSES)
    model.to(device)
    checkpoint_path = "vit_ebc_shtechA.pth"  # Adjust if needed.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    mae, mape, predicted_counts, gt_counts = evaluate_model(model, test_images_dir, test_gt_dir, transform, device)
    print(f"Test MAE: {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")

    # Optionally, you can print out results for each image.
    for i, (pred, gt) in enumerate(zip(predicted_counts, gt_counts)):
        print(f"Image {i + 1}: Predicted = {pred:.1f}, Ground Truth = {gt}")
