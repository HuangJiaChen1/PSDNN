import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from mat4py import loadmat
from timm.models.vision_transformer import PatchEmbed, Block

# --------------------------
# Helper functions: Gaussian kernel and density map generation
# --------------------------

def gaussian_kernel(kernel_size=15, sigma=4):
    """Generate a 2D Gaussian kernel."""
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def generate_density_map(points, img_shape, sigma=4):
    """
    Generate a density map given head coordinates.

    Args:
        points: list of (x, y) tuples (in the resized image coordinate system)
        img_shape: tuple (H, W)
        sigma: standard deviation for the Gaussian kernel
    Returns:
        density_map: numpy array of shape (H, W)
    """
    density_map = np.zeros(img_shape, dtype=np.float32)
    k_size = 15  # kernel size (odd number)
    kernel = gaussian_kernel(kernel_size=k_size, sigma=sigma)
    offset = k_size // 2
    H, W = img_shape
    for pt in points:
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        # Clamp coordinates to be within valid image bounds
        x = min(max(x, 0), W - 1)
        y = min(max(y, 0), H - 1)
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(W, x + offset + 1)
        y2 = min(H, y + offset + 1)
        kx1 = offset - (x - x1)
        ky1 = offset - (y - y1)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)
        density_map[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
    return density_map

# --------------------------
# Helper: 2D sin-cos positional embeddings
# --------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: total embedding dimension (must be divisible by 4)
    grid_size: number of patches per side (assumes square grid)
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0, "Embed dim must be divisible by 4"
    dim_each = embed_dim // 4  # for sin and cos per coordinate
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # grid[0] and grid[1] each (grid_size, grid_size)
    grid = np.stack(grid, axis=-1)       # (grid_size, grid_size, 2)
    grid = grid.reshape(-1, 2)            # (L, 2) where L = grid_size*grid_size

    pos_embed_list = []
    for i in range(2):  # for x and y coordinates
        pos_i = grid[:, i:i+1]  # (L, 1)
        omega = 1.0 / (10000 ** (np.arange(dim_each, dtype=np.float32) / dim_each))
        pos_i = pos_i * omega[None, :]  # (L, dim_each)
        sin_embed = np.sin(pos_i)
        cos_embed = np.cos(pos_i)
        pos_embed_list.append(np.concatenate([sin_embed, cos_embed], axis=1))  # (L, 2*dim_each)
    pos_embed = np.concatenate(pos_embed_list, axis=1)  # (L, 4*dim_each) = (L, embed_dim)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

# --------------------------
# Decoder: Upsampling to obtain the density map
# --------------------------
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(Decoder, self).__init__()
        # Four upsampling blocks to go from 24x24 to 384x384 (scale factor 2 each time: 24->48->96->192->384)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x

# --------------------------
# Dataset for ShanghaiTech Part A (384x384 images)
# --------------------------
class ShanghaiTechA(Dataset):
    def __init__(self, root, split='train', transform=None):
        if split == 'train':
            self.image_dir = os.path.join(root, 'train_data', 'images')
            self.gt_dir = os.path.join(root, 'train_data', 'ground_truth')
        else:
            self.image_dir = os.path.join(root, 'test_data', 'images')
            self.gt_dir = os.path.join(root, 'test_data', 'ground_truth')
        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        if self.transform is not None:
            img_transformed = self.transform(img)
        else:
            img_transformed = img
        gt_name = "GT_" + img_name.split('.')[0] + ".mat"
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = loadmat(gt_path)
        locations = mat['image_info']['location']
        locations = [(float(p[0]), float(p[1])) for p in locations]
        # Scale coordinates for 384x384
        new_size = (384, 384)
        scale_x = new_size[0] / orig_w
        scale_y = new_size[1] / orig_h
        scaled_locations = [(p[0]*scale_x, p[1]*scale_y) for p in locations]
        density_map = generate_density_map(scaled_locations, new_size, sigma=4)
        density_map = torch.from_numpy(density_map).unsqueeze(0)  # (1, 384, 384)
        count = len(locations)
        return img_transformed, density_map, count

# --------------------------
# Counting ViT Model using timm modules
# --------------------------
class CountingViT(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, dropout=0.1):
        super(CountingViT, self).__init__()
        # Use timm's PatchEmbed (already imported) to split the image into patches.
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # For 384x384 with patch size 16: 24*24 = 576
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        # Initialize fixed sin-cos positional embeddings
        pos_embed_np = get_2d_sincos_pos_embed(embed_dim, int(np.sqrt(num_patches)), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float())
        self.pos_drop = nn.Dropout(p=dropout)
        # Use timm's Block modules for transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Decoder to upsample from 24x24 feature map to 384x384 density map.
        self.decoder = Decoder(in_channels=embed_dim, out_channels=1)
    def forward(self, x):
        x = self.patch_embed(x)  # (B, 576, 1024)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (B, 576, 1024)
        B, N, D = x.shape
        h = w = int(np.sqrt(N))  # should be 24
        feat_map = x.transpose(1, 2).reshape(B, D, h, w)  # (B, 1024, 24, 24)
        density_map = self.decoder(feat_map)  # (B, 1, 384, 384)
        return density_map



# --------------------------
# Visualization helper
# --------------------------

def visualize_results(img_tensor, gt_density, pred_density, idx, gt_count, pred_count):
    """Plot input image, ground truth and predicted density maps."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    gt_map = gt_density.squeeze().cpu().numpy()

    pred_map = pred_density.squeeze(0).detach().cpu().numpy()

    plt.figure(figsize=(15, 4))
    plt.suptitle(f"Sample {idx} | GT Count: {gt_count} | Predicted Count: {pred_count:.2f}")

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_map, cmap="jet")
    plt.title("Ground Truth Density")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_map, cmap="jet")
    plt.title("Predicted Density")
    plt.colorbar()
    plt.axis("off")

    plt.show()


# --------------------------
# Inference Script
# --------------------------

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms (resize, tensor conversion, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset_root = "datasets/partB"  # Update this path if needed
    test_dataset = ShanghaiTechA(root=dataset_root, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = CountingViT().to(device)
    # If you have trained weights, uncomment and update the path:
    model.load_state_dict(torch.load("YCV_best.pth", map_location=device))
    model.eval()

    total_mae = 0.0
    total_mape = 0.0
    num_samples = len(test_dataset)

    all_gt_counts = []
    all_pred_counts = []

    # For visualization, we will display the first 5 samples.
    vis_samples = 5
    with torch.no_grad():
        for idx, (img, gt_density, gt_count) in enumerate(test_loader):
            img = img.to(device)
            pred_density = model(img)
            # Count is the sum of the density map (for one image)
            pred_count = pred_density.sum().item()

            all_gt_counts.append(gt_count.item())
            all_pred_counts.append(pred_count)

            mae = abs(pred_count - gt_count.item())
            total_mae += mae
            if gt_count.item() != 0:
                mape = abs(pred_count - gt_count.item()) / gt_count.item() * 100
            else:
                mape = 0.0
            total_mape += mape

            # Visualize the first few samples
            if idx < vis_samples:
                # For display, undo normalization
                inv_normalize = transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                )
                img_disp = inv_normalize(img.squeeze(0).cpu())
                img_disp = torch.clamp(img_disp, 0, 1)
                visualize_results(img_disp, gt_density, pred_density.squeeze(0),
                                  idx, gt_count.item(), pred_count)

    overall_mae = total_mae / num_samples
    overall_mape = total_mape / num_samples
    print(f"Test MAE: {overall_mae:.2f}")
    print(f"Test MAPE: {overall_mape:.2f}%")


if __name__ == "__main__":
    inference()
