import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from mat4py import loadmat

# Import timm's modules for the Vision Transformer
from timm_check.models.vision_transformer import PatchEmbed, Block

# --------------------------
# Helper functions: Gaussian kernel and density map generation
# --------------------------
def gaussian_kernel(kernel_size=15, sigma=4):
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)

def generate_density_map(points, img_shape, sigma=4):
    density_map = np.zeros(img_shape, dtype=np.float32)
    k_size = 15
    kernel = gaussian_kernel(kernel_size=k_size, sigma=sigma)
    offset = k_size // 2
    H, W = img_shape
    for pt in points:
        x = int(round(pt[0]))
        y = int(round(pt[1]))
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
# -----------------------------------------------------------------------------
# Helper: Sin-cos positional embedding (2D)
# -----------------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: total dimension (should be divisible by 4)
    grid_size: int, assuming a square grid (grid_size x grid_size patches)
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0, "Embed dim must be divisible by 4"
    dim_each = embed_dim // 4  # for each coordinate: sin and cos

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # grid[0] and grid[1] shape: (grid_size, grid_size)
    grid = np.stack(grid, axis=-1)       # (grid_size, grid_size, 2)
    grid = grid.reshape(-1, 2)            # (L, 2), where L = grid_size*grid_size

    pos_embed_list = []
    for i in range(2):
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

# -----------------------------------------------------------------------------
# Dataset: ShanghaiTech Part A
# -----------------------------------------------------------------------------
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
        if self.transform:
            img_transformed = self.transform(img)
        else:
            img_transformed = img
        gt_name = "GT_" + img_name.split('.')[0] + ".mat"
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = loadmat(gt_path)
        locations = mat['image_info']['location']
        locations = [(float(p[0]), float(p[1])) for p in locations]
        new_size = (384, 384)
        scale_x = new_size[0] / orig_w
        scale_y = new_size[1] / orig_h
        scaled_locations = [(p[0]*scale_x, p[1]*scale_y) for p in locations]
        density_map = generate_density_map(scaled_locations, new_size, sigma=4)
        density_map = torch.from_numpy(density_map).unsqueeze(0)  # (1, 384, 384)
        count = len(locations)
        return img_transformed, density_map, count

# -----------------------------------------------------------------------------
# Helper: Resize positional embeddings if needed
# -----------------------------------------------------------------------------
def resize_pos_embed(old_posemb, new_posemb):
    # old_posemb: (1, old_num_tokens, dim), new_posemb: (1, new_num_tokens, dim)
    old_posemb = old_posemb[0]  # (old_num_tokens, dim)
    old_num_tokens, dim = old_posemb.shape
    new_num_tokens = new_posemb.shape[1]
    old_grid_size = int(np.sqrt(old_num_tokens))
    new_grid_size = int(np.sqrt(new_num_tokens))
    old_posemb = old_posemb.reshape(1, old_grid_size, old_grid_size, dim)
    new_posemb = torch.nn.functional.interpolate(old_posemb, size=(new_grid_size, new_grid_size),
                                                 mode='bicubic', align_corners=False)
    new_posemb = new_posemb.reshape(1, new_grid_size * new_grid_size, dim)
    return new_posemb

# -----------------------------------------------------------------------------
# Counting Model: Encoder using timm's PatchEmbed & Block, plus decoder
# -----------------------------------------------------------------------------
class CountingEncoder(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, dropout=0.1):
        super(CountingEncoder, self).__init__()
        # Use timm's PatchEmbed and Block modules.
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # For 384x384 and patch 16, num_patches = 576
        # Fixed sin-cos positional embeddings (not learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed_np = get_2d_sincos_pos_embed(embed_dim, int(np.sqrt(num_patches)), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float())
        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        # x: (B, 3, 384, 384)
        x = self.patch_embed(x)           # (B, 576, 1024)
        x = x + self.pos_embed            # add fixed positional embeddings
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, 576, 1024)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(Decoder, self).__init__()
        # Upsample from 24x24 to 384x384 (4 stages: 24->48->96->192->384)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 24->48
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 48->96
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 96->192
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 192->384
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.encoder = CountingEncoder(img_size=384, patch_size=16, in_chans=3,
                                       embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, dropout=0.1)
        # The decoder takes the encoder output reshaped as a 2D feature map.
        # For 384x384 images and patch size 16, grid size = 24x24.
        self.decoder = Decoder(in_channels=1024, out_channels=1)
    def forward(self, x):
        # x: (B, 3, 384, 384)
        tokens = self.encoder(x)  # (B, 576, 1024)
        B, N, D = tokens.shape
        grid_size = int(np.sqrt(N))  # should be 24
        feat_map = tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)  # (B, 1024, 24, 24)
        density_map = self.decoder(feat_map)  # (B, 1, 384, 384)
        return density_map

# -----------------------------------------------------------------------------
# Helper: Load MAE Pretrained Encoder
# -----------------------------------------------------------------------------
def load_pretrained_encoder(model, checkpoint_path, device):
    """
    Loads pretrained MAE weights into the counting model's encoder.
    Assumes the MAE checkpoint was saved with timm's PatchEmbed and Block modules.
    If the positional embeddings sizes differ, they will be resized.
    """
    mae_state = torch.load(checkpoint_path, map_location=device)
    # We expect keys: "patch_embed.proj.weight", "pos_embed", "blocks", "norm"
    encoder_keys = ["patch_embed", "pos_embed", "blocks", "norm"]
    pretrained_state = {k: v for k, v in mae_state.items() if any(k.startswith(prefix) for prefix in encoder_keys)}
    if "pos_embed" in pretrained_state:
        pretrained_posemb = pretrained_state["pos_embed"]
        if pretrained_posemb.shape[1] != model.encoder.pos_embed.shape[1]:
            print("Resizing pos_embed from", pretrained_posemb.shape, "to", model.encoder.pos_embed.shape)
            new_posemb = resize_pos_embed(pretrained_posemb, model.encoder.pos_embed)
            pretrained_state["pos_embed"] = new_posemb
    missing, unexpected = model.encoder.load_state_dict(pretrained_state, strict=False)
    print("Pretrained encoder load -- Missing keys:", missing, "Unexpected keys:", unexpected)

# -----------------------------------------------------------------------------
# Visualization helper
# -----------------------------------------------------------------------------
def visualize_results(img_tensor, gt_density, pred_density, epoch, count_gt, count_pred):
    img = img_tensor.permute(1,2,0).cpu().numpy()
    gt_map = gt_density.squeeze(0).cpu().numpy()
    pred_map = pred_density.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(15,4))
    plt.suptitle(f"Epoch {epoch} | GT Count: {count_gt} | Predicted Count: {count_pred:.2f}")
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(gt_map, cmap="jet")
    plt.title("Ground Truth Density")
    plt.colorbar()
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(pred_map, cmap="jet")
    plt.title("Predicted Density")
    plt.colorbar()
    plt.axis("off")
    plt.show()

# -----------------------------------------------------------------------------
# Training and Validation Loop
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    dataset_root = "datasets/partA"  # update path as needed
    dataset = ShanghaiTechA(root=dataset_root, split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    model = CrowdCounter().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Load pretrained MAE weights into encoder
    load_pretrained_encoder(model, "mae_pretrain_epoch155.pth", device)

    num_epochs = 100
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, gt_densities, _ in dataloader:
            imgs = imgs.to(device)
            gt_densities = gt_densities.to(device)
            optimizer.zero_grad()
            pred_densities = model(imgs)
            loss = criterion(pred_densities, gt_densities)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0) *60
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

        model.eval()
        with torch.no_grad():
            img_sample, gt_density_sample, count_gt = dataset[1]
            img_sample_tensor = img_sample.unsqueeze(0).to(device)
            pred_density_sample = model(img_sample_tensor)
            count_pred = pred_density_sample.sum().item()
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            img_sample_disp = inv_normalize(img_sample_tensor.squeeze(0).cpu())
            img_sample_disp = torch.clamp(img_sample_disp, 0, 1)
            visualize_results(img_sample_disp, gt_density_sample, pred_density_sample.squeeze(0),
                              epoch, count_gt, count_pred)

    print("Training complete.")
    torch.save(model.state_dict(), "ED_sha.pth")

if __name__ == "__main__":
    main()
