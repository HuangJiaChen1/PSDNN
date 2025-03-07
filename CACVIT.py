import glob
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mat4py import loadmat
import torchvision.transforms as transforms
import timm
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from torch.cuda.amp import autocast, GradScaler

#############################################
# Dataset Definition
#############################################
class CrowdCountingDataset(Dataset):
    def __init__(self, img_dir, gt_dir, exemplars_dir, transform=None, exemplar_transform=None, num_exemplars=3,
                 resize_shape=(384, 384)):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.exemplars_dir = exemplars_dir
        self.transform = transform
        self.exemplar_transform = exemplar_transform
        self.num_exemplars = num_exemplars
        self.resize_shape = resize_shape
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Load image
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size

        # Load ground truth annotations
        gt_name = 'GT_' + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = loadmat(gt_path)
        locations = mat['image_info']['location']
        locations = [(float(p[0]), float(p[1])) for p in locations]

        # Scale annotations to match the resized image
        scale_x = self.resize_shape[1] / img_width
        scale_y = self.resize_shape[0] / img_height
        scaled_locations = [(p[0] * scale_x, p[1] * scale_y) for p in locations]

        # Generate density map using scaled locations
        density_map = self.generate_density_map(self.resize_shape, scaled_locations)

        # Load exemplars using a naming pattern
        exemplar_prefix = 'EXAMPLARS_' + img_name.split('_')[1].split('.')[0] + '_'
        exemplar_paths = [os.path.join(self.exemplars_dir, f) for f in os.listdir(self.exemplars_dir)
                          if f.startswith(exemplar_prefix)]
        exemplars = []
        for p in exemplar_paths[:self.num_exemplars]:
            try:
                ex = Image.open(p).convert('RGB')
                exemplars.append(ex)
            except Exception as e:
                print(f"Error loading exemplar {p}: {e}")

        if len(exemplars) < self.num_exemplars:
            if len(exemplars) == 0:
                dummy_ex = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                exemplars = [dummy_ex] * self.num_exemplars
            else:
                exemplars = exemplars * (self.num_exemplars // len(exemplars)) + exemplars[:self.num_exemplars % len(exemplars)]
        elif len(exemplars) > self.num_exemplars:
            exemplars = exemplars[:self.num_exemplars]

        # Scales (assuming exemplar size is 64x64)
        scales = torch.ones((self.num_exemplars, 2), dtype=torch.float32) * (64 / max(img_width, img_height))

        if self.transform:
            image = self.transform(image)
        if self.exemplar_transform:
            exemplars = torch.stack([self.exemplar_transform(ex) for ex in exemplars])
        density_map = torch.from_numpy(density_map).float()

        # Return image, density map, exemplars, scales, and scaled annotation coordinates.
        return image, density_map, exemplars, scales, scaled_locations

    def generate_density_map(self, img_shape, points, sigma=2):
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

def gaussian_kernel(kernel_size=15, sigma=4):
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)

#############################################
# Transforms
#############################################
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

exemplar_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#############################################
# Patch Embedding Module (adapted from timm)
#############################################
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

#############################################
# CACViT Model Definition
#############################################
class CACViT(nn.Module):
    def __init__(self, num_exemplars=3, img_size=384, patch_size=16, embed_dim=768):
        super(CACViT, self).__init__()
        self.num_exemplars = num_exemplars
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=0)
        self.patch_embed_exemplar = PatchEmbed(img_size=64, patch_size=16, in_chans=3, embed_dim=embed_dim)
        self.num_patches_exemplar = self.patch_embed_exemplar.num_patches
        self.exemplar_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_exemplar, embed_dim))
        nn.init.trunc_normal_(self.exemplar_pos_embed, std=0.02)
        self.combined_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_exemplars + 1, embed_dim))
        nn.init.trunc_normal_(self.combined_pos_embed, std=0.02)
        self.scale_embed = nn.Linear(2, embed_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU()  # Final activation; consider adjusting if outputs remain near 0.
        )

    def scale_embedding(self, exemplars, scale_infos):
        # Using method 1.
        bs, n, c, h, w = exemplars.shape
        scales_batch = []
        for i in range(bs):
            scales = []
            for j in range(n):
                w_scale = torch.linspace(0, scale_infos[i, j, 0], w)
                w_scale = repeat(w_scale, 'w->h w', h=h).unsqueeze(0)
                h_scale = torch.linspace(0, scale_infos[i, j, 1], h)
                h_scale = repeat(h_scale, 'h->h w', w=w).unsqueeze(0)
                scale = w_scale + h_scale
                scales.append(scale)
            scales = torch.stack(scales)
            scales_batch.append(scales)
        scales_batch = torch.stack(scales_batch)
        scales_batch = scales_batch.to(exemplars.device)
        exemplars = torch.cat((exemplars, scales_batch), dim=2)
        return exemplars

    def forward(self, inputs):
        samples, boxes, scales = inputs  # samples: (B, 3, 384, 384); boxes: (B, num_exemplars, 3, 64, 64)
        batch_size = samples.shape[0]
        boxes = self.scale_embedding(boxes, scales)
        x = self.vit.patch_embed(samples)
        x = x + self.vit.pos_embed[:, 1:, :]
        cls_token = self.vit.cls_token.expand(batch_size, -1, -1)
        cls_token = cls_token + self.vit.pos_embed[:, :1, :]
        x = torch.cat([cls_token, x], dim=1)
        for blk in self.vit.blocks:
            x = blk(x)
        img_features = self.vit.norm(x)[:, 1:, :]
        exemplar_input = boxes.view(batch_size * (self.num_exemplars+1), 3, 64, 64)
        exemplar_features = self.patch_embed_exemplar(exemplar_input)
        exemplar_features = exemplar_features + self.exemplar_pos_embed
        exemplar_features = exemplar_features.view(batch_size, (self.num_exemplars+1), self.num_patches_exemplar, self.embed_dim)
        exemplar_features = exemplar_features.mean(dim=2)
        combined_features = torch.cat([img_features, exemplar_features], dim=1)
        combined_features = combined_features + self.combined_pos_embed
        combined_features = self.vit.norm(combined_features)
        img_features = combined_features[:, :self.num_patches, :]
        h = w = self.img_size // self.patch_size  # For 384/16 = 24
        img_features = img_features.permute(0, 2, 1).reshape(batch_size, self.embed_dim, h, w)
        density_map = self.decoder(img_features)
        density_map = F.interpolate(density_map, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return density_map.squeeze(1)

#############################################
# Custom Collate Function for DataLoader
#############################################
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    density_maps = torch.stack([item[1] for item in batch])
    exemplars = torch.stack([item[2] for item in batch])
    scales = torch.stack([item[3] for item in batch])
    # Now return the annotation coordinates (scaled_locations) for Bayesian loss.
    locations = [item[4] for item in batch]
    return images, density_maps, exemplars, scales, locations

#############################################
# Training & Evaluation Setup
#############################################
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = CACViT(num_exemplars=3, img_size=384, patch_size=16, embed_dim=768).to(device)

    # Dataset paths
    train_img_dir = 'datasets/partA/train_data/images'
    train_gt_dir = 'datasets/partA/train_data/ground_truth'
    exemplars_dir = 'datasets/partA/examplars'
    test_img_dir = 'datasets/partA/test_data/images'
    test_gt_dir = 'datasets/partA/test_data/ground_truth'

    train_dataset = CrowdCountingDataset(
        train_img_dir, train_gt_dir, exemplars_dir,
        transform=transform, exemplar_transform=exemplar_transform, num_exemplars=3, resize_shape=(384, 384)
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)

    test_dataset = CrowdCountingDataset(
        test_img_dir, test_gt_dir, exemplars_dir,
        transform=transform, exemplar_transform=exemplar_transform, num_exemplars=3, resize_shape=(384, 384)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 300
    saved_checkpoints = []
    checkpoint_files = glob.glob("CACVIT_*.pth")

    if checkpoint_files:
        def get_epoch_from_filename(filename):
            try:
                return int(filename.split("_")[1].split(".pth")[0])
            except ValueError:
                return -1
        latest_checkpoint = max(checkpoint_files, key=get_epoch_from_filename)
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint}")
    else:
        start_epoch = 1
        print("No checkpoint found. Starting training from scratch.")
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, gt_density, exemplars, scales, locations_batch) in enumerate(train_loader):
            # Move inputs to device
            images = images.to(device)
            exemplars = exemplars.to(device)
            scales = scales.to(device)  # may not be used if you don't scale
            gt_density = gt_density.to(device)

            optimizer.zero_grad()
            inputs = [images, exemplars, scales]

            # Resize ground truth density to the fixed size (384x384)
            gt_density_resized = F.interpolate(gt_density.unsqueeze(1), size=(384, 384),
                                               mode='bilinear', align_corners=False).squeeze(1)

            # Forward pass: predicted density map of shape (B, 384, 384)
            pred_density = model(inputs)

            # Set up grid for the fixed image size.
            H, W = 384, 384
            # Create coordinate tensors for the grid.
            xs = torch.arange(0, W, device=device).float()
            ys = torch.arange(0, H, device=device).float()
            # Use meshgrid (with explicit indexing) so that grid_y and grid_x have shape (H, W)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            # Expand dims so that we can easily subtract point coordinates later.
            # These grids will have shape (1, H, W)
            grid_x = grid_x.unsqueeze(0)
            grid_y = grid_y.unsqueeze(0)

            sigma = 8.0  # Gaussian sigma (adjust as needed)
            batch_loss = 0.0
            B = pred_density.shape[0]

            # Loop over each image in the batch
            for i in range(B):
                D_pred = pred_density[i]  # shape: (384, 384)
                # print(locations_batch[i])
                pts = torch.tensor(locations_batch[i], dtype=torch.float32, device=device)
                # If no annotations are available, we want the network to predict zero count.
                if pts.shape[0] == 0:
                    loss_i = torch.abs(torch.sum(D_pred) - 0.0)
                else:
                    kernels = []
                    # For each annotation, compute a Gaussian over the entire grid.
                    for j in range(pts.shape[0]):
                        x0, y0 = pts[j]  # scalars
                        # Compute squared distance from this point to all grid locations.
                        dist_sq = (grid_x - x0) ** 2 + (grid_y - y0) ** 2  # shape: (1, H, W)
                        # Compute the Gaussian kernel (unnormalized probability)
                        gauss = torch.exp(-dist_sq / (2 * sigma ** 2))  # shape: (1, H, W)
                        kernels.append(gauss)

                    # Stack kernels so that shape becomes (N, H, W)
                    kernels = torch.cat(kernels, dim=0)
                    # Flatten spatial dimensions: (N, H*W)
                    kernels_flat = kernels.view(pts.shape[0], -1)
                    # For each pixel (over the H*W dimension), compute a softmax across annotations.
                    # This gives a probability p(y_n|x_m) for each annotation.
                    prob = torch.softmax(kernels_flat, dim=0)  # shape: (N, H*W)

                    # Flatten the predicted density map for the image.
                    D_pred_flat = D_pred.view(-1)  # shape: (H*W)
                    # Compute the expected count for each annotation:
                    # E[c_n] = sum_{m} p(y_n|x_m) * D_pred(x_m)
                    expected_counts = torch.sum(prob * D_pred_flat, dim=1)  # shape: (N,)
                    # The target count for each annotation is 1.
                    target_counts = torch.ones_like(expected_counts)
                    # Compute the L1 loss per image over all annotations.
                    loss_i = torch.sum(torch.abs(target_counts - expected_counts))

                batch_loss += loss_i

            # Average the loss over the batch
            bayesian_loss = batch_loss / B

            # Backpropagation and optimizer step
            bayesian_loss.backward()
            optimizer.step()

            total_loss += bayesian_loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {bayesian_loss.item():.6f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.6f}")

        # Evaluation & Visualization on 1 test image.
        model.eval()
        with torch.no_grad():
            sample = next(iter(test_loader))
            test_img, test_gt_density, test_exemplars, test_scales, test_locations = sample
            test_img = test_img.to(device)
            test_gt_density = test_gt_density.to(device)
            test_exemplars = test_exemplars.to(device)
            test_scales = test_scales.to(device)
            output_density = model([test_img, test_exemplars, test_scales])  # (1, 384, 384)
            output_density_np = output_density.squeeze(0).cpu().numpy()
            gt_density_np = test_gt_density.squeeze(0).cpu().numpy()

            pred_count = output_density_np.sum()
            gt_count = len(test_locations[0])  # Number of annotations in the sample.
            mae = abs(gt_count - pred_count)
            print(f"Eval MAE: {mae:.2f}")

            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            disp_img = test_img.squeeze(0).cpu()
            disp_img = inv_normalize(disp_img)
            disp_img = torch.clamp(disp_img, 0, 1)
            disp_img = transforms.ToPILImage()(disp_img)

            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(disp_img)
            plt.title("Input Image")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(gt_density_np, cmap='jet')
            plt.title(f"GT Density Map\nCount: {gt_count}")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(output_density_np, cmap='jet')
            plt.title(f"Predicted Density Map\nCount: {pred_count:.2f}")
            plt.axis('off')
            plt.suptitle(f"Epoch {epoch+1} Evaluation", fontsize=16)
            plt.tight_layout()
            plt.savefig("CACVIT_VIS.png")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_filename = f"CACVIT_{epoch}.pth"
        torch.save(checkpoint_data, checkpoint_filename)
        print(f"Saved checkpoint: {checkpoint_filename}")
        saved_checkpoints.append(checkpoint_filename)
        if len(saved_checkpoints) > 3:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Deleted old checkpoint: {oldest_checkpoint}")

    torch.save(model.state_dict(), 'cacvit_crowd_counting.pth')
    print("Training completed and model saved as 'cacvit_crowd_counting.pth'")
