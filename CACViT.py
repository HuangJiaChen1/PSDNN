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


#############################################
# Dataset Definition
#############################################
class CrowdCountingDataset(Dataset):
    def __init__(self, img_dir, gt_dir, exemplars_dir, transform=None, exemplar_transform=None, num_exemplars=3,
                 resize_shape=(384, 384)):
        """
        Args:
            img_dir: Directory with input images.
            gt_dir: Directory with ground truth .mat files.
            exemplars_dir: Directory with exemplar images.
            transform: Transform for the query images.
            exemplar_transform: Transform for the exemplar images.
            num_exemplars: Number of exemplars to use.
            resize_shape: Target size for query images.
        """
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

        # Load ground truth
        gt_name = 'GT_' + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = loadmat(gt_path)
        locations = mat['image_info']['location']
        locations = [(float(p[0]), float(p[1])) for p in locations]

        # Generate density map (scale head locations to match resized image)
        scale_x = self.resize_shape[1] / img_width
        scale_y = self.resize_shape[0] / img_height
        scaled_locations = [(p[0] * scale_x, p[1] * scale_y) for p in locations]
        density_map = self.generate_density_map(self.resize_shape, scaled_locations)

        # Load exemplars based on a naming pattern (adjust this as needed)
        exemplar_prefix = 'EXAMPLARS_' + img_name.split('_')[1].split('.')[0] + '_'
        exemplar_paths = [
            os.path.join(self.exemplars_dir, f)
            for f in os.listdir(self.exemplars_dir)
            if f.startswith(exemplar_prefix)
        ]
        exemplars = []
        for p in exemplar_paths[:self.num_exemplars]:
            try:
                ex = Image.open(p).convert('RGB')
                exemplars.append(ex)
            except Exception as e:
                print(f"Error loading exemplar {p}: {e}")

        # If missing exemplars, fill in
        if len(exemplars) < self.num_exemplars:
            if len(exemplars) == 0:
                dummy_ex = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                exemplars = [dummy_ex] * self.num_exemplars
            else:
                exemplars = exemplars * (self.num_exemplars // len(exemplars)) + exemplars[
                                                                                 :self.num_exemplars % len(exemplars)]
        elif len(exemplars) > self.num_exemplars:
            exemplars = exemplars[:self.num_exemplars]

        # Scales (assuming exemplar size is 64x64)
        scales = torch.ones((self.num_exemplars, 2), dtype=torch.float32) * (64 / max(img_width, img_height))

        # Apply transforms
        if self.transform:
            image = self.transform(image)  # (C, H, W)
        if self.exemplar_transform:
            exemplars = torch.stack([self.exemplar_transform(ex) for ex in exemplars])  # (num_exemplars, C, H, W)
        density_map = torch.from_numpy(density_map).float()  # (H, W)

        return image, density_map, exemplars, scales, len(locations)

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
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


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

        # Vision Transformer backbone (encoder) from timm
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=0)
        # Use the patch embedding and positional embeddings from the pretrained ViT.

        # Separate PatchEmbed for exemplars (64x64 input)
        self.patch_embed_exemplar = PatchEmbed(img_size=64, patch_size=16, in_chans=3, embed_dim=embed_dim)
        self.num_patches_exemplar = self.patch_embed_exemplar.num_patches
        self.exemplar_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_exemplar, embed_dim))
        nn.init.trunc_normal_(self.exemplar_pos_embed, std=0.02)

        # Positional embeddings for combining image and exemplar features
        self.combined_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_exemplars, embed_dim))
        nn.init.trunc_normal_(self.combined_pos_embed, std=0.02)

        # Scale embedding for exemplars
        self.scale_embed = nn.Linear(2, embed_dim)

        # Decoder to upsample features to a density map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU()  # Ensure non-negative density values
        )

    def forward(self, inputs):
        samples, boxes, scales = inputs  # samples: (B, 3, 384, 384); boxes: (B, num_exemplars, 3, 64, 64)
        batch_size = samples.shape[0]

        # --- Process Query Image through ViT ---
        # Use pretrained ViT patch embedding and positional embeddings.
        x = self.vit.patch_embed(samples)  # (B, num_patches, embed_dim)
        x = x + self.vit.pos_embed[:, 1:, :]  # (B, num_patches, embed_dim)
        cls_token = self.vit.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        cls_token = cls_token + self.vit.pos_embed[:, :1, :]
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches+1, embed_dim)
        for blk in self.vit.blocks:
            x = blk(x)
        img_features = self.vit.norm(x)  # (B, num_patches+1, embed_dim)
        img_features = img_features[:, 1:, :]  # Remove CLS token --> (B, num_patches, embed_dim)

        # --- Process Exemplars ---
        exemplars = boxes.view(batch_size * self.num_exemplars, 3, 64, 64)
        exemplar_features = self.patch_embed_exemplar(exemplars)  # (B*num_exemplars, num_patches_ex, embed_dim)
        exemplar_features = exemplar_features + self.exemplar_pos_embed
        exemplar_features = exemplar_features.view(batch_size, self.num_exemplars, self.num_patches_exemplar,
                                                   self.embed_dim)
        exemplar_features = exemplar_features.mean(dim=2)  # Average over patches: (B, num_exemplars, embed_dim)
        scale_features = self.scale_embed(scales)  # (B, num_exemplars, embed_dim)
        exemplar_features = exemplar_features + scale_features  # (B, num_exemplars, embed_dim)

        # --- Combine Image and Exemplar Features ---
        combined_features = torch.cat([img_features, exemplar_features],
                                      dim=1)  # (B, num_patches+num_exemplars, embed_dim)
        combined_features = combined_features + self.combined_pos_embed
        combined_features = self.vit.norm(combined_features)

        # Extract only the image features for density prediction and reshape
        img_features = combined_features[:, :self.num_patches, :]  # (B, num_patches, embed_dim)
        h = w = self.img_size // self.patch_size  # For 384 and 16 -> 24
        img_features = img_features.permute(0, 2, 1).reshape(batch_size, self.embed_dim, h, w)

        # --- Decode to Density Map ---
        density_map = self.decoder(img_features)  # (B, 1, H_dec, W_dec)
        density_map = F.interpolate(density_map, size=(self.img_size, self.img_size), mode='bilinear',
                                    align_corners=False)
        return density_map.squeeze(1)  # (B, H, W)


#############################################
# Custom Collate Function for DataLoader
#############################################
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    density_maps = torch.stack([item[1] for item in batch])
    exemplars = torch.stack([item[2] for item in batch])  # (B, num_exemplars, C, H, W)
    scales = torch.stack([item[3] for item in batch])
    return images, density_maps, exemplars, scales


#############################################
# Training & Evaluation Setup
#############################################
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = CACViT(num_exemplars=3, img_size=384, patch_size=16, embed_dim=768).to(device)

    # Dataset paths (adjust these paths as necessary)
    train_img_dir = 'datasets/partA/train_data/images'
    train_gt_dir = 'datasets/partA/train_data/ground_truth'
    exemplars_dir = 'datasets/partA/examplars'  # Ensure exemplars exist here
    test_img_dir = 'datasets/partA/test_data/images'
    test_gt_dir = 'datasets/partA/test_data/ground_truth'

    # Create DataLoaders for training and evaluation
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

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 300
    saved_checkpoints = []
    # Find all checkpoint files matching the pattern
    checkpoint_files = glob.glob("CACVIT_*.pth")

    if checkpoint_files:
        def get_epoch_from_filename(filename):
            try:
                return int(filename.split("_")[1].split(".pth")[0])
            except ValueError:
                return -1


        # Get the checkpoint with the highest epoch number
        latest_checkpoint = max(checkpoint_files, key=get_epoch_from_filename)
        checkpoint = torch.load(latest_checkpoint)

        # If your checkpoint is a dictionary containing the state dicts and epoch info:
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
        for batch_idx, (images, gt_density, exemplars, scales) in enumerate(train_loader):
            images = images.to(device)
            gt_density = gt_density.to(device)
            exemplars = exemplars.to(device)
            scales = scales.to(device)

            optimizer.zero_grad()
            inputs = [images, exemplars, scales]
            pred_density = model(inputs)  # (B, 384, 384)

            # Resize ground truth density map if needed
            gt_density_resized = F.interpolate(gt_density.unsqueeze(1), size=(384, 384), mode='bilinear',
                                               align_corners=False).squeeze(1)
            loss = criterion(pred_density, gt_density_resized)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.6f}")
        if epoch % 5 == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_filename = f"CACVIT_{epoch}.pth"
            torch.save(checkpoint_data, checkpoint_filename)
            print(f"Saved checkpoint: {checkpoint_filename}")
            saved_checkpoints.append(checkpoint_filename)

            # If more than 3 checkpoints exist, delete the oldest one
            if len(saved_checkpoints) > 3:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Deleted old checkpoint: {oldest_checkpoint}")

        # --- Evaluation & Visualization on 1 Test Image ---
        model.eval()
        with torch.no_grad():
            sample = next(iter(test_loader))
            test_img, test_gt_density, test_exemplars, test_scales, gt_count = sample
            test_img = test_img.to(device)
            test_exemplars = test_exemplars.to(device)
            test_scales = test_scales.to(device)
            output_density = model([test_img, test_exemplars, test_scales])  # (1, 384, 384)
            output_density_np = output_density.squeeze(0).cpu().numpy()
            gt_density_np = test_gt_density.squeeze(0).cpu().numpy()

            pred_count = output_density_np.sum()

            # Convert normalized image back to PIL image for display (reverse normalization)
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            disp_img = test_img.squeeze(0).cpu()
            disp_img = inv_normalize(disp_img)
            disp_img = torch.clamp(disp_img, 0, 1)
            disp_img = transforms.ToPILImage()(disp_img)

            # Plotting
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(disp_img)
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(gt_density_np, cmap='jet')
            plt.title(f"GT Density Map\nCount: {gt_count:.2f}")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(output_density_np, cmap='jet')
            plt.title(f"Predicted Density Map\nCount: {pred_count:.2f}")
            plt.axis('off')

            plt.suptitle(f"Epoch {epoch + 1} Evaluation", fontsize=16)
            plt.tight_layout()
            plt.savefig("CACVIT_VIS.png")

    # Save the final model
    torch.save(model.state_dict(), 'cacvit_crowd_counting.pth')
    print("Training completed and model saved as 'cacvit_crowd_counting.pth'")