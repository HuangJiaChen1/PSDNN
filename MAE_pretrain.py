from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import glob
# timm modules for Vision Transformer components
from timm_check.models.vision_transformer import PatchEmbed, Block


# -----------------------------------------------------------------------------
# Helper: 2D sin-cos positional embeddings
# -----------------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: total dimension (should be divisible by 4)
    grid_size: int, assuming square grid (grid_size x grid_size patches)
    Return:
        pos_embed: (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0, "Embed dim must be divisible by 4"
    dim_each = embed_dim // 4  # for each sin/cos per coordinate

    # Create grid of coordinates
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # grid[0] and grid[1] are each (grid_size, grid_size)
    grid = np.stack(grid, axis=-1)  # (grid_size, grid_size, 2)
    grid = grid.reshape(-1, 2)  # (L, 2) where L = grid_size*grid_size

    pos_embed_list = []
    for i in range(2):
        # For each coordinate dimension
        pos_i = grid[:, i:i + 1]  # (L, 1)
        omega = 1.0 / (10000 ** (np.arange(dim_each, dtype=np.float32) / dim_each))
        pos_i = pos_i * omega[None, :]  # (L, dim_each)
        sin_embed = np.sin(pos_i)  # (L, dim_each)
        cos_embed = np.cos(pos_i)  # (L, dim_each)
        pos_embed_list.append(np.concatenate([sin_embed, cos_embed], axis=1))  # (L, 2*dim_each)
    pos_embed = np.concatenate(pos_embed_list, axis=1)  # (L, 4*dim_each) = (L, embed_dim)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# -----------------------------------------------------------------------------
# Dataset for Pretraining
# -----------------------------------------------------------------------------
class PretrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_list = [os.path.join(root, f) for f in os.listdir(root)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


# -----------------------------------------------------------------------------
# MAE Model following the official design
# -----------------------------------------------------------------------------
class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

        # ----- Encoder -----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed_np = get_2d_sincos_pos_embed(embed_dim, int(np.sqrt(num_patches)), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float())

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ----- Decoder -----
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        decoder_pos_embed_np = get_2d_sincos_pos_embed(decoder_embed_dim, int(np.sqrt(num_patches)), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed_np).float())

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        Returns:
            x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_embed.patch_size[0]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        Returns:
            imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)  # (N, L, embed_dim)
        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)  # (N, L_keep, decoder_embed_dim)
        N, L_keep, D = x.shape
        L = self.patch_embed.num_patches
        len_mask = L - L_keep
        mask_tokens = self.mask_token.repeat(N, len_mask, 1)
        x_combined = torch.cat([x, mask_tokens], dim=1)
        ids_restore_unsq = ids_restore.unsqueeze(-1).repeat(1, 1, D)
        x_full = torch.zeros(N, L, D, device=x.device)
        x_full.scatter_(1, ids_restore_unsq, x_combined)
        x_full = x_full + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        x_full = self.decoder_pred(x_full)  # (N, L, patch_dim)
        return x_full

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # loss per patch
        N, L = mask.shape
        loss = loss.sum() / (N * L)
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# -----------------------------------------------------------------------------
# Debug Functions: Verify patchify/unpatchify and variance comparison
# -----------------------------------------------------------------------------
def debug_preprocessing(model, sample_img):
    """
    Verifies that patchify and unpatchify work correctly,
    and compares the variance of the original image vs. the reconstruction.
    """
    with torch.no_grad():
        # Patchify and then unpatchify the sample image
        patches = model.patchify(sample_img)
        unpatched_img = model.unpatchify(patches)

    # Compute Mean Squared Error between original and reconstructed
    mse = ((sample_img - unpatched_img) ** 2).mean().item()
    print(f"Patchify/Unpatchify MSE: {mse:.6f}")

    # Compute variance for each image (over all pixels and channels)
    orig_var = sample_img.var().item()
    recon_var = unpatched_img.var().item()
    print(f"Original image variance: {orig_var:.6f}")
    print(f"Reconstructed image (via patchify/unpatchify) variance: {recon_var:.6f}")

    # Visualize the original and unpatched image
    sample_np = sample_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
    unpatch_np = unpatched_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(unpatch_np)
    plt.title("After Patchify & Unpatchify")
    plt.axis("off")

    plt.show()


def debug_prediction_variance(model, sample_img, mask_ratio=0.75):
    """
    Runs the model on a sample image and compares the variance
    of the predicted patches (after unpatchifying) to the original image.
    """
    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(sample_img, mask_ratio=mask_ratio)
        recon_img = model.unpatchify(pred)

    orig_var = sample_img.var().item()
    recon_var = recon_img.var().item()
    print(f"Original image variance: {orig_var:.6f}")
    print(f"Reconstructed image variance: {recon_var:.6f}")

    return recon_img


# -----------------------------------------------------------------------------
# Training Loop with Verbose Logging and Visualization
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    dataset_root = "datasets/partA/train_data/images"  # update as needed
    dataset = PretrainDataset(root=dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    mae_model = MaskedAutoencoderViT(
        img_size=384,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False
    ).to(device)

    optimizer = optim.AdamW(mae_model.parameters(), lr=5e-6)
    num_epochs = 300

    saved_checkpoints = []

    mae_model.train()



    # Search for checkpoint files matching the pattern
    checkpoint_files = glob.glob("mae_pretrain_epoch*.pth")

    if checkpoint_files:
        # Extract epoch numbers and get the checkpoint with the highest epoch number
        def get_epoch_from_filename(filename):
            try:
                return int(filename.split("mae_pretrain_epoch")[1].split(".pth")[0])
            except ValueError:
                return -1

        latest_checkpoint = max(checkpoint_files, key=get_epoch_from_filename)
        checkpoint = torch.load(latest_checkpoint)
        mae_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint}")
    else:
        start_epoch = 1
        print("No checkpoint found. Starting training from scratch.")

    # Continue training from the determined start epoch
    for epoch in range(start_epoch, num_epochs + 1):
        mae_model.train()
        epoch_loss = 0.0
        for batch_idx, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            loss, pred, mask = mae_model(imgs, mask_ratio=0.75)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.6f}")
        epoch_loss /= len(dataset)
        print(f"Epoch [{epoch}/{num_epochs}] completed. Average Loss: {epoch_loss:.6f}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': mae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss  # optional
            }
            checkpoint_filename = f"mae_pretrain_epoch{epoch}.pth"
            torch.save(checkpoint_data, checkpoint_filename)
            print(f"Saved checkpoint: {checkpoint_filename}")
            saved_checkpoints.append(checkpoint_filename)

            # If more than 3 checkpoints exist, delete the oldest one
            if len(saved_checkpoints) > 3:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Deleted old checkpoint: {oldest_checkpoint}")

            # -----------------------------------------------------------------------------
            # Debug: Verify patchify/unpatchify on a sample image
            # -----------------------------------------------------------------------------
            # If you have trained weights, uncomment and update the path:
            mae_model.eval()
            sample_img = dataset[2].unsqueeze(0).to(device)  # (1,3,384,384)
            # debug_preprocessing(mae_model, sample_img)
            # debug_prediction_variance(mae_model,sample_img,0.5)
            # -----------------------------------------------------------------------------
            # Visualization: Plot one sample's original, masked, and reconstructed images
            # Also compare the variance of the original vs. the reconstruction.
            # -----------------------------------------------------------------------------
            with torch.no_grad():
                # Obtain the patch embeddings and perform random masking for visualization
                x = mae_model.patch_embed(sample_img) + mae_model.pos_embed  # (1, L, embed_dim)
                _, mask, ids_restore = mae_model.random_masking(x, 0.75)
                N, L, D = x.shape
                grid_size = int(np.sqrt(L))
                mask_img = mask.reshape(1, grid_size, grid_size)
                mask_img = torch.nn.functional.interpolate(mask_img.unsqueeze(1).float(), size=(384, 384), mode='nearest')
                # Create masked image (set masked patches to gray value 0.5)
                masked_img = sample_img * (1 - mask_img) + 0.5 * mask_img

                # Full forward pass to get reconstruction
                loss_val, pred, mask_val = mae_model(sample_img, mask_ratio=0.75)
                recon_img = mae_model.unpatchify(pred)
                recon_img = recon_img.float()
                recon_img = torch.einsum('nchw->nhwc', recon_img)
                recon_img = torch.clamp(recon_img, 0, 1)


            # Convert tensors to numpy arrays for plotting
            orig_np = sample_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            masked_np = masked_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            recon_np = recon_img.squeeze(0).cpu().numpy()

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(orig_np)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(masked_np)
            plt.title("Masked Image")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(recon_np)
            plt.title("Reconstructed Image")
            plt.axis("off")
            plt.savefig('MAE_vis.png')

    torch.save(mae_model.state_dict(), "mae_pretrained.pth")
    print("MAE Pretraining complete. Model weights saved to mae_pretrained.pth")



if __name__ == "__main__":
    main()
