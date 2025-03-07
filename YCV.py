import glob
import os
from functools import partial

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mat4py import loadmat
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from einops import rearrange,repeat
from timm.models.vision_transformer import PatchEmbed
from Blocks import Block
from pos_embed import get_2d_sincos_pos_embed

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
        density_map = torch.from_numpy(density_map).float()*60  # (H, W)

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




class CACVIT(nn.Module):
    """ CntVit with VisionTransformer backbone
    """

    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate=0):
        super().__init__()
        ## Setting the model
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        ## Global Setting
        self.patch_size = patch_size
        self.img_size = img_size
        ex_size = 64
        self.norm_pix_loss = norm_pix_loss
        ## Global Setting

        ## Encoder specifics
        self.scale_embeds = nn.Linear(2, embed_dim, bias=True)
        self.patch_embed_exemplar = PatchEmbed(ex_size, patch_size, in_chans + 1, embed_dim)
        num_patches_exemplar = self.patch_embed_exemplar.num_patches
        self.pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, embed_dim), requires_grad=False)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.norm = norm_layer(embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.v_y = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.density_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, decoder_embed_dim),
                                                       requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_norm = norm_layer(decoder_embed_dim)
        ### decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        ### decoder blocks
        ## Decoder specifics
        ## Regressor
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(513, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )
        ## Regressor

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1],
                                                     int(self.patch_embed_exemplar.num_patches ** .5), cls_token=False)
        self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_exemplar = get_2d_sincos_pos_embed(self.decoder_pos_embed_exemplar.shape[-1],
                                                             int(self.patch_embed_exemplar.num_patches ** .5),
                                                             cls_token=False)
        self.decoder_pos_embed_exemplar.data.copy_(torch.from_numpy(decoder_pos_embed_exemplar).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w1 = self.patch_embed_exemplar.proj.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def scale_embedding(self, exemplars, scale_infos):
        method = 1
        if method == 0:
            bs, n, c, h, w = exemplars.shape
            scales_batch = []
            for i in range(bs):
                scales = []
                for j in range(n):
                    w_scale = torch.linspace(0, scale_infos[i, j, 0], w)
                    w_scale = repeat(w_scale, 'w->h w', h=h).unsqueeze(0)
                    h_scale = torch.linspace(0, scale_infos[i, j, 1], h)
                    h_scale = repeat(h_scale, 'h->h w', w=w).unsqueeze(0)
                    scale = torch.cat((w_scale, h_scale), dim=0)
                    scales.append(scale)
                scales = torch.stack(scales)
                scales_batch.append(scales)
            scales_batch = torch.stack(scales_batch)

        if method == 1:
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

    def forward_encoder(self, x, y, scales=None):
        y_embed = []
        y = rearrange(y, 'b n c w h->n b c w h')
        for box in y:
            box = self.patch_embed_exemplar(box)
            box = box + self.pos_embed_exemplar
            y_embed.append(box)
        y_embed = torch.stack(y_embed, dim=0)
        box_num, _, n, d = y_embed.shape
        y = rearrange(y_embed, 'box_num batch n d->batch (box_num  n) d')
        x = self.patch_embed(x)
        x = x + self.pos_embed
        _, l, d = x.shape
        attns = []
        x_y = torch.cat((x, y), axis=1)
        for i, blk in enumerate(self.blocks):
            x_y, attn = blk(x_y)
            attns.append(attn)
        x_y = self.norm(x_y)
        x = x_y[:, :l, :]
        for i in range(box_num):
            y[:, i * n:(i + 1) * n, :] = x_y[:, l + i * n:l + (i + 1) * n, :]
        y = rearrange(y, 'batch  (box_num  n) d->box_num batch n d', box_num=box_num, n=n)
        return x, y

    def forward_decoder(self, x, y, scales=None):
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed
        b, l_x, d = x.shape
        y_embeds = []
        num, batch, l, dim = y.shape
        for i in range(num):
            y_embed = self.decoder_embed(y[i])
            y_embed = y_embed + self.decoder_pos_embed_exemplar
            y_embeds.append(y_embed)
        y_embeds = torch.stack(y_embeds)
        num, batch, l, dim = y_embeds.shape
        y_embeds = rearrange(y_embeds, 'n b l d -> b (n l) d')
        x = torch.cat((x, y_embeds), axis=1)
        attns = []
        xs = []
        ys = []
        for i, blk in enumerate(self.decoder_blocks):
            x, attn = blk(x)
            if i == 2:
                x = self.decoder_norm(x)
            attns.append(attn)
            xs.append(x[:, :l_x, :])
            ys.append(x[:, l_x:, :])
        return xs, ys, attns

    def AttentionEnhance(self, attns, l=24, n=1):
        l_x = int(l * l)
        l_y = int(4 * 4)
        r = self.img_size // self.patch_size
        attns = torch.mean(attns, dim=1)

        attns_x2y = attns[:, l_x:, :l_x]
        attns_x2y = rearrange(attns_x2y, 'b (n ly) l->b n ly l', ly=l_y)
        attns_x2y = attns_x2y * n.unsqueeze(-1).unsqueeze(-1)
        attns_x2y = attns_x2y.sum(2)

        attns_x2y = torch.mean(attns_x2y, dim=1).unsqueeze(-1)
        attns_x2y = rearrange(attns_x2y, 'b (w h) c->b c w h', w=r, h=r)
        return attns_x2y

    def MacherMode(self, xs, ys, attn, scales=None, name='0.jpg'):
        x = xs[-1]
        B, L, D = x.shape
        y = ys[-1]
        B, Ly, D = y.shape
        n = int(Ly / 16)
        r2 = (scales[:, :, 0] + scales[:, :, 1]) ** 2
        n = 16 / (r2 * 384)
        density_feature = rearrange(x, 'b (w h) d->b d w h', w=24)
        density_enhance = self.AttentionEnhance(attn[-1], l=int(np.sqrt(L)), n=n)
        density_feature2 = torch.cat((density_feature.contiguous(), density_enhance.contiguous()), axis=1)

        return density_feature2

    def Regressor(self, feature):
        feature = F.interpolate(
            self.decode_head0(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
            self.decode_head1(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
            self.decode_head2(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
            self.decode_head3(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = feature.squeeze(-3)
        return feature

    def forward(self, samples, name=None):
        imgs = samples[0]
        boxes = samples[1]
        scales = samples[2]
        boxes = self.scale_embedding(boxes, scales)
        latent, y_latent = self.forward_encoder(imgs, boxes, scales=scales)
        xs, ys, attns = self.forward_decoder(latent, y_latent)
        density_feature = self.MacherMode(xs, ys, attns, scales, name=None)
        density_map = self.Regressor(density_feature)

        return density_map


#############################################
# Custom Collate Function for DataLoader
#############################################
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    density_maps = torch.stack([item[1] for item in batch])
    exemplars = torch.stack([item[2] for item in batch])
    scales = torch.stack([item[3] for item in batch])
    gt_counts = [item[4] for item in batch]  # This returns a list of counts
    return images, density_maps, exemplars, scales, gt_counts



#############################################
# Training & Evaluation Setup
#############################################
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = CACVIT(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = model.to(device)
    # Dataset paths (adjust these paths as necessary)
    train_img_dir = 'datasets/partB/train_data/images'
    train_gt_dir = 'datasets/partB/train_data/ground_truth'
    exemplars_dir = 'datasets/partA/examplars'  # Ensure exemplars exist here
    test_img_dir = 'datasets/partB/test_data/images'
    test_gt_dir = 'datasets/partB/test_data/ground_truth'

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
    checkpoint_files = glob.glob("YCV*.pth")

    if checkpoint_files:
        def get_epoch_from_filename(filename):
            try:
                return int(filename[1].split(".pth")[0])
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
    model.load_state_dict(torch.load("best-model.pth")['model'])
    # optimizer.load_state_dict(torch.load("best_model.pth")[['optimizer']])
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, gt_density, exemplars, scales,gt_count) in enumerate(train_loader):
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
            pred_count = pred_density.cpu().detach().numpy().sum()/60

            loss = (pred_density - gt_density_resized) ** 2
            loss = loss.mean()

            # mape_loss = abs(gt_count - pred_count)/gt_count
            # loss = (pred_density - gt_density_resized) ** 2
            # loss = loss.mean()  # Ensure loss is a scalar if needed
            # loss += 0.2 * mape_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.6f}")
        if epoch % 1 == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_filename = f"YCV{epoch}.pth"
            torch.save(checkpoint_data, checkpoint_filename)
            print(f"Saved checkpoint: {checkpoint_filename}")
            saved_checkpoints.append(checkpoint_filename)

            # If more than 3 checkpoints exist, delete the oldest one
            if len(saved_checkpoints) > 1 :
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
                test_gt_density = test_gt_density.to(device)
                test_exemplars = test_exemplars.to(device)
                test_scales = test_scales.to(device)
                output_density = model([test_img, test_exemplars, test_scales])  # (1, 384, 384)
                output_density_np = output_density.squeeze(0).cpu().numpy()
                gt_density_np = test_gt_density.squeeze(0).cpu().numpy()

                pred_count = output_density_np.sum()/60
                gt_density_resized = F.interpolate(test_gt_density.unsqueeze(1), size=(384, 384), mode='bilinear',
                                                   align_corners=False).squeeze(1)
                loss = criterion(gt_density_resized, output_density)
                print(f"Eval loss: {loss}")
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

                if isinstance(gt_count, list):
                    gt_count = float(gt_count[0])

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
                plt.savefig("YCV_VIS.png")

    # Save the final model
    torch.save(model.state_dict(), 'YCV.pth')
    print("Training completed and model saved as 'YCV.pth'")