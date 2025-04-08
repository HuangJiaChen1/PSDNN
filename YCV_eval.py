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
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from Blocks import Block
from pos_embed import get_2d_sincos_pos_embed
from geomloss import SamplesLoss
import torch.nn.functional as F

#############################################
# Dataset Definition
#############################################


class CrowdCountingLoss(nn.Module):
    def __init__(self, alpha=1, sinkhorn_blur=0.2, density_scale=10):  # Increased blur
        super().__init__()
        self.alpha = alpha
        self.density_scale = density_scale
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            backend="multiscale",
            blur=sinkhorn_blur,
            scaling=0.9,
            reach=0.1
        )

    def forward(self, pred_map, gt_map, gt_blur_map):
        # Add density map reconstruction loss
        density_loss = F.mse_loss(pred_map, gt_blur_map)

        # Scale counts appropriately
        pred_count = pred_map.sum(dim=[1, 2, 3])/360
        gt_count = gt_map.sum(dim=[1, 2, 3])/360

        count_loss = F.l1_loss(pred_count, gt_count)

        pred_map = pred_map.squeeze(1)
        gt_map = gt_map.squeeze(1)

        density_loss = F.mse_loss(pred_map, gt_map)
        spatial_loss = torch.mean(self.sinkhorn(pred_map, gt_map))

        return density_loss + count_loss + self.alpha * spatial_loss

#############################################
# Modified Training Dataset
#############################################
class CrowdCountingDataset(Dataset):
    def __init__(self, img_dir, gt_dir, exemplars_dir, transform=None, exemplar_transform=None,
                 num_exemplars=3, resize_shape=(1152, 768), crop_size=(384, 384)):
        """
        Args:
            img_dir: Directory with input images.
            gt_dir: Directory with ground truth .mat files.
            exemplars_dir: Directory with exemplar images.
            transform: Transform for the query images.
            exemplar_transform: Transform for the exemplar images.
            num_exemplars: Number of exemplars to use.
            resize_shape: Size to which the original image is resized (width, height), e.g. (1152,768).
            crop_size: Size of each crop, e.g. (384,384).
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.exemplars_dir = exemplars_dir
        self.transform = transform
        self.exemplar_transform = exemplar_transform
        self.num_exemplars = num_exemplars
        self.resize_shape = resize_shape  # (width, height)
        self.crop_size = crop_size        # (crop_width, crop_height)

        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

        # Build an index mapping: each image yields multiple crops.
        # For a resized image of size (1152,768) and crop (384,384):
        # Number of columns = 1152 // 384 = 3, rows = 768 // 384 = 2, total = 6 crops.
        self.cols = self.resize_shape[0] // self.crop_size[0]
        self.rows = self.resize_shape[1] // self.crop_size[1]
        self.samples_per_image = self.rows * self.cols

        # Build a list of (img_index, crop_row, crop_col)
        self.index_map = []
        for i in range(len(self.img_names)):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.index_map.append((i, r, c))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # Get mapping: which image and which crop.
        img_idx, crop_r, crop_c = self.index_map[idx]
        img_name = self.img_names[img_idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Resize image to fixed size (1152,768)
        new_w, new_h = self.resize_shape
        image_resized = image.resize((new_w, new_h), Image.BILINEAR)

        # Compute scaling factors for annotations
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        # Load ground truth annotations and scale them.
        gt_name = 'GT_' + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = loadmat(gt_path)
        locations = mat['image_info']['location']
        # Scale annotations to the resized image
        try:
            scaled_locations = [(float(p[0]) * scale_x, float(p[1]) * scale_y) for p in locations]
        except Exception as e:
            print(e)
            print(locations)
            print(img_path)
            print(gt_path)


        # Compute full density map for the resized image (optional: you could generate a density map per crop instead)
        full_density = self.generate_density_map((new_h, new_w), scaled_locations)
        full_density = full_density.astype(np.float32)  # as before

        # Crop the image: crop_size (384,384)
        crop_w, crop_h = self.crop_size
        left = crop_c * crop_w
        top = crop_r * crop_h
        image_crop = image_resized.crop((left, top, left + crop_w, top + crop_h))
        # Crop density map accordingly
        density_crop = full_density[top:top+crop_h, left:left+crop_w]

        # Adjust annotation coordinates to the crop:
        # Keep only points within the crop, and subtract the crop offset.
        crop_points = []
        for (x, y) in scaled_locations:
            if left <= x < left+crop_w and top <= y < top+crop_h:
                crop_points.append((x - left, y - top))
        # Optionally, if no points fall inside, you can still return an empty list.
        # For Bayesian loss, an empty list might be handled specially.

        # Load exemplars (they remain unchanged)
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

        # Scales for exemplars (using the resized image size)
        scales = torch.ones((self.num_exemplars, 2), dtype=torch.float32) * (64 / max(new_w, new_h))

        # Apply transforms
        if self.transform:
            image_crop = self.transform(image_crop)  # (C, 384, 384)
        if self.exemplar_transform:
            exemplars = torch.stack([self.exemplar_transform(ex) for ex in exemplars])
        density_crop = torch.from_numpy(density_crop).float()*360

        return image_crop, density_crop, exemplars, scales, len(crop_points), scaled_locations

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
    scaled_locations = [item[5] for item in batch]
    return images, density_maps, exemplars, scales, gt_counts, scaled_locations


def sliding_window_inference(model, image, exemplars, scales, window_size=(384, 384), stride=128, device='cpu'):
    """
    Args:
        model: Trained model.
        image: Full image as a tensor of shape (C,H,W) (assumed to be normalized as in training).
        exemplars: Exemplar images tensor.
        scales: Scale info tensor.
        window_size: Tuple (width, height).
        stride: Sliding stride.
        device: device to run inference.
    Returns:
        Combined density map for the entire image.
    """
    model.eval()
    C, H, W = image.shape
    w_win, h_win = window_size
    # Create an empty accumulation tensor and a counter tensor for averaging overlaps.
    density_sum = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    # Slide over the image
    for top in range(0, H - h_win + 1, stride):
        for left in range(0, W - w_win + 1, stride):
            crop = image[:, top:top + h_win, left:left + w_win].unsqueeze(0)  # (1,C,h_win,w_win)
            # Run model on the crop. Make sure exemplars and scales are on device.
            with torch.no_grad():
                pred_density = model([crop.to(device), exemplars.to(device), scales.to(device)])
                # pred_density is (1, h_win, w_win)
            pred_density = pred_density.squeeze(0)
            # Accumulate predictions (for overlapping regions, add the predictions)
            density_sum[top:top + h_win, left:left + w_win] += pred_density
            count_map[top:top + h_win, left:left + w_win] += 1

    # Avoid division by zero
    count_map[count_map == 0] = 1
    combined_density = density_sum / count_map
    return combined_density.cpu()


#############################################
# Loss
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrowdCountingLoss(nn.Module):
    def __init__(self, alpha=0.00000036, beta=0.00008, sinkhorn_blur=0.2, density_scale=10, epsilon=0.01, sinkhorn_iters=20):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.density_scale = density_scale
        self.epsilon = epsilon  # Entropic regularization for Sinkhorn
        self.sinkhorn_iters = sinkhorn_iters  # Number of Sinkhorn iterations

    def sinkhorn_loss(self, pred_map, gt_map):
        """ Computes the Sinkhorn loss between predicted and ground-truth density maps """
        # Flatten maps
        a = pred_map.view(-1) + 1e-8  # Avoid zero division
        b = gt_map.view(-1) + 1e-8

        # Compute cost matrix (Squared Euclidean distance)
        coords_pred = torch.nonzero(pred_map, as_tuple=False).float()
        coords_gt = torch.nonzero(gt_map, as_tuple=False).float()

        if coords_pred.shape[0] == 0 or coords_gt.shape[0] == 0:
            return torch.tensor(0.0, device=pred_map.device)  # Return zero if no valid density

        num_pred, num_gt = coords_pred.shape[0], coords_gt.shape[0]

        cost_matrix = torch.cdist(coords_pred, coords_gt).pow(2)  # Shape: [num_pred, num_gt]

        # Initialize Sinkhorn dual potentials with correct shapes
        u = torch.zeros(num_pred, device=pred_map.device)  # Shape: [num_pred]
        v = torch.zeros(num_gt, device=pred_map.device)  # Shape: [num_gt]

        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iters):
            u = -self.epsilon * torch.logsumexp((v[None, :] - cost_matrix) / self.epsilon, dim=1)  # Fix: dim=1
            v = -self.epsilon * torch.logsumexp((u[:, None] - cost_matrix) / self.epsilon, dim=0)  # Fix: dim=0

        # Compute transport cost
        transport_cost = torch.sum(torch.exp((u[:, None] + v[None, :] - cost_matrix) / self.epsilon) * cost_matrix)
        return transport_cost


    def forward(self, pred_map, gt_map, gt_blur_map, pred_count, gt_count):
        pred_count = torch.tensor(pred_count, dtype=torch.float32, device=pred_map.device).view(1)
        gt_count = torch.tensor(gt_count, dtype=torch.float32, device=pred_map.device).view(1)
        gt_map = torch.tensor(gt_map, dtype=torch.float32, device=pred_map.device)

        # Density map reconstruction loss
        density_loss = F.mse_loss(pred_map, gt_blur_map)

        # Count loss
        count_loss = F.l1_loss(pred_count, gt_count)

        # Compute Sinkhorn loss
        sinkhorn_loss = self.sinkhorn_loss(pred_map, gt_map)

        # Final loss
        return self.beta*density_loss + self.beta*count_loss + self.beta*self.alpha * sinkhorn_loss



import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from functools import partial
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from YCV import CACVIT  # Assuming your model class is in YCV.py

def evaluate_model(model, test_loader, device):
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for batch_idx, (test_img, test_gt_density, test_exemplars, test_scales, gt_count, gt_map) in enumerate(test_loader):
            test_img = test_img.to(device)
            test_gt_density = test_gt_density.to(device)
            test_exemplars = test_exemplars.to(device)
            test_scales = test_scales.to(device)

            output_density = model([test_img, test_exemplars, test_scales])
            pred_count = output_density.cpu().detach().numpy().sum() / 360  # Adjust scale factor if needed

            mae = abs(pred_count - gt_count.item())
            total_mae += mae

            print(f"Image {batch_idx + 1}: Predicted Count = {pred_count:.2f}, GT Count = {gt_count.item()}, MAE = {mae:.2f}")
    
    avg_mae = total_mae / len(test_loader)
    print(f"Mean Absolute Error (MAE) over {len(test_loader)} images: {avg_mae:.2f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    model = CACVIT(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                   decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                   mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    checkpoint = torch.load("YCV93.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    test_img_dir = 'datasets/partB/test_data/images'
    test_gt_dir = 'datasets/partB/test_data/ground_truth'
    
    test_dataset = CrowdCountingDataset(
        test_img_dir, test_gt_dir, None,
        transform=transform, exemplar_transform=exemplar_transform,
        num_exemplars=3, resize_shape=(576, 384), crop_size=(576, 384)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
