#!/usr/bin/env python
import os
import math
import random
import numpy as np
from PIL import Image
import argparse
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models

from mat4py import loadmat  # pip install mat4py

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================
# Helper Functions (GPU version)
# ============================
def compute_euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_pseudo_gt_from_points(points):
    pseudo_boxes = []
    default_size = 32.0
    for i, pt in enumerate(points):
        if len(points) > 1:
            dists = [compute_euclidean_distance(pt, q) for j, q in enumerate(points) if i != j]
            d = min(dists)
        else:
            d = default_size
        half = d / 2.0
        box = [pt[0] - half, pt[1] - half, pt[0] + half, pt[1] + half]
        pseudo_boxes.append(box)
    return np.array(pseudo_boxes, dtype=np.float32)


def generate_anchors(feature_size, stride, scales, ratios):
    anchors = []
    H, W = feature_size
    for i in range(H):
        for j in range(W):
            center_x = j * stride + stride / 2.0
            center_y = i * stride + stride / 2.0
            for scale in scales:
                for ratio in ratios:
                    w = scale * math.sqrt(ratio)
                    h = scale / math.sqrt(ratio)
                    x1 = center_x - w / 2.0
                    y1 = center_y - h / 2.0
                    x2 = center_x + w / 2.0
                    y2 = center_y + h / 2.0
                    anchors.append([x1, y1, x2, y2])
    return np.array(anchors, dtype=np.float32)


def compute_iou_tensor(anchors, gt_boxes):
    # anchors: (N,4), gt_boxes: (M,4)
    N = anchors.shape[0]
    M = gt_boxes.shape[0]
    x1 = torch.max(anchors[:, None, 0], gt_boxes[None, :, 0])
    y1 = torch.max(anchors[:, None, 1], gt_boxes[None, :, 1])
    x2 = torch.min(anchors[:, None, 2], gt_boxes[None, :, 2])
    y2 = torch.min(anchors[:, None, 3], gt_boxes[None, :, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_anchor = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_anchor[:, None] + area_gt[None, :] - inter + 1e-6
    iou = inter / union
    return iou


def assign_anchors_to_gt_torch(anchors, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.3):
    iou_matrix = compute_iou_tensor(anchors, gt_boxes)  # (N,M)
    max_iou, argmax_iou = torch.max(iou_matrix, dim=1)
    labels = torch.full((anchors.shape[0],), -1, dtype=torch.int64, device=anchors.device)
    labels[max_iou < neg_iou_threshold] = 0
    labels[max_iou >= pos_iou_threshold] = 1
    assigned_gt = gt_boxes[argmax_iou]
    return labels, assigned_gt


def vectorized_bbox_transform(anchors, gt_boxes):
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_w
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - anchor_ctr_x) / anchor_w
    dy = (gt_ctr_y - anchor_ctr_y) / anchor_h
    dw = torch.log(gt_w / anchor_w + 1e-6)
    dh = torch.log(gt_h / anchor_h + 1e-6)
    targets = torch.stack((dx, dy, dw, dh), dim=1)
    return targets


def decode_boxes(anchors, offsets):
    ax = anchors[:, 0]
    ay = anchors[:, 1]
    ax2 = anchors[:, 2]
    ay2 = anchors[:, 3]
    anchor_w = ax2 - ax
    anchor_h = ay2 - ay
    anchor_ctr_x = ax + 0.5 * anchor_w
    anchor_ctr_y = ay + 0.5 * anchor_h

    dx = offsets[:, 0]
    dy = offsets[:, 1]
    dw = offsets[:, 2]
    dh = offsets[:, 3]

    pred_ctr_x = dx * anchor_w + anchor_ctr_x
    pred_ctr_y = dy * anchor_h + anchor_ctr_y
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes


def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    tensor = tensor.cpu().numpy()
    tensor = tensor * std + mean
    tensor = np.clip(tensor, 0, 1)
    img = np.transpose(tensor, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return img


# ============================
# Vectorized Pseudo GT Update (Revised per Paper)
# ============================
def update_pseudo_gt_torch(pseudo_gt_np, anchors_np, pred_scores, pred_offsets, iou_threshold=0.7):
    """
    For each initialized pseudo GT box (g0), among all anchors with IoU > iou_threshold,
    select the candidate with the highest score whose predicted box’s minimum side (width or height)
    is smaller than d0 (the original g0 side length). If none, keep g0.
    Inputs:
      pseudo_gt_np: numpy array (M,4)
      anchors_np: either numpy array or tensor (N,4)
      pred_scores: tensor (N,)
      pred_offsets: tensor (N,4)
    Returns:
      updated pseudo GT as a numpy array (M,4)
    """
    device = pred_scores.device
    if isinstance(pseudo_gt_np, np.ndarray):
        pseudo_gt = torch.from_numpy(pseudo_gt_np).to(device).float()
    else:
        pseudo_gt = pseudo_gt_np.to(device).float()
    if isinstance(anchors_np, np.ndarray):
        anchors = torch.from_numpy(anchors_np).to(device).float()
    else:
        anchors = anchors_np.to(device).float()

    M = pseudo_gt.shape[0]
    updated_boxes = []
    for i in range(M):
        g0 = pseudo_gt[i]  # (4,)
        d0 = g0[2] - g0[0]  # side length (square)
        # Compute IoU between g0 and all anchors
        g0_exp = g0.unsqueeze(0)  # (1,4)
        iou = compute_iou_tensor(anchors, g0_exp).squeeze(1)  # (N,)
        cand_idx = (iou > iou_threshold).nonzero(as_tuple=False).squeeze(1)
        if cand_idx.numel() == 0:
            updated_boxes.append(g0)
            continue
        cand_anchors = anchors[cand_idx]
        cand_scores = pred_scores[cand_idx]
        cand_offsets = pred_offsets[cand_idx]
        cand_boxes = decode_boxes(cand_anchors, cand_offsets)
        widths = cand_boxes[:, 2] - cand_boxes[:, 0]
        heights = cand_boxes[:, 3] - cand_boxes[:, 1]
        cand_min_side = torch.min(widths, heights)
        valid_mask = cand_min_side < d0
        if valid_mask.sum() == 0:
            updated_boxes.append(g0)
        else:
            valid_indices = cand_idx[valid_mask]
            valid_scores = pred_scores[valid_indices]
            best_score, best_idx_in_valid = torch.max(valid_scores, dim=0)
            best_candidate_idx = valid_indices[best_idx_in_valid]
            best_anchor = anchors[best_candidate_idx]
            best_offset = pred_offsets[best_candidate_idx]
            best_box = decode_boxes(best_anchor.unsqueeze(0), best_offset.unsqueeze(0)).squeeze(0)
            updated_boxes.append(best_box)
    updated_boxes = torch.stack(updated_boxes, dim=0)
    return updated_boxes.cpu().numpy()


# ============================
# Visualization Functions
# ============================
def visualize_training_samples(dataset, num_low=2, num_high=2):
    difficulties = np.array(dataset.difficulties)
    sorted_indices = np.argsort(difficulties)
    low_indices = sorted_indices[:num_low]
    high_indices = sorted_indices[-num_high:]
    indices = np.concatenate([low_indices, high_indices])
    plt.figure(figsize=(15, 4 * len(indices)))
    for i, idx in enumerate(indices):
        sample = dataset.samples[idx]
        img = denormalize_image(sample['image'])
        ax = plt.subplot(len(indices), 1, i + 1)
        ax.imshow(img)
        pts = np.array(sample['points'])
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c='r', s=20)
        ax.set_title(f"Difficulty: {sample['difficulty']:.3f}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_evaluation_samples(model, test_dataset, anchors_np, device, num_samples=5, score_thresh=0.8):
    model.eval()
    indices = random.sample(range(len(test_dataset)), num_samples)
    plt.figure(figsize=(12, 5 * num_samples))
    for i, idx in enumerate(indices):
        sample = test_dataset.samples[idx]
        img_tensor = sample['image'].unsqueeze(0).to(device)
        img_disp = denormalize_image(sample['image'])
        gt_points = sample['points']
        with torch.no_grad():
            pred = model(img_tensor)[0]
        scores = pred[:, 0]
        offsets = pred[:, 1:]
        anchors_tensor = torch.from_numpy(anchors_np).float().to(device)
        boxes = decode_boxes(anchors_tensor, offsets)
        probs = torch.sigmoid(scores)
        keep = (probs > score_thresh)
        selected_boxes = boxes[keep].cpu().numpy()
        centers = []
        for b in selected_boxes:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            centers.append((cx, cy))

        ax1 = plt.subplot(num_samples, 2, 2 * i + 1)
        ax1.imshow(img_disp)
        pts = np.array(gt_points)
        if pts.size > 0:
            ax1.scatter(pts[:, 0], pts[:, 1], c='r', s=20)
        ax1.set_title("Ground Truth")
        ax1.axis('off')

        ax2 = plt.subplot(num_samples, 2, 2 * i + 2)
        ax2.imshow(img_disp)
        for b in selected_boxes:
            rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                     linewidth=1.5, edgecolor='g', facecolor='none')
            ax2.add_patch(rect)
        if len(centers) > 0:
            centers = np.array(centers)
            ax2.scatter(centers[:, 0], centers[:, 1], c='b', s=20)
        ax2.set_title("Predictions (Boxes & Centers)")
        ax2.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_pseudo_gt_update(sample, pseudo_before, pseudo_after):
    img = denormalize_image(sample['image'])
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    for bbox in pseudo_before:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title("Before Pseudo GT Update")
    ax1.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(img)
    for bbox in pseudo_after:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title("After Pseudo GT Update")
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


# ============================
# Dataset: Preloaded and Cached ShanghaiTechDataset
# ============================
class ShanghaiTechDataset(Dataset):
    def __init__(self, root, part='partA', split='train', image_size=(512, 512), transform=None, cache_dir="cache"):
        self.root = root
        self.part = part
        self.split = split
        self.image_size = image_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.data_dir = os.path.join(root, part, f"{'train_data' if split == 'train' else 'test_data'}")
        self.images_dir = os.path.join(self.data_dir, "images")
        self.gt_dir = os.path.join(self.data_dir, "ground_truth")

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith('.jpg')])
        self.gt_files = [os.path.join(self.gt_dir,"GT_" + os.path.splitext(f)[0] + '.mat') for f in self.image_files]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        cache_path = os.path.join(cache_dir, f"preprocessed_{part}_{split}_{image_size[0]}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading preprocessed data from {cache_path} ...")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.samples = cache["samples"]
            self.difficulties = cache["difficulties"]
        else:
            print("Preprocessing data...")
            self.samples = []
            self.difficulties = []
            for idx in range(len(self.image_files)):
                img_path = os.path.join(self.images_dir, self.image_files[idx])
                image = Image.open(img_path).convert("RGB")
                orig_w, orig_h = image.size
                new_w, new_h = self.image_size
                scale_x = new_w / orig_w
                scale_y = new_h / orig_h
                image_tensor = self.transform(image)

                gt_path = self.gt_files[idx]
                gt_data = loadmat(gt_path)
                points = gt_data['image_info']['location']
                points = [(float(p[0]), float(p[1])) for p in points]
                scaled_points = [(p[0] * scale_x, p[1] * scale_y) for p in points]
                pseudo_gt = generate_pseudo_gt_from_points(scaled_points)

                dists = []
                for i, pt in enumerate(scaled_points):
                    if len(scaled_points) > 1:
                        d = min([compute_euclidean_distance(pt, q) for j, q in enumerate(scaled_points) if i != j])
                    else:
                        d = 32.0
                    dists.append(d)
                dists = np.array(dists)
                mu = np.mean(dists)
                sigma = np.std(dists) if np.std(dists) > 0 else 1.0
                scores = np.exp(-((dists - mu) ** 2) / (2 * sigma ** 2))
                difficulty = 1 - np.mean(scores)

                gt_count = int(gt_data['image_info']['number'])
                sample = {
                    'image': image_tensor,
                    'points': scaled_points,
                    'pseudo_gt': pseudo_gt,
                    'gt_count': gt_count,
                    'difficulty': difficulty
                }
                self.samples.append(sample)
                self.difficulties.append(difficulty)
            print("Saving preprocessed data to cache...")
            with open(cache_path, "wb") as f:
                pickle.dump({"samples": self.samples, "difficulties": self.difficulties}, f)
            print("Preprocessing done.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================
# Model: PSDDN
# ============================
class PSDDN(nn.Module):
    def __init__(self, num_anchors=9):
        super(PSDDN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        in_channels_layer3 = 1024
        in_channels_layer4 = 2048
        out_channels = num_anchors * (1 + 4)

        self.det_head3 = nn.Conv2d(in_channels_layer3, out_channels, kernel_size=1)
        self.det_head4 = nn.Conv2d(in_channels_layer4, out_channels, kernel_size=1)

        nn.init.normal_(self.det_head3.weight, std=0.01)
        nn.init.constant_(self.det_head3.bias, 0)
        nn.init.normal_(self.det_head4.weight, std=0.01)
        nn.init.constant_(self.det_head4.bias, 0)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat3 = self.layer3(x)
        feat4 = self.layer4(feat3)

        pred3 = self.det_head3(feat3)
        pred4 = self.det_head4(feat4)
        pred4_up = F.interpolate(pred4, size=pred3.shape[2:], mode='bilinear', align_corners=False)
        final_pred = pred3 + pred4_up

        B, C, Hf, Wf = final_pred.shape
        self.feature_map_shape = (Hf, Wf)  # Save feature map size for anchor generation.
        T = C // 5
        final_pred = final_pred.view(B, T, 5, Hf, Wf).permute(0, 3, 4, 1, 2).contiguous()
        final_pred = final_pred.view(B, -1, 5)
        return final_pred


# ============================
# Loss Functions (Vectorized GPU)
# ============================
def compute_total_loss(pred, anchors_np, gt_boxes_np):
    anchors = torch.from_numpy(anchors_np).to(pred.device).float()  # (N,4)
    gt_boxes = torch.from_numpy(gt_boxes_np).to(pred.device).float()  # (M,4)
    labels, assigned_gt = assign_anchors_to_gt_torch(anchors, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.3)
    target_offsets = vectorized_bbox_transform(anchors, assigned_gt)
    scores = pred[:, 0]
    offsets = pred[:, 1:]
    valid_mask = labels != -1
    if valid_mask.sum() > 0:
        cls_loss = F.binary_cross_entropy_with_logits(scores[valid_mask].float(), labels[valid_mask].float())
    else:
        cls_loss = torch.tensor(0.0, device=pred.device)
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        pred_offsets_pos = offsets[pos_mask]
        target_offsets_pos = target_offsets[pos_mask]
        anchors_pos = anchors[pos_mask]
        pred_boxes = decode_boxes(anchors_pos, pred_offsets_pos)
        reg_loss = F.smooth_l1_loss(pred_offsets_pos, target_offsets_pos, reduction='mean')
        lc_loss = locally_constrained_regression_loss(pred_offsets_pos, target_offsets_pos, pred_boxes)
    else:
        reg_loss = torch.tensor(0.0, device=pred.device)
        lc_loss = torch.tensor(0.0, device=pred.device)
    return cls_loss + reg_loss + lc_loss


def locally_constrained_regression_loss(pred_offsets, target_offsets, pred_boxes, local_band_size=3):
    loss_center = F.smooth_l1_loss(pred_offsets[:, :2], target_offsets[:, :2], reduction='mean')
    N = pred_boxes.shape[0]
    loss_wh = 0.0
    group_size = local_band_size
    num_groups = max(1, N // group_size)
    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, N)
        group_boxes = pred_boxes[start:end]
        widths = group_boxes[:, 2] - group_boxes[:, 0]
        heights = group_boxes[:, 3] - group_boxes[:, 1]
        mu_w = torch.mean(widths)
        sigma_w = torch.std(widths) + 1e-6
        mu_h = torch.mean(heights)
        sigma_h = torch.std(heights) + 1e-6
        for w in widths:
            if w > mu_w + 3 * sigma_w:
                loss_wh += (w - (mu_w + 3 * sigma_w)) ** 2
            elif w < mu_w - 3 * sigma_w:
                loss_wh += ((mu_w - 3 * sigma_w) - w) ** 2
        for h in heights:
            if h > mu_h + 3 * sigma_h:
                loss_wh += (h - (mu_h + 3 * sigma_h)) ** 2
            elif h < mu_h - 3 * sigma_h:
                loss_wh += ((mu_h - 3 * sigma_h) - h) ** 2
    loss_wh = loss_wh / (N + 1e-6)
    return loss_center + loss_wh


def assign_anchors_to_gt_torch(anchors, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.3):
    iou_matrix = compute_iou_tensor(anchors, gt_boxes)  # (N,M)
    max_iou, argmax_iou = torch.max(iou_matrix, dim=1)
    labels = torch.full((anchors.shape[0],), -1, dtype=torch.int64, device=anchors.device)
    labels[max_iou < neg_iou_threshold] = 0
    labels[max_iou >= pos_iou_threshold] = 1
    assigned_gt = gt_boxes[argmax_iou]
    return labels, assigned_gt


def vectorized_bbox_transform(anchors, gt_boxes):
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_w
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - anchor_ctr_x) / anchor_w
    dy = (gt_ctr_y - anchor_ctr_y) / anchor_h
    dw = torch.log(gt_w / anchor_w + 1e-6)
    dh = torch.log(gt_h / anchor_h + 1e-6)
    targets = torch.stack((dx, dy, dw, dh), dim=1)
    return targets


def compute_iou_tensor(anchors, gt_boxes):
    N = anchors.shape[0]
    M = gt_boxes.shape[0]
    x1 = torch.max(anchors[:, None, 0], gt_boxes[None, :, 0])
    y1 = torch.max(anchors[:, None, 1], gt_boxes[None, :, 1])
    x2 = torch.min(anchors[:, None, 2], gt_boxes[None, :, 2])
    y2 = torch.min(anchors[:, None, 3], gt_boxes[None, :, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_anchor = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_anchor[:, None] + area_gt[None, :] - inter + 1e-6
    iou = inter / union
    return iou


# ============================
# Vectorized Pseudo GT Update (Revised per Paper)
# ============================
def update_pseudo_gt_torch(pseudo_gt_np, anchors_np, pred_scores, pred_offsets, iou_threshold=0.7):
    """
    For each pseudo GT box g0, among all anchors with IoU > iou_threshold,
    select the candidate with the highest score whose predicted box's
    minimum side is smaller than the original g0 side length.
    """
    device = pred_scores.device
    if isinstance(pseudo_gt_np, np.ndarray):
        pseudo_gt = torch.from_numpy(pseudo_gt_np).to(device).float()
    else:
        pseudo_gt = pseudo_gt_np.to(device).float()
    if isinstance(anchors_np, np.ndarray):
        anchors = torch.from_numpy(anchors_np).to(device).float()
    else:
        anchors = anchors_np.to(device).float()
    M = pseudo_gt.shape[0]
    updated_boxes = []
    for i in range(M):
        g0 = pseudo_gt[i]  # (4,)
        d0 = g0[2] - g0[0]
        g0_exp = g0.unsqueeze(0)  # (1,4)
        iou = compute_iou_tensor(anchors, g0_exp).squeeze(1)  # (N,)
        cand_idx = (iou > iou_threshold).nonzero(as_tuple=False).squeeze(1)
        if cand_idx.numel() == 0:
            updated_boxes.append(g0)
            continue
        cand_anchors = anchors[cand_idx]
        cand_scores = pred_scores[cand_idx]
        cand_offsets = pred_offsets[cand_idx]
        cand_boxes = decode_boxes(cand_anchors, cand_offsets)
        widths = cand_boxes[:, 2] - cand_boxes[:, 0]
        heights = cand_boxes[:, 3] - cand_boxes[:, 1]
        cand_min_side = torch.min(widths, heights)
        valid_mask = cand_min_side < d0
        if valid_mask.sum() == 0:
            updated_boxes.append(g0)
        else:
            valid_indices = cand_idx[valid_mask]
            valid_scores = pred_scores[valid_indices]
            best_score, best_idx_in_valid = torch.max(valid_scores, dim=0)
            best_candidate_idx = valid_indices[best_idx_in_valid]
            best_anchor = anchors[best_candidate_idx]
            best_offset = pred_offsets[best_candidate_idx]
            best_box = decode_boxes(best_anchor.unsqueeze(0), best_offset.unsqueeze(0)).squeeze(0)
            updated_boxes.append(best_box)
    updated_boxes = torch.stack(updated_boxes, dim=0)
    return updated_boxes.cpu().numpy()


# ============================
# Manual Batching Function
# ============================
def get_batches(samples, batch_size):
    random.shuffle(samples)
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


# ============================
# Training and Evaluation (Manual Batching, Preloaded Data)
# ============================
def train_model(args):
    train_dataset = ShanghaiTechDataset(
        root=args.data_root,
        part=args.part,
        split='train',
        image_size=(args.img_size, args.img_size),
        cache_dir=args.cache_dir
    )
    total_epochs = args.epochs

    if args.visualize_train:
        print("Visualizing training samples with extreme difficulties...")
        visualize_training_samples(train_dataset, num_low=2, num_high=2)

    # Instead of using DataLoader, we use manual batching.
    feature_map_size = (args.img_size // 16, args.img_size // 16)
    stride = 16
    anchor_scales = [32, 64, 128]
    anchor_ratios = [0.5, 1.0, 2.0]
    anchors_np = generate_anchors(feature_map_size, stride, anchor_scales, anchor_ratios)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSDDN(num_anchors=len(anchor_scales) * len(anchor_ratios)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    global_step = 0
    for epoch in range(1, total_epochs + 1):
        print(f"Epoch {epoch}/{total_epochs}")
        selected_indices = curriculum_sampler(train_dataset, epoch, total_epochs)
        print(f"Using {len(selected_indices)}/{len(train_dataset)} samples this epoch (curriculum fraction)")
        subset_samples = [train_dataset.samples[i] for i in selected_indices]
        epoch_batches = list(get_batches(subset_samples, args.batch_size))

        running_loss = 0.0
        for batch in epoch_batches:
            batch_images = torch.stack([s['image'] for s in batch], dim=0).to(device)
            B = batch_images.size(0)
            optimizer.zero_grad()
            preds = model(batch_images)
            batch_loss = 0.0
            for b in range(B):
                pred = preds[b]
                gt_boxes = batch[b]['pseudo_gt']
                loss = compute_total_loss(pred, anchors_np, gt_boxes)
                batch_loss += loss
            batch_loss = batch_loss / B
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
            global_step += 1
            if global_step % 10 == 0:
                print(f"Step {global_step}: Loss = {batch_loss.item():.4f}")
        epoch_loss = running_loss / len(epoch_batches)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

        # Save a copy of pseudo_gt before update for one sample.
        vis_idx = selected_indices[0]
        sample_before = train_dataset.samples[vis_idx]['pseudo_gt'].copy()
        model.eval()
        for idx in selected_indices:
            sample = train_dataset.samples[idx]
            img_path = os.path.join(train_dataset.images_dir, train_dataset.image_files[idx])
            image = Image.open(img_path).convert("RGB")
            image = train_dataset.transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(image)[0].cpu()
            pred_scores = torch.sigmoid(pred[:, 0])
            pred_offsets = pred[:, 1:]
            new_pseudo_gt = update_pseudo_gt_torch(sample['pseudo_gt'], anchors_np, pred_scores, pred_offsets,
                                                   iou_threshold=0.7)
            train_dataset.samples[idx]['pseudo_gt'] = new_pseudo_gt
        if args.visualize_train:
            sample_after = train_dataset.samples[vis_idx]['pseudo_gt']
            print("Visualizing pseudo GT update for one sample:")
            visualize_pseudo_gt_update(train_dataset.samples[vis_idx], sample_before, sample_after)
        model.train()

        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.save_dir, f"psddn_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete.")


def evaluate_model(args):
    test_dataset = ShanghaiTechDataset(
        root=args.data_root,
        part=args.part,
        split='test',
        image_size=(args.img_size, args.img_size),
        cache_dir=args.cache_dir
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSDDN(num_anchors=len([32, 64, 128]) * len([0.5, 1.0, 2.0])).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Generate anchors using feature map size from a forward pass
    with torch.no_grad():
        sample_img = test_dataset.samples[0]['image'].unsqueeze(0).to(device)
        _ = model(sample_img)  # this sets model.feature_map_shape
    Hf, Wf = model.feature_map_shape
    stride = 16
    anchor_scales = [32, 64, 128]
    anchor_ratios = [0.5, 1.0, 2.0]
    anchors_np = generate_anchors((Hf, Wf), stride, anchor_scales, anchor_ratios)

    total_error = 0.0
    num_images = len(test_dataset)
    for sample in test_dataset.samples:
        image = sample['image'].unsqueeze(0).to(device)
        gt_count = sample['gt_count']
        with torch.no_grad():
            preds = model(image)[0].to(device)
        scores = preds[:, 0]
        offsets = preds[:, 1:]
        anchors_tensor = torch.from_numpy(anchors_np).float().to(device)
        boxes = decode_boxes(anchors_tensor, offsets)
        probs = torch.sigmoid(scores)
        score_thresh = 0.8
        keep = (probs > score_thresh)
        pred_count = keep.sum().item()
        if gt_count > 0:
            error = abs(pred_count - gt_count) / gt_count
        else:
            error = 0.0
        total_error += error
    mape = (total_error / num_images) * 100
    print(f"Evaluation MAPE: {mape:.2f}%")

    if args.visualize_eval:
        visualize_evaluation_samples(model, test_dataset, anchors_np, device, num_samples=5, score_thresh=score_thresh)
    return mape

def curriculum_sampler(dataset, current_epoch, total_epochs):
    difficulties = np.array(dataset.difficulties)
    sorted_indices = np.argsort(difficulties)
    frac = min(1.0, (current_epoch / total_epochs) * 2)
    num_samples = int(frac * len(dataset))
    selected_indices = sorted_indices[:num_samples]
    return selected_indices.tolist()
# ============================
# Main and Argument Parsing
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="PSDDN for Crowd Counting on ShanghaiTech (GPU Vectorized)")
    parser.add_argument("--data_root", type=str, default="datasets", help="Root folder of datasets")
    parser.add_argument("--part", type=str, choices=["partA", "partB"], default="partA", help="Dataset part (A or B)")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train",
                        help="Mode: train or evaluation")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size (square)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model_path", type=str, default="checkpoints/psddn_epoch5.pth",
                        help="Path to trained model for evaluation")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to save preprocessed dataset cache")
    parser.add_argument("--visualize_train", action="store_true",
                        help="Visualize training samples and pseudo GT updates")
    parser.add_argument("--visualize_eval", action="store_true", help="Visualize evaluation samples")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        evaluate_model(args)


if __name__ == '__main__':
    main()
