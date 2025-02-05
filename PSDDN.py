#!/usr/bin/env python
import os
import math
import random
import numpy as np
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset

from mat4py import loadmat  # pip install mat4py


# ============================
# Custom Collate Function
# ============================

def custom_collate(batch):
    """
    Collate function for the ShanghaiTechDataset.
    - Stacks images and scalar values.
    - Leaves variable-length items (points and pseudo_gt) as lists.
    """
    collated = {}
    collated['image'] = torch.stack([b['image'] for b in batch], dim=0)
    collated['points'] = [b['points'] for b in batch]  # leave as list
    collated['pseudo_gt'] = [b['pseudo_gt'] for b in batch]  # leave as list (numpy arrays)
    collated['gt_count'] = torch.tensor([b['gt_count'] for b in batch])
    collated['difficulty'] = torch.tensor([b['difficulty'] for b in batch])
    return collated


# ============================
# Helper Functions
# ============================

def compute_euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_pseudo_gt_from_points(points):
    """
    Given a list of points [(x,y), ...], compute a pseudo ground truth
    bounding box for each point. The box is square, centered at the point,
    with side length equal to the nearest-neighbor distance (or a default if only one point).
    Returns an array of shape (N,4) in [x1, y1, x2, y2] format.
    """
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
    """
    Generate anchor boxes for a feature map.
    Args:
        feature_size: tuple (H, W) of the feature map.
        stride: the stride (in pixels) of the feature map relative to the input image.
        scales: list of scales (e.g., [32, 64, 128])
        ratios: list of aspect ratios (e.g., [0.5, 1.0, 2.0])
    Returns:
        anchors: numpy array of shape (num_anchors, 4) in [x1, y1, x2, y2] format.
    """
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


def compute_iou(box1, box2):
    """
    Compute Intersection-over-Union (IoU) between two boxes.
    Boxes: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def bbox_transform(anchor, gt_box):
    """
    Compute the regression targets for transforming an anchor to a ground truth box.
    Both anchor and gt_box are arrays in [x1, y1, x2, y2] format.
    Returns:
        [dx, dy, dw, dh]
    """
    anchor_w = anchor[2] - anchor[0]
    anchor_h = anchor[3] - anchor[1]
    anchor_ctr_x = anchor[0] + 0.5 * anchor_w
    anchor_ctr_y = anchor[1] + 0.5 * anchor_h

    gt_w = gt_box[2] - gt_box[0]
    gt_h = gt_box[3] - gt_box[1]
    gt_ctr_x = gt_box[0] + 0.5 * gt_w
    gt_ctr_y = gt_box[1] + 0.5 * gt_h

    dx = (gt_ctr_x - anchor_ctr_x) / anchor_w
    dy = (gt_ctr_y - anchor_ctr_y) / anchor_h
    dw = math.log(gt_w / anchor_w + 1e-6)
    dh = math.log(gt_h / anchor_h + 1e-6)
    return np.array([dx, dy, dw, dh], dtype=np.float32)


def decode_boxes(anchors, offsets):
    """
    Decode predicted offsets back into box coordinates.
    Args:
        anchors: Tensor of shape (N,4)
        offsets: Tensor of shape (N,4) predicted offsets [dx, dy, dw, dh]
    Returns:
        boxes: Tensor of shape (N,4) in [x1, y1, x2, y2] format.
    """
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


# ============================
# Dataset: ShanghaiTechDataset
# ============================

class ShanghaiTechDataset(Dataset):
    """
    Dataset class for ShanghaiTech PartA or PartB.
    Assumes the following folder structure:
      datasets/
          partA/
              train_data/
                  images/
                  ground_truth/
              test_data/
                  images/
                  ground_truth/
          partB/
              train_data/
                  images/
                  ground_truth/
              test_data/
                  images/
                  ground_truth/
    The .mat files contain a dict with the form:
      {'image_info': {'location': [[x1, y1], [x2, y2], ...], 'number': count}}
    This class precomputes:
      - The scaled point annotations (based on the target image_size)
      - The pseudo ground truth boxes (using nearest neighbor distances)
      - A difficulty score per image (for curriculum learning)
      - The ground truth count
    """

    def __init__(self, root, part='partA', split='train', image_size=(512, 512), transform=None):
        self.root = root
        self.part = part
        self.split = split  # 'train' or 'test'
        self.image_size = image_size
        self.data_dir = os.path.join(root, part, f"{'train_data' if split == 'train' else 'test_data'}")
        self.images_dir = os.path.join(self.data_dir, "images")
        self.gt_dir = os.path.join(self.data_dir, "ground_truth")

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith('.jpg')])
        self.gt_files = [os.path.join(self.gt_dir, 'GT_'+os.path.splitext(f)[0] + '.mat') for f in self.image_files]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Precompute per-sample attributes.
        self.scaled_points_list = []
        self.pseudo_gts = []
        self.difficulties = []
        self.gt_counts = []

        for idx in range(len(self.image_files)):
            gt_path = self.gt_files[idx]
            gt_data = loadmat(gt_path)
            points = gt_data['image_info']['location']
            points = [(float(p[0]), float(p[1])) for p in points]
            img_path = os.path.join(self.images_dir, self.image_files[idx])
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
            new_w, new_h = self.image_size
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            scaled_points = [(p[0] * scale_x, p[1] * scale_y) for p in points]
            self.scaled_points_list.append(scaled_points)

            pseudo_gt = generate_pseudo_gt_from_points(scaled_points)
            self.pseudo_gts.append(pseudo_gt)

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
            self.difficulties.append(difficulty)

            gt_count = int(gt_data['image_info']['number'])
            self.gt_counts.append(gt_count)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        sample = {
            'image': image,  # Tensor (3, H, W)
            'points': self.scaled_points_list[idx],  # List of (x,y)
            'pseudo_gt': self.pseudo_gts[idx],  # numpy array (N,4)
            'gt_count': self.gt_counts[idx],  # integer
            'difficulty': self.difficulties[idx]  # float
        }
        return sample


# ============================
# Model: PSDDN
# ============================

class PSDDN(nn.Module):
    def __init__(self, num_anchors=9):
        """
        PSDDN using a ResNet-101 backbone.
        Detection heads are attached to layer3 and layer4.
        The head outputs (score, dx, dy, dw, dh) per anchor.
        """
        super(PSDDN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

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
        T = C // 5
        final_pred = final_pred.view(B, T, 5, Hf, Wf).permute(0, 3, 4, 1, 2).contiguous()
        final_pred = final_pred.view(B, -1, 5)
        return final_pred  # (B, N, 5)


# ============================
# Loss Functions
# ============================

def assign_anchors_to_gt(anchors, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.3):
    N = anchors.shape[0]
    M = gt_boxes.shape[0]
    labels = -np.ones((N,), dtype=np.int32)
    assigned_gt = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        max_iou = 0
        best_gt = None
        for j in range(M):
            iou = compute_iou(anchors[i], gt_boxes[j])
            if iou > max_iou:
                max_iou = iou
                best_gt = gt_boxes[j]
        if max_iou >= pos_iou_threshold:
            labels[i] = 1
            assigned_gt[i] = best_gt
        elif max_iou < neg_iou_threshold:
            labels[i] = 0
        else:
            labels[i] = -1
    return labels, assigned_gt


def compute_regression_targets(anchors, assigned_gt, labels):
    N = anchors.shape[0]
    target_offsets = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        if labels[i] == 1:
            target_offsets[i] = bbox_transform(anchors[i], assigned_gt[i])
    return target_offsets


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


def compute_total_loss(pred, anchors, gt_boxes):
    labels_np, assigned_gt_np = assign_anchors_to_gt(anchors, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.3)
    labels = torch.from_numpy(labels_np).to(pred.device).long()
    target_offsets_np = compute_regression_targets(anchors, assigned_gt_np, labels_np)
    target_offsets = torch.from_numpy(target_offsets_np).to(pred.device).float()

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
        anchors_pos = torch.from_numpy(anchors).to(pred.device)[pos_mask]
        pred_boxes = decode_boxes(anchors_pos, pred_offsets_pos)
        reg_loss = F.smooth_l1_loss(pred_offsets_pos, target_offsets_pos, reduction='mean')
        lc_loss = locally_constrained_regression_loss(pred_offsets_pos, target_offsets_pos, pred_boxes)
    else:
        reg_loss = torch.tensor(0.0, device=pred.device)
        lc_loss = torch.tensor(0.0, device=pred.device)

    return cls_loss + reg_loss + lc_loss


# ============================
# Pseudo Ground Truth Updating
# ============================

def update_pseudo_gt(pseudo_gt, anchors, pred_scores, pred_offsets, iou_threshold=0.5):
    new_pseudo_gt = pseudo_gt.copy()
    for i, gt_box in enumerate(pseudo_gt):
        best_score = -1
        best_box = None
        for j, anchor in enumerate(anchors):
            iou = compute_iou(anchor, gt_box)
            if iou > iou_threshold and pred_scores[j] > best_score:
                ax, ay, ax2, ay2 = anchor
                aw = ax2 - ax
                ah = ay2 - ay
                anchor_ctr_x = ax + 0.5 * aw
                anchor_ctr_y = ay + 0.5 * ah
                dx, dy, dw, dh = pred_offsets[j]
                pred_ctr_x = dx * aw + anchor_ctr_x
                pred_ctr_y = dy * ah + anchor_ctr_y
                pred_w = aw * math.exp(dw)
                pred_h = ah * math.exp(dh)
                pred_box = [pred_ctr_x - 0.5 * pred_w,
                            pred_ctr_y - 0.5 * pred_h,
                            pred_ctr_x + 0.5 * pred_w,
                            pred_ctr_y + 0.5 * pred_h]
                best_score = pred_scores[j]
                best_box = pred_box
        if best_box is not None:
            new_pseudo_gt[i] = best_box
    return new_pseudo_gt


# ============================
# Curriculum Learning Sampler
# ============================

def curriculum_sampler(dataset, current_epoch, total_epochs):
    difficulties = np.array(dataset.difficulties)
    sorted_indices = np.argsort(difficulties)
    frac = min(1.0, (current_epoch / total_epochs) * 2)
    num_samples = int(frac * len(dataset))
    selected_indices = sorted_indices[:num_samples]
    return selected_indices.tolist()


# ============================
# Training and Evaluation
# ============================

def train_model(args):
    train_dataset = ShanghaiTechDataset(
        root=args.data_root,
        part=args.part,
        split='train',
        image_size=(args.img_size, args.img_size)
    )
    total_epochs = args.epochs

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
        subset = Subset(train_dataset, selected_indices)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, collate_fn=custom_collate)

        running_loss = 0.0
        for batch in dataloader:
            images = batch['image'].to(device)
            B = images.size(0)
            pseudo_gt_list = batch['pseudo_gt']

            optimizer.zero_grad()
            preds = model(images)
            batch_loss = 0.0
            for b in range(B):
                pred = preds[b]
                gt_boxes = pseudo_gt_list[b]
                loss = compute_total_loss(pred, anchors_np, gt_boxes)
                batch_loss += loss
            batch_loss = batch_loss / B
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            global_step += 1
            if global_step % 10 == 0:
                print(f"Step {global_step}: Loss = {batch_loss.item():.4f}")
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

        # Pseudo ground truth update:
        model.eval()
        with torch.no_grad():
            for idx in selected_indices:
                sample = train_dataset[idx]
                img_path = os.path.join(train_dataset.images_dir, train_dataset.image_files[idx])
                image = Image.open(img_path).convert("RGB")
                image = train_dataset.transform(image).unsqueeze(0).to(device)
                pred = model(image)[0].cpu().numpy()
                pred_scores = pred[:, 0]
                pred_offsets = pred[:, 1:]
                new_pseudo_gt = update_pseudo_gt(sample['pseudo_gt'], anchors_np, pred_scores, pred_offsets)
                train_dataset.pseudo_gts[idx] = new_pseudo_gt
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
        image_size=(args.img_size, args.img_size)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSDDN(num_anchors=len([32, 64, 128]) * len([0.5, 1.0, 2.0])).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    feature_map_size = (args.img_size // 16, args.img_size // 16)
    stride = 16
    anchor_scales = [32, 64, 128]
    anchor_ratios = [0.5, 1.0, 2.0]
    anchors_np = generate_anchors(feature_map_size, stride, anchor_scales, anchor_ratios)

    total_error = 0.0
    num_images = len(test_dataset)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate)
    for batch in dataloader:
        image = batch['image'].to(device)
        gt_count = batch['gt_count'][0]
        with torch.no_grad():
            preds = model(image)[0].cpu()
        scores = preds[:, 0]
        offsets = preds[:, 1:]
        anchors_tensor = torch.from_numpy(anchors_np).float()
        boxes = decode_boxes(anchors_tensor, offsets)
        probs = torch.sigmoid(scores)
        score_thresh = 0.8
        keep = (probs > score_thresh)
        pred_count = keep.sum().item()
        total_error += abs(pred_count - gt_count)
    mae = total_error / num_images
    print(f"Evaluation MAE: {mae:.2f}")
    return mae


# ============================
# Main
# ============================

def parse_args():
    parser = argparse.ArgumentParser(description="PSDDN for Crowd Counting on ShanghaiTech")
    parser.add_argument("--data_root", type=str, default="datasets", help="Root folder of datasets")
    parser.add_argument("--part", type=str, choices=["partA", "partB"], default="partA", help="Dataset part (A or B)")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train",
                        help="Mode: train or evaluation")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size (square)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model_path", type=str, default="checkpoints/psddn_epoch50.pth",
                        help="Path to trained model for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        evaluate_model(args)


if __name__ == '__main__':
    main()
