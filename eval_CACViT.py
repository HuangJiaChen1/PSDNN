from functools import partial

import numpy as np
import torch
from PIL.Image import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from beiyong2 import CrowdCountingDataset, transform, exemplar_transform, CACViT, custom_collate_fn
from ultralytics import YOLO
from YCV import CACVIT

import os
from PIL import Image
from mat4py import loadmat

def gaussian_kernel(kernel_size=15, sigma=4):
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)

class CrowdCountingDatasetTest(Dataset):
    def __init__(self, img_dir, gt_dir, exemplars_dir,
                 transform=None, exemplar_transform=None, num_exemplars=3):
        """
        Args:
            img_dir: Directory with test images.
            gt_dir: Directory with ground truth .mat files.
            exemplars_dir: Directory with exemplar images.
            transform: Transform to apply to the query image (e.g., normalization).
                       Do not include resizing if you wish to keep the original resolution.
            exemplar_transform: Transform for the exemplar images.
            num_exemplars: Number of exemplars to use.
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.exemplars_dir = exemplars_dir
        self.transform = transform
        self.exemplar_transform = exemplar_transform
        self.num_exemplars = num_exemplars
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Load image at original resolution.
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        # Load ground truth annotations
        gt_name = 'GT_' + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = loadmat(gt_path)
        locations = mat['image_info']['location']
        locations = [(float(p[0]), float(p[1])) for p in locations]

        # Generate density map for the entire image
        density_map = self.generate_density_map((orig_height, orig_width), locations)

        # Load exemplar images based on naming pattern (adjust as needed)
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

        # Fill exemplars if missing
        if len(exemplars) < self.num_exemplars:
            if len(exemplars) == 0:
                dummy_ex = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                exemplars = [dummy_ex] * self.num_exemplars
            else:
                exemplars = exemplars * (self.num_exemplars // len(exemplars)) + exemplars[:self.num_exemplars % len(exemplars)]
        elif len(exemplars) > self.num_exemplars:
            exemplars = exemplars[:self.num_exemplars]

        # Compute scale factors for exemplars based on the original image size.
        # Here we assume the exemplar images are 64x64, and we set the scale such that
        # 64 corresponds to the maximum dimension of the original image.
        scale_value = 64 / max(orig_width, orig_height)
        scales = torch.ones((self.num_exemplars, 2), dtype=torch.float32) * scale_value

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        if self.exemplar_transform:
            exemplars = torch.stack([self.exemplar_transform(ex) for ex in exemplars])
        else:
            # If no transform is provided, convert exemplars to tensors
            exemplars = torch.stack([transforms.ToTensor()(ex) for ex in exemplars])

        # Convert density map to tensor and scale if needed.
        density_map = torch.from_numpy(density_map).float() * 60  # Adjust scaling as before

        # Ground truth count (number of annotations)
        gt_count = len(locations)

        return image, density_map, exemplars, scales, gt_count

    def generate_density_map(self, img_shape, points, sigma=2):
        """
        Args:
            img_shape: Tuple (height, width) of the image.
            points: List of (x,y) coordinates.
            sigma: Standard deviation for the Gaussian kernel.
        Returns:
            A density map as a numpy array of shape (height, width)
        """
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
# Evaluation Code (MAE and MAPE metrics)
#############################################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths (update these paths as necessary)
    test_img_dir = 'datasets/partB/test_data/images'
    test_gt_dir = 'datasets/partB/test_data/ground_truth'
    exemplars_dir = 'datasets/partA/examplars'

    test_dataset = CrowdCountingDatasetTest(
        test_img_dir, test_gt_dir, exemplars_dir,
        transform=transform, exemplar_transform=exemplar_transform, num_exemplars=3,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    # Initialize model and load weights
    # model = CACViT(num_exemplars=3, img_size=384, patch_size=16, embed_dim=768).to(device)
    model = CACVIT(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                   decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                   mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = model.to(device)
    checkpoint = torch.load("YCV300.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_mae = 0.0
    total_mape = 0.0
    num_samples = 0
    epsilon = 1e-6

    with torch.no_grad():
        for sample in test_loader:
            image, gt_density, exemplars, scales, gt_counts = sample
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            img_tensor = image.cpu().squeeze(0)
            disp_img = inv_normalize(img_tensor)
            disp_img = torch.clamp(disp_img, 0, 1)
            pil_image = transforms.ToPILImage()(disp_img)

            image = image.to(device)
            scales = scales.to(device)
            exemplars = exemplars.to(device)
            output_density = sliding_window_inference(model, image, exemplars, scales, device)
            pred_count = output_density.sum().item()/60
            gt_count = float(gt_counts[0])
            total_mae += abs(pred_count - gt_count)
            total_mape += abs(pred_count - gt_count) / (gt_count + epsilon)
            num_samples += 1

    mae = total_mae / num_samples
    mape = (total_mape / num_samples) * 100

    print(f"Test MAE: {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")
    with torch.no_grad():
        # Visualize one sample
        sample = next(iter(test_loader))
        test_img, test_gt_density, test_exemplars, test_scales, gt_counts = sample
        test_img = test_img.to(device)
        test_exemplars = test_exemplars.to(device)
        test_scales = test_scales.to(device)
        output_density = model([test_img, test_exemplars, test_scales])
        output_density_np = output_density.squeeze(0).cpu().numpy()
        gt_density_np = test_gt_density.squeeze(0).cpu().numpy()
        pred_count = output_density_np.sum()/60
        gt_count = float(gt_counts[0])

        # Convert normalized image back to PIL image for display (reverse normalization)
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        disp_img = test_img.squeeze(0).cpu()
        disp_img = inv_normalize(disp_img)
        disp_img = torch.clamp(disp_img, 0, 1)
        disp_img = transforms.ToPILImage()(disp_img)

        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1)
        plt.imshow(disp_img)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(gt_density_np, cmap='jet')
        plt.title(f"GT Density Map\nCount: {gt_count:.2f}")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(output_density_np, cmap='jet')
        plt.title(f"Predicted Density Map\nCount: {pred_count:.2f}")
        plt.axis('off')

        plt.suptitle("Test Evaluation Sample", fontsize=16)
        plt.tight_layout()
        plt.show()