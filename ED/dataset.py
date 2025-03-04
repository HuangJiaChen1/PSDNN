import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from mat4py import loadmat

class SHADataset(Dataset):
    def __init__(self, image_dir, gt_dir, target_size=(384, 384), train=True):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.target_size = target_size
        self.train = train
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

        # Split for train/test (e.g., 80% train, 20% test)
        split_idx = int(0.8 * len(self.image_files))
        self.image_files = self.image_files[:split_idx] if train else self.image_files[split_idx:]

        # Preprocessing for ViT (ImageNet normalization)
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def generate_density_map(self, image_size, points, sigma=8):
        density = np.zeros(image_size, dtype=np.float32)
        h, w = image_size
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                density[y, x] += 1
        density = gaussian_filter(density, sigma=sigma, mode='constant')
        return density

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, 'GT_' + self.image_files[idx].replace('.jpg', '.mat'))

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img_resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img_tensor = self.preprocess(img_resized)

        # Load ground truth annotations
        mat = loadmat(gt_path)
        points = mat['image_info'][0, 0]['location'][0, 0]  # Adjust based on actual .mat structure

        # Scale points to target size
        scale_w = self.target_size[0] / orig_w
        scale_h = self.target_size[1] / orig_h
        points[:, 0] *= scale_w
        points[:, 1] *= scale_h

        # Generate density map
        density = self.generate_density_map(self.target_size, points)
        density_tensor = torch.from_numpy(density).unsqueeze(0)  # (1, H, W)

        return img_tensor, density_tensor


# Example usage
if __name__ == "__main__":
    dataset = SHADataset(
        image_dir='C:/Users/admin\Documents\GitHub\PSDNN\datasets\partA/train_data\images',
        gt_dir='C:/Users/admin\Documents\GitHub\PSDNN\datasets\partA/train_data\ground_truth',
        train=True
    )
    img, density = dataset[0]
    print(img.shape, density.shape)  # Should be (3, 384, 384) and (1, 384, 384)