"""
eval_cacvit_crowd.py

This script loads the trained CACViT model and evaluates it on the test_data from the PartA dataset.
For each test sample, it visualizes:
    - The original image.
    - The ground-truth density map (derived from the .mat file).
    - The predicted density map.
The title on each figure displays both the ground truth count and the predicted count.
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from mat4py import loadmat
from scipy.ndimage import gaussian_filter

# --------------------------
# If available, import your model & dataset definitions.
# Otherwise, we include minimal definitions here.
# --------------------------
try:
    from CACViT import cacvit_single, CrowdCountingDataset
except ImportError:
    # Minimal dataset definition for evaluation
    class CrowdCountingDataset(torch.utils.data.Dataset):
        def __init__(self, root, mode='test', img_size=384, transform=None):
            """
            Args:
                root: path to datasets/partA
                mode: 'train' or 'test'
                img_size: image resolution (both height and width)
                transform: transformation to apply to the image (e.g., resize, normalization)
            """
            self.mode = mode
            self.img_size = img_size
            self.transform = transform
            if mode == 'train':
                self.img_dir = os.path.join(root, 'train_data', 'images')
                self.gt_dir = os.path.join(root, 'train_data', 'ground-truth')
            else:
                self.img_dir = os.path.join(root, 'test_data', 'images')
                self.gt_dir = os.path.join(root, 'test_data', 'ground-truth')
            self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith('.jpg')])
        def __len__(self):
            return len(self.img_files)
        def __getitem__(self, idx):
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            # Ground truth file: assume filename pattern "GT_" + base name + ".mat"
            gt_name = 'GT_' + os.path.splitext(img_name)[0] + '.mat'
            gt_path = os.path.join(self.gt_dir, gt_name)
            # Load image and keep a copy for visualization
            img = Image.open(img_path).convert('RGB')
            orig_img = img.copy()
            orig_w, orig_h = img.size
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
                img = transforms.Resize((self.img_size, self.img_size))(img)
            # Load ground truth using mat4py
            mat = loadmat(gt_path)
            try:
                locations = mat['image_info']['location']
            except KeyError:
                locations = []
            points = []
            for p in locations:
                if isinstance(p, list) and len(p) >= 2:
                    points.append((float(p[0]), float(p[1])))
            # Create an empty density map of size (img_size, img_size)
            density = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            # Compute scaling factors (from original image to resized image)
            scale_w = self.img_size / orig_w
            scale_h = self.img_size / orig_h
            for point in points:
                x = point[0] * scale_w
                y = point[1] * scale_h
                x = min(self.img_size - 1, max(0, int(round(x))))
                y = min(self.img_size - 1, max(0, int(round(y))))
                density[y, x] += 1
            # Apply Gaussian filter to spread the point annotations
            density = gaussian_filter(density, sigma=4)
            density = torch.from_numpy(density).unsqueeze(0)  # (1, H, W)
            # Return three items
            return img, density, orig_img

    # Minimal model definition (should be replaced by your actual CACViT implementation)
    def cacvit_single(**kwargs):
        raise ImportError("Model definition not found. Please ensure 'train_cacvit_crowd.py' is in your PYTHONPATH.")

# --------------------------
# Function to convert normalized tensor back to a PIL image
# --------------------------
def tensor_to_pil(tensor):
    # Assuming normalization with mean=[0.485,0.456,0.406] and std=[0.229,0.224,0.225]
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    tensor = inv_normalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

# --------------------------
# Evaluation and Visualization Function
# --------------------------
def evaluate_and_visualize(model, test_loader, device, num_visualizations=10):
    """
    Evaluates the model on the test dataset and visualizes results.
    For each sample, displays:
      - Original image.
      - Ground truth density map.
      - Predicted density map.
    The title shows ground truth count and predicted count.
    """
    model.eval()
    visualized = 0
    with torch.no_grad():
        # Using custom collate, each batch is a list of samples.
        for batch in test_loader:
            # Since batch_size is 1, get the single sample.
            sample = batch[0]
            # Check the number of items returned by the dataset
            if len(sample) == 3:
                image, gt_density, orig_img = sample
            elif len(sample) == 2:
                image, gt_density = sample
                # Reconstruct the original image from the normalized tensor
                orig_img = tensor_to_pil(image.cpu()[0])
            else:
                raise ValueError("Dataset __getitem__ must return 2 or 3 items.")

            image = image.to(device).unsqueeze(0)       # (B, C, H, W)
            gt_density = gt_density.to(device).unsqueeze(0)  # (B, 1, H, W)
            output = model(image)                       # (B, H, W)
            pred_count = output.sum().item()
            gt_count = gt_density.sum().item()
            pred_density = output.cpu().squeeze(0).numpy()
            gt_density_np = gt_density.cpu().squeeze(0).numpy()

            # Plot the original image, ground truth, and prediction.
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            axs[0].imshow(orig_img)
            axs[0].axis('off')
            axs[0].set_title("Original Image")
            axs[1].imshow(gt_density_np, cmap='jet')
            axs[1].axis('off')
            axs[1].set_title(f"GT Density Map\nCount: {gt_count:.2f}")
            axs[2].imshow(pred_density, cmap='jet')
            axs[2].axis('off')
            axs[2].set_title(f"Predicted Density Map\nCount: {pred_count:.2f}")
            plt.suptitle(f"GT Count: {gt_count:.2f} | Pred Count: {pred_count:.2f}", fontsize=16)
            plt.tight_layout()
            plt.show()

            visualized += 1
            if visualized >= num_visualizations:
                break

# --------------------------
# Custom collate function that returns the list of samples directly.
# --------------------------
def custom_collate(batch):
    return batch

# --------------------------
# Argument Parsing and Main Function
# --------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CACViT Model on PartA Test Data")
    parser.add_argument('--dataset_path', type=str, default='datasets/partA', help='Path to the PartA dataset folder')
    parser.add_argument('--img_size', type=int, default=384, help='Image size (images will be resized to img_size x img_size)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation (use 1 for visualization)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--model_path', type=str, default='cacvit_crowd.pth', help='Path to the saved model weights')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--num_visualizations', type=int, default=10, help='Number of samples to visualize')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Define evaluation transform (same as used during training)
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the test dataset and DataLoader with the custom collate function
    test_dataset = CrowdCountingDataset(root=args.dataset_path, mode='test', img_size=args.img_size, transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    # Build and load the model
    model = cacvit_single(img_size=args.img_size, patch_size=16, in_chans=3,
                          embed_dim=768, depth=12, decoder_embed_dim=512, decoder_depth=3)
    model = model.to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded model weights from", args.model_path)

    # Evaluate and visualize results
    evaluate_and_visualize(model, test_loader, device, num_visualizations=args.num_visualizations)

if __name__ == '__main__':
    main()
