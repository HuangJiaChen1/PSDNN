import os
import torch
import clip
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from mat4py import loadmat
import time
from tqdm import tqdm

checkpoint_dir = 'checkpoints'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def save_checkpoint(model, epoch):
    """
    Save the model's state dictionary at the given epoch.

    Args:
        model (torch.nn.Module): The model to save.
        epoch (int): The current epoch number.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Model saved to {checkpoint_path}')
# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing function
# When loading the model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model = model.float()  # Convert to float32


# Define text descriptions for classification
texts = [
    "This picture contains no people.",
    "This picture contains 1 person.",
    "This picture contains 2 people.",
    "This picture contains 3 people.",
    "This picture contains 4 people.",
    "This picture contains 5 people.",
    "This picture contains 6 people.",
    "This picture contains 7 people.",
    "This picture contains 8 people.",
    "This picture contains 9 people.",
    "This picture contains 10 people.",
    "This picture contains more than 10 people."
]

# Tokenize and compute text features
text_tokens = clip.tokenize(texts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize text features


# Custom Dataset class (unchanged)
class CrowdCountingDataset(Dataset):
    def __init__(self, image_dir, gt_dir, block_size=32):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.block_size = block_size
        self.image_files = sorted(os.listdir(image_dir))
        self.gt_files = sorted(os.listdir(gt_dir))
        assert len(self.image_files) == len(self.gt_files), "Mismatch between images and ground-truth files."

        self.keys = []
        for img_idx in range(len(self.image_files)):
            img_path = os.path.join(image_dir, self.image_files[img_idx])
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            pw = ((w + block_size - 1) // block_size) * block_size
            ph = ((h + block_size - 1) // block_size) * block_size
            num_blocks_h = ph // block_size
            num_blocks_w = pw // block_size
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    self.keys.append((img_idx, i, j))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        img_idx, i, j = self.keys[index]
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[img_idx])

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        pw = ((w + self.block_size - 1) // self.block_size) * self.block_size
        ph = ((h + self.block_size - 1) // self.block_size) * self.block_size
        padded_img = Image.new('RGB', (pw, ph), (0, 0, 0))
        padded_img.paste(img, (0, 0))

        block = padded_img.crop((j * self.block_size, i * self.block_size,
                                 (j + 1) * self.block_size, (i + 1) * self.block_size))

        mat = loadmat(gt_path)
        locations = mat['image_info']['location']
        locations = [(float(p[0]), float(p[1])) for p in locations]

        count = 0
        for x, y in locations:
            if (j * self.block_size <= x < (j + 1) * self.block_size and
                    i * self.block_size <= y < (i + 1) * self.block_size):
                count += 1

        if count == 0:
            label = 0
        elif count == 1:
            label = 1
        elif count == 2:
            label = 2
        elif count == 3:
            label = 3
        elif count == 4:
            label = 4
        elif count == 5:
            label = 5
        elif count == 6:
            label = 6
        elif count == 7:
            label = 7
        elif count == 8:
            label = 8
        elif count == 9:
            label = 9
        elif count == 10:
            label = 10
        else:
            label = 11

        block = preprocess(block)
        return block, label


# Evaluation and visualization function (unchanged)
def evaluate_and_visualize(image_path, gt_path, block_size=32):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    pw = ((w + block_size - 1) // block_size) * block_size
    ph = ((h + block_size - 1) // block_size) * block_size
    padded_img = Image.new('RGB', (pw, ph), (0, 0, 0))
    padded_img.paste(img, (0, 0))
    num_blocks_h = ph // block_size
    num_blocks_w = pw // block_size

    density_map = np.zeros((num_blocks_h, num_blocks_w))
    total_pred_count = 0
    rep_counts = [0, 1, 2, 3,4,5,6,7,8,9,10,15]  # Representative counts

    with torch.no_grad():
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = padded_img.crop((j * block_size, i * block_size,
                                         (j + 1) * block_size, (i + 1) * block_size))
                block = preprocess(block).unsqueeze(0).to(device)
                image_features = model.encode_image(blocks)
                norms = image_features.norm(dim=-1, keepdim=True)
                normalized_image_features = image_features / (norms + 1e-8)  # Add epsilon
                logits = normalized_image_features @ text_features.T  # No scaling
                probs = logits.softmax(dim=-1).cpu().numpy()[0]
                estimated_count = sum(p * rc for p, rc in zip(probs, rep_counts))
                density_map[i, j] = estimated_count
                total_pred_count += estimated_count

    mat = loadmat(gt_path)
    locations = mat['image_info']['location']
    gt_count = len(locations)

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted Total Count: {total_pred_count:.2f}")
    print(f"Ground-Truth Total Count: {gt_count}")
    print(f"Mean Absolute Error: {abs(total_pred_count - gt_count):.2f}")

    density_map_resized = np.kron(density_map, np.ones((block_size, block_size)))
    density_map_resized = density_map_resized[:h, :w]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(density_map_resized, cmap='hot')
    plt.title(f"Density Map\nPred: {total_pred_count:.2f}, GT: {gt_count}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Define paths (adjust as needed)
    train_image_dir = 'C:/Users/admin/Documents/GitHub/PSDNN/datasets/partA/train_data/images'
    train_gt_dir = 'C:/Users/admin/Documents/GitHub/PSDNN/datasets/partA/train_data/ground_truth'
    test_image_dir = 'C:/Users/admin/Documents/GitHub/PSDNN/datasets/partA/test_data/images'
    test_gt_dir = 'C:/Users/admin/Documents/GitHub/PSDNN/datasets/partA/test_data/ground_truth'

    # Create dataset and dataloader
    train_dataset = CrowdCountingDataset(train_image_dir, train_gt_dir, block_size=32)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop with verbose output
    num_epochs = 5
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Starting Epoch {epoch + 1}/{num_epochs} at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

        # Wrap train_loader with tqdm for progress bar
        for batch_idx, (blocks, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            blocks = blocks.to(device)
            labels = labels.to(device)
            blocks = blocks.float()
            text_features = text_features.float()
            # Forward pass
            image_features = model.encode_image(blocks)
            if torch.isnan(image_features).any():
                print("NaN detected in image_features")
            normalized_image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            if torch.isnan(normalized_image_features).any():
                print("NaN detected in normalized_image_features")
            logits = 100.0 * (normalized_image_features @ text_features.T)
            if torch.isnan(logits).any():
                print("NaN detected in logits")
            # norms = image_features.norm(dim=-1, keepdim=True)
            # normalized_image_features = image_features / (norms + 1e-8)  # Add epsilon
            # logits = normalized_image_features @ text_features.T  # No scaling
            print(logits)
            print(labels)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track performance
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Print batch info every 10 batches
            if (batch_idx + 1) % 10 == 0:
                batch_accuracy = (preds == labels).sum().item() / labels.size(0)
                tqdm.write(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Batch Accuracy: {batch_accuracy:.4f}")

        # Calculate and print epoch summary
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f} seconds, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        save_checkpoint(model, epoch + 1)

    # Evaluate on a test image
    test_image_path = os.path.join(test_image_dir, 'IMG_2.jpg')
    test_gt_path = os.path.join(test_gt_dir, 'GT_IMG_2.mat')
    evaluate_and_visualize(test_image_path, test_gt_path)