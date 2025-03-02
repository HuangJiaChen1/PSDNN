import os
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from mat4py import loadmat

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_predicted_count(image_path, model, text_features, preprocess, block_size=32, rep_counts=[0, 1, 2, 3,4,5,6,7,8,9,10,15] ):
    """
    Estimate the total number of people in the image by dividing it into blocks and summing the predicted counts.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): The trained CLIP model.
        text_features (torch.Tensor): Precomputed text features for crowd count categories.
        preprocess (callable): Preprocessing function for image blocks.
        block_size (int): Size of each block (default: 32).
        rep_counts (list): Representative counts for each category (default: [0, 2, 7.5, 15]).

    Returns:
        float: Estimated total number of people in the image.
    """
    # Load and pad the image
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    pw = ((w + block_size - 1) // block_size) * block_size  # Pad width to multiple of block_size
    ph = ((h + block_size - 1) // block_size) * block_size  # Pad height to multiple of block_size
    padded_img = Image.new('RGB', (pw, ph), (0, 0, 0))
    padded_img.paste(img, (0, 0))

    # Calculate number of blocks
    num_blocks_h = ph // block_size
    num_blocks_w = pw // block_size
    total_pred_count = 0

    # Process each block
    with torch.no_grad():
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # Extract block
                block = padded_img.crop((j * block_size, i * block_size, (j + 1) * block_size, (i + 1) * block_size))
                # Preprocess block for CLIP (resize to 224x224 and normalize)
                block = preprocess(block).unsqueeze(0).to(device)
                # Extract image features
                image_features = model.encode_image(block)
                normalized_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # Compute logits and probabilities
                logits = 100.0 * (normalized_image_features @ text_features.T)

                probs = logits.softmax(dim=-1).cpu().numpy()[0]
                # Estimate count for this block
                estimated_count = sum(p * rc for p, rc in zip(probs, rep_counts))
                total_pred_count += estimated_count

    return total_pred_count


def get_ground_truth_count(gt_path):
    """
    Load the ground truth number of people from the .mat file.

    Args:
        gt_path (str): Path to the ground truth .mat file.

    Returns:
        int: Number of people in the image.
    """
    mat = loadmat(gt_path)
    locations = mat['image_info']['location']
    locations = [(float(p[0]), float(p[1])) for p in locations]
    return len(locations)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a crowd counting model on ShanghaiTech Part A test data.")
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file', default="C:/Users/admin\Documents\GitHub\PSDNN\checkpoints/model_epoch_4.pth")
    args = parser.parse_args()

    # Load the CLIP model and preprocessing function
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Define text descriptions and compute text features
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
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Define test data directories
    test_image_dir = 'C:/Users/admin/Documents/GitHub/PSDNN/datasets/partA/test_data/images'
    test_gt_dir = 'C:/Users/admin/Documents/GitHub/PSDNN/datasets/partA/test_data/ground_truth'

    # Get sorted list of test image files
    test_image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.jpg')])

    # Lists to store predictions and ground truth
    predicted_counts = []
    ground_truth_counts = []

    # Evaluate each test image
    for img_file in tqdm(test_image_files, desc="Evaluating"):
        img_path = os.path.join(test_image_dir, img_file)
        gt_file = 'GT_' + img_file.replace('.jpg', '.mat')
        gt_path = os.path.join(test_gt_dir, gt_file)

        # Get predicted and ground truth counts
        pred_count = get_predicted_count(img_path, model, text_features, preprocess)
        gt_count = get_ground_truth_count(gt_path)

        # Store results
        predicted_counts.append(pred_count)
        ground_truth_counts.append(gt_count)

        # Display progress
        print(f"Image: {img_file}, Predicted: {pred_count:.2f}, Ground Truth: {gt_count}")

    # Convert to numpy arrays for metric computation
    predicted_counts = np.array(predicted_counts)
    ground_truth_counts = np.array(ground_truth_counts)

    # Compute MAE and MAPE
    mae = np.mean(np.abs(predicted_counts - ground_truth_counts))
    mape = np.mean(np.abs(predicted_counts - ground_truth_counts) / ground_truth_counts) * 100

    # Output final metrics
    print(f"\nEvaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")