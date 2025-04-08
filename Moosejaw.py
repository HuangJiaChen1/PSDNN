import cv2
import torch
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed

# Import necessary modules from your existing code
from Blocks import Block
from pos_embed import get_2d_sincos_pos_embed
from YCV import CACVIT  # Assuming your model class is in a file called CACVIT.py


def process_video(video_path, output_path, model, device, transform, exemplar_transform):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    exemplar_dir = "NWPU_60/examplars"  # Change to your exemplars directory

    # Load exemplars (you'll need to replace this with your actual exemplars)
    exemplars = load_exemplars(exemplar_dir, num_exemplars=3, transform=exemplar_transform)

    # Set scales for exemplars (adjust as needed)
    scales = torch.ones((1, 3, 2), dtype=torch.float32) * (64 / max(width, height))

    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Process the frame
        with torch.no_grad():
            # Transform image
            img_tensor = transform(pil_image).unsqueeze(0).to(device)
            exemplars_tensor = exemplars.to(device)
            scales_tensor = scales.to(device)

            # Get model prediction
            inputs = [img_tensor, exemplars_tensor, scales_tensor]
            pred_density = model(inputs)

            # Calculate count from density map
            pred_count = int(round(pred_density.sum().item() / 360))

            # Convert density map to heatmap for visualization
            density_map = pred_density.squeeze(0).cpu().numpy()
            density_map_normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
            density_map_normalized = cv2.resize(density_map_normalized, (width, height))

            # Convert to colormap
            heatmap = cv2.applyColorMap((density_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Blend original frame with heatmap
            alpha = 0.5  # Transparency factor
            blended = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

            # Add count text to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blended, f"Count: {pred_count}", (30, 60), font, 2, (255, 255, 255), 5)
            cv2.putText(blended, f"Count: {pred_count}", (30, 60), font, 2, (0, 0, 0), 2)

            # Write the frame to output video
            out.write(blended)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}")


def load_exemplars(exemplar_dir, num_exemplars=3, transform=None):
    """
    Load exemplar images for the crowd counting model.

    Args:
        exemplar_dir: Directory containing exemplar images
        num_exemplars: Number of exemplars to load
        transform: Transform to apply to exemplars

    Returns:
        Tensor containing exemplar images
    """
    # Replace this with your actual loading logic
    # This is a placeholder implementation
    import os
    from PIL import Image

    exemplar_files = os.listdir(exemplar_dir)[:num_exemplars]
    exemplars = []

    for file in exemplar_files:
        try:
            ex_path = os.path.join(exemplar_dir, file)
            ex = Image.open(ex_path).convert('RGB')
            if transform:
                ex = transform(ex)
            exemplars.append(ex)
        except Exception as e:
            print(f"Error loading exemplar {file}: {e}")

    # If we don't have enough exemplars, create dummy ones
    if len(exemplars) < num_exemplars:
        dummy = torch.zeros((3, 64, 64))
        while len(exemplars) < num_exemplars:
            exemplars.append(dummy)

    # Stack exemplars
    return torch.stack(exemplars).unsqueeze(0)  # Add batch dimension


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define transforms (same as in your training code)
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

    # Initialize model
    model = CACVIT(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                   decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                   mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    # Load model weights
    checkpoint = torch.load("YCV93.pth")
    # checkpoint = torch.load("YCV_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load("YCV_best.pth", map_location=device))
    model = model.to(device)
    model.eval()


    # Define input and output paths
    video_path = "MJ2.mov"  # Change this to your input video path
    output_path = "output_crowd_counting.mp4"
    exemplar_dir = "NWPU_60/examplars"  # Change to your exemplars directory

    # Process the video
    process_video(video_path, output_path, model, device, transform, exemplar_transform)


if __name__ == "__main__":
    main()