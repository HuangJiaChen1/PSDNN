import cv2
import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import timm
from tqdm import tqdm

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Basic transforms
to_tensor = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


# -----------------------------------------------------------------------------
# Helper function: predict_full_image
# -----------------------------------------------------------------------------
def predict_full_image(model, image, device):
    """
    Splits a full image (448x672) into 6 fixed 224x224 patches,
    runs each patch through the model, and stitches the 28x28 token outputs
    into a full prediction.

    Args:
        model: the trained CrowdCountingSwin model.
        image: a tensor of shape [3, 448, 672].
        device: torch device.

    Returns:
        full_output: tensor of shape [56*84, 5].
    """
    patches = []
    for i in range(2):  # 2 rows
        for j in range(3):  # 3 columns
            patch = image[:, i * 224:(i + 1) * 224, j * 224:(j + 1) * 224]
            patches.append(patch)
    patch_batch = torch.stack(patches, dim=0).to(device)  # [6, 3, 224, 224]
    patch_logits = model(patch_batch)  # [6, 784, 5] (each patch gives 28x28 tokens)
    num_classes = patch_logits.shape[-1]
    # Reshape each patch's output to [28, 28, num_classes]
    patch_outputs = patch_logits.reshape(6, 28, 28, num_classes)  # [6, 28, 28, 5]
    top_row = torch.cat((patch_outputs[0], patch_outputs[1], patch_outputs[2]), dim=1)  # [28, 84, 5]
    bottom_row = torch.cat((patch_outputs[3], patch_outputs[4], patch_outputs[5]), dim=1)  # [28, 84, 5]
    full_output = torch.cat((top_row, bottom_row), dim=0)  # [56, 84, 5]
    full_output = full_output.reshape(-1, num_classes)  # [56*84, 5]
    return full_output


# -----------------------------------------------------------------------------
# Model Definition (same as before)
# -----------------------------------------------------------------------------
class LearnableUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Upsample from 7x7 to 28x28 using a transposed convolution.
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=4, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CrowdCountingSwin(nn.Module):
    def __init__(self):
        super().__init__()
        # Model is trained on 224x224 patches.
        self.base_model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            img_size=(224, 224)
        )
        self.head = nn.Linear(self.base_model.num_features, 5)
        self.upsample = LearnableUpsample(in_channels=self.base_model.num_features)

    def forward(self, x):
        B = x.size(0)
        # Expected input: [B, 3, 224, 224]. Output from backbone: [B, 7, 7, 1024].
        x = self.base_model.forward_features(x)
        x = x.permute(0, 3, 1, 2)  # [B, 1024, 7, 7]
        x = self.upsample(x)  # [B, 1024, 28, 28]
        x = x.permute(0, 2, 3, 1)  # [B, 28, 28, 1024]
        x = x.reshape(B, 28 * 28, -1)  # [B, 784, 1024]
        logits = self.head(x)  # [B, 784, 5]
        return logits


# Load the trained model checkpoint.
model = CrowdCountingSwin().to(device)
ckpt_files = glob.glob("best_vit_model_*_*.pth")
if ckpt_files:
    ckpt = ckpt_files[0]
    print(f"Loading checkpoint from {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
else:
    print("No checkpoint found!")
model.eval()


# -----------------------------------------------------------------------------
# Process a Single Frame
# -----------------------------------------------------------------------------
def process_frame(model, frame, device, density_threshold=0.1):
    """
    Processes one video frame:
      - Resizes to 448x672.
      - Uses the model to predict the density map.
      - Overlays the total count at the top left.
      - Draws an 8x8 grid.
      - Highlights grid cells (in green) where the predicted density exceeds density_threshold.

    Args:
        model: the trained CrowdCountingSwin model.
        frame: a frame as a BGR numpy array (from cv2).
        device: torch device.
        density_threshold: threshold for highlighting grid cells.

    Returns:
        vis_frame: the processed frame as a BGR image (numpy array).
    """
    # Convert frame (BGR) to RGB and then to PIL image.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    # Resize to full evaluation size: 672 (width) x 448 (height)
    pil_img = pil_img.resize((672, 448), Image.BILINEAR)
    # Convert to tensor and normalize.
    img_tensor = to_tensor(pil_img)
    img_tensor = normalize(img_tensor)

    # Run prediction: get full density prediction.
    with torch.no_grad():
        output = predict_full_image(model, img_tensor, device)  # [56*84, 5]
        probs = torch.softmax(output, dim=-1)
        counts_range = torch.arange(5, dtype=torch.float32, device=device)
        density = (probs * counts_range).sum(dim=-1)  # [56*84]
        density_map = density.cpu().numpy().reshape(56, 84)
        total_count = density_map.sum()

    # Prepare visualization: working on full image (448x672).
    vis_frame = np.array(pil_img)  # RGB
    vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

    # Draw grid lines every 8 pixels.
    H, W = 448, 672
    cell_h, cell_w = 8, 8
    for x in range(0, W + 1, cell_w):
        cv2.line(vis_frame, (x, 0), (x, H), (0, 0, 255), 1)
    for y in range(0, H + 1, cell_h):
        cv2.line(vis_frame, (0, y), (W, y), (0, 0, 255), 1)

    # Highlight cells where predicted density exceeds density_threshold.
    for i in range(56):
        for j in range(84):
            if density_map[i, j] > density_threshold:
                x = j * cell_w
                y = i * cell_h
                cv2.rectangle(vis_frame, (x, y), (x + cell_w, y + cell_h), (0, 255, 0), 1)

    # Overlay total count on top left.
    cv2.putText(vis_frame, f"Count: {total_count:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return vis_frame


# -----------------------------------------------------------------------------
# Process Video
# -----------------------------------------------------------------------------
def process_video(model, input_video_path, output_video_path, device, density_threshold=0.1):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file!")
        return

    # Get video properties.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # We will output frames at the evaluation size: 672x448.
    out_width, out_height = 672, 448
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    pbar = tqdm(total=frame_count, desc="Processing Video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(model, frame, device, density_threshold)
        out.write(processed_frame)
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    input_video_path = "10 to 20 people.MOV"  # Replace with your input video file path.
    output_video_path = "output_video.mp4"
    process_video(model, input_video_path, output_video_path, device, density_threshold=0.1)
