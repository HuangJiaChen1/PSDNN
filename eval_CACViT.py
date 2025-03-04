import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from CACViT import CrowdCountingDataset, transform, exemplar_transform, CACViT, custom_collate_fn
from ultralytics import YOLO


#############################################
# YOLO-Based Exemplars Extraction
#############################################
def get_exemplars_from_yolo(pil_image, yolo_model, exemplar_transform, conf_threshold=0.5, num_exemplars=3):
    """
    Runs YOLO on the input PIL image to obtain bounding boxes with confidence >= conf_threshold.
    Crops and resizes the regions to 64x64 and applies exemplar_transform.
    """
    results = yolo_model(pil_image)
    # Assuming results[0].boxes.data is a tensor of shape (N, 6): [x1, y1, x2, y2, conf, cls]
    boxes = results[0].boxes.data.cpu().numpy()
    exemplars = []
    for box in boxes:
        if box[4] >= conf_threshold:
            x1, y1, x2, y2 = box[:4]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped = pil_image.crop((x1, y1, x2, y2))
            cropped = cropped.resize((64, 64))
            exemplars.append(cropped)
            if len(exemplars) >= num_exemplars:
                break
    if len(exemplars) < num_exemplars:
        if len(exemplars) == 0:
            dummy_ex = Image.fromarray(np.zeros((64,64,3), dtype=np.uint8))
            exemplars = [dummy_ex] * num_exemplars
        else:
            while len(exemplars) < num_exemplars:
                exemplars.append(exemplars[-1])
    exemplars = torch.stack([exemplar_transform(ex) for ex in exemplars])
    return exemplars
#############################################
# Evaluation Code (MAE and MAPE metrics)
#############################################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths (update these paths as necessary)
    test_img_dir = 'datasets/partA/test_data/images'
    test_gt_dir = 'datasets/partA/test_data/ground_truth'
    exemplars_dir = 'datasets/partA/examplars'

    test_dataset = CrowdCountingDataset(
        test_img_dir, test_gt_dir, exemplars_dir,
        transform=transform, exemplar_transform=exemplar_transform, num_exemplars=3, resize_shape=(384,384)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    # Initialize model and load weights
    model = CACViT(num_exemplars=3, img_size=384, patch_size=16, embed_dim=768).to(device)
    checkpoint = torch.load("cacvit_crowd_counting.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load YOLO model
    yolo_model = YOLO("best.pt")

    total_mae = 0.0
    total_mape = 0.0
    num_samples = 0
    epsilon = 1e-6

    with torch.no_grad():
        for sample in test_loader:
            # sample is a tuple: (image, gt_density, dummy_exemplars, scales, gt_count)
            image, gt_density, _, scales, gt_counts = sample
            # Un-normalize to get a PIL image for YOLO inference
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            img_tensor = image.cpu().squeeze(0)
            disp_img = inv_normalize(img_tensor)
            disp_img = torch.clamp(disp_img, 0, 1)
            pil_image = transforms.ToPILImage()(disp_img)

            # Extract exemplars from YOLO
            exemplars = get_exemplars_from_yolo(pil_image, yolo_model, exemplar_transform, conf_threshold=0.5,
                                                num_exemplars=3)

            image = image.to(device)
            scales = scales.to(device)
            exemplars = exemplars.to(device)
            output_density = model([image, exemplars, scales])  # (1, 384, 384)
            pred_count = output_density.sum().item()
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
        pred_count = output_density_np.sum()
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