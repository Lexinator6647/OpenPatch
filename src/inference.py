import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import glob
from training import SimpleObjectDetector
import numpy as np
from torchvision.ops import batched_nms
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage

def bbox_iou_chunked(box1, boxes, chunk_size=1024):
    total = boxes.size(0)
    ious = torch.empty(total, device=boxes.device, dtype=boxes.dtype)  # Preallocate once

    for i in range(0, total, chunk_size):
        chunk = boxes[i:i+chunk_size]
        ious[i:i+chunk.size(0)] = bbox_iou(box1, chunk)  # In-place write

    return ious

def bbox_iou(box1, boxes):
    """
    box1: Tensor (4,) [x, y, w, h]
    boxes: Tensor (N, 4)
    returns: Tensor (N,) IoU between box1 and each box in boxes
    """

    box1 = box1.unsqueeze(0)  # shape (1, 4)
    
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = boxes[:, 0] - boxes[:, 2] / 2
    b2_y1 = boxes[:, 1] - boxes[:, 3] / 2
    b2_x2 = boxes[:, 0] + boxes[:, 2] / 2
    b2_y2 = boxes[:, 1] + boxes[:, 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)


    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Areas
    area1 = box1[:, 2] * box1[:, 3]
    area2 = boxes[:, 2] * boxes[:, 3]

    union_area = area1 + area2 - inter_area + 1e-6 # added for numerical stability to avoid division by 0
    return inter_area / union_area

def classwise_nms(predictions, iou_thresh=0.1, conf_thresh=0.1): #conf thresh = 0.45
    """
    Apply class-wise non-maximum suppression.
    Each prediction is (x, y, w, h, confidence, class_id)
    Returns: Tensor of filtered predictions (N, 6)
    """
    predictions = predictions.float()
    filtered_preds = []

    num_classes = int(predictions[:, -1].max().item()) + 1

    for cls in range(num_classes):
        cls_preds = predictions[predictions[:, -1] == cls]
        #print(f"raw cls conf {cls_preds[:, 4]}")
        cls_preds = cls_preds[cls_preds[:, 4] > conf_thresh]

        while cls_preds.size(0):
            # Pick box with highest confidence
            best_idx = torch.argmax(cls_preds[:, 4])
            best = cls_preds[best_idx].unsqueeze(0)
            filtered_preds.append(best.squeeze(0))

            if cls_preds.size(0) == 1:
                break

            # Remove best from rest
            rest = torch.cat([cls_preds[:best_idx], cls_preds[best_idx + 1:]])

            # Compute IoUs in vectorized form
            with torch.no_grad():
                ious = bbox_iou_chunked(best[0, :4], rest[:, :4])

            # Keep boxes with IoU below threshold
            cls_preds = rest[ious < iou_thresh]

    return torch.stack(filtered_preds) if filtered_preds else torch.empty((0, 6))

def get_window_grid_positions(image_size, window_size, stride):
    positions = []
    for y in range(0, image_size - window_size + 1, stride):
        for x in range(0, image_size - window_size + 1, stride):
            positions.append((x, y))
    return positions


def run_inference(model_path, image_dir, num_boxes=2, num_classes=75, image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'object_detector_windows_dogs_128_small.pth' # Testing
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config
    overlap = checkpoint['overlap_factor']
    print(f"overlap factor: {overlap}")
    num_boxes = checkpoint['num_boxes']
    num_classes = checkpoint['num_classes']
    grid_size = checkpoint['grid_size']
    print(f"grid_size: {grid_size}")

    # Recreate the model with correct params
    model = SimpleObjectDetector(num_boxes=num_boxes, num_classes=num_classes,
                                grid_size=grid_size, overlap_factor=overlap)

    # Load weights only
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)
    print("First conv layer weights stats:", model.backbone[0].weight.data.mean(), model.backbone[0].weight.data.std())

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"image paths: {image_paths}")
    for img_path in image_paths:
        print(f"curent img path: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        pil_tensor = transform(img).squeeze(0).to(device)
        print(f"tensor for {img_path}: {img_tensor}")

        with torch.no_grad():
            output = model(img_tensor)
            print("Output stats:", output.min().item(), output.max().item(), output.mean().item())

        print(f"output shape: {output.shape}")
        output = output.view(-1, num_boxes * (5 + num_classes))

        all_preds = []
        # Assume square windows that cover the image with stride derived from grid
        window_size = grid_size
        stride = int(window_size * (1 - overlap))  # Reconstruct stride from overlap factor
        window_positions = get_window_grid_positions(image_size, window_size, stride)
        
        with torch.no_grad():
            for i, window in enumerate(output):
                if i >= len(window_positions):
                    print(f"Warning: More windows than expected for image. Skipping index {i}.")
                    break

                x0, y0 = window_positions[i]  # Window top-left offset
                for b in range(num_boxes):
                    
                    start = b * (5 + num_classes)
                    end = start + (5 + num_classes)

                    slice = window[start:end]         # Get the slice from the current window

                    box = slice[:4].unsqueeze(0)      # Shape [1, 4] for consistency

                    conf = slice[4]                   # Confidence (scalar)

                    class_scores = slice[5:]         # Shape [num_classes]
                    print("class_scores (logits):", class_scores[0])
                    
                    'test training activations'
                    # Replace sigmoid(conf) with just clamped confidence
                    print("Conf logits stats:", conf.min().item(), conf.max().item(), conf.mean().item())
                    conf = torch.sigmoid(conf) # skip sigmoid for testing
                    print("Conf after sigmoid:", conf.min().item(), conf.max().item(), conf.mean().item())
                    conf = torch.clamp(conf, min=1e-4)
                    print("conf with clamped confidence:", conf)
                    
                    objectness = conf  
                    class_probs = torch.softmax(class_scores, dim=-1)  # shape [1, num_classes]
                    class_conf, class_ids = torch.max(class_probs, dim=-1)  # shape [1]
                    
                    final_scores = objectness * class_conf  # shape: [num_windows]

                    total_conf = final_scores

                    # Step 1: Filter
                    # obj_thresh = 0.47
                    obj_thresh = 0.1
                    obj_keep = objectness > obj_thresh

                    if not obj_keep.any():
                        continue

                    #keep = final_scores > 0.4
                    keep = final_scores > 0.05
                    
                    box = box[keep]
                    print(f"box: {box}")

                    w = box[:,:, 2]
                    h = box[:,:, 3]
                    x = box[:,:, 0] + x0
                    y = box[:,:, 1] + y0
                    
                    remapped_boxes = torch.stack([x, y, w, h], dim=1)

                    boxes_kept = remapped_boxes.view(1,-1)
                    print(f"boxes_kept: {boxes_kept}")
                    scores_kept = total_conf[keep].unsqueeze(1)
                    print(f"scores_kept: {scores_kept}")
                    classes_kept = class_ids[keep].float().unsqueeze(1)
                    print(f"classes_kept: {classes_kept}")

                    if remapped_boxes.shape[0] == 0:
                        continue

                    selected = torch.cat([boxes_kept, scores_kept, classes_kept], dim=1)

                    all_preds.append(selected)

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            scores = all_preds[:, 4] * all_preds[:, 5:].max(dim=-1).values
            filtered_preds = classwise_nms(all_preds)
            to_pil = ToPILImage()
            resized_pil = to_pil(pil_tensor)
            print(f"\nResults for {os.path.basename(img_path)}: {all_preds}")
            for pred in filtered_preds:
                x, y, w, h, conf, class_id = pred.tolist()
                print(f"Class {int(class_id)+1}: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}), conf: {conf:.2f}")
                draw = ImageDraw.Draw(resized_pil)
                
                if w < 0:
                    w = 0
                if h <0:
                    h = 0

                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2

                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"{int(class_id)+1}: {conf:.2f}", fill="red")

            resized_pil.save(f"{img_path}_output.jpg")
            
        else:
            print(f"\nNo confident predictions for {os.path.basename(img_path)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--images", type=str, required=True, help="Directory with input images")
    args = parser.parse_args()

    run_inference(args.model, args.images)
