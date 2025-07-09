import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from PIL import Image
import process_json_labels as pj

# ----------------------------
# 1. Define the Custom Dataset
# ----------------------------
class ObjectDetectionDataset(Dataset):
    def __init__(self, image_paths, annotations, transform, grid_size = 128, num_boxes =2, num_classes=75):
        """
            Args:
            - image_paths: List of image file paths.
            - annotations: List of lists, where each sublist contains (x, y, w, h, class_id) tuples.
            - grid_size: Number of grid cells (e.g., 7 for YOLO).
            - num_boxes: Number of bounding boxes per cell.
            - num_classes: Number of object classes.
            - transform: Optional image transform function.
        """
        self.image_paths = image_paths
        self.annotations = annotations
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.transform = transform
        self.width = None
        self.height = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = self.load_image(self.image_paths[idx])  # Implement this function to load images
        bboxes =  self.annotations[idx]  # List of (x, y, w, h, class_id)
        #print(f"bboxes: {bboxes} at idx {idx}")
        
        # Convert bounding boxes into the YOLO-style target tensor
        target= generate_windowed_targets(bboxes, self.height, self.width)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

    def load_image(self, path):
        # Implement this based on your image format (e.g., PIL, OpenCV, torchvision)
        img = Image.open(path).convert("RGB")
        width, height = img.size
        self.width = width
        self.height = height
        return img  # Dummy example for now

# ----------------------------
# 2. Define the Model
# ----------------------------
class SimpleObjectDetector(nn.Module):
    def __init__(self, grid_size=128, num_boxes=2, num_classes=75, overlap_factor=0.1):
        super(SimpleObjectDetector, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.overlap_factor = overlap_factor  # Determines how much grids overlap
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.backbone2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        self.output_size = (5 + num_classes)
        self.output_size2 = num_boxes * (5 + num_classes)  # just per patch

        self.pool1 = nn.AdaptiveAvgPool2d((7, 7)) 
        self.pool2a = nn.AdaptiveAvgPool2d((128, 128)) # correspond to grid size or window size
        self.pool2 = nn.AdaptiveAvgPool2d((64, 64))

        # define fully connected layers assuming output 128
        self.fc1 = nn.Linear(128, self.output_size)
        self.fc2 = nn.Linear(128 * grid_size * grid_size, self.output_size2)

    
    def _extract_sliding_windows(self, feature_map, window_size, overlap_factor):
        """
        Extracts overlapping feature map windows.
        """
        batch_size, channels, height, width = feature_map.shape
        stride = int(window_size * (1 - overlap_factor))  # Step size based on overlap

        # Ensure at least one stride step
        stride = max(1, stride)

        # Unfold extracts patches with the given stride
        unfolded = feature_map.unfold(2, window_size, stride).unfold(3, window_size, stride)
        unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()

        # Reshape into (batch, num_windows, channels, window_size, window_size)
        num_windows_h, num_windows_w = unfolded.shape[1], unfolded.shape[2]
        #print(f"num_windows_h: {num_windows_h}")
        #print(f"num_windows_w: {num_windows_w}")
        windows = unfolded.view(batch_size, num_windows_h * num_windows_w, channels, window_size, window_size)

        return windows  # (batch, num_windows, channels, window_size, window_size)

    def _run_detection_head(self, window):
        """
        Runs detection on a given window.
        """
        batch_size = window.size(0)
        window = window.transpose(2, 1)
        window = self.backbone2(window)
        window = self.pool2a(window)
        window = self.pool2(window)
        window = window.view(batch_size, -1)  # Flatten window features
        return self.fc2(window)  # Output (class + bbox)

    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)
        feature_map = x
        # Extract overlapping windows
        windows = self._extract_sliding_windows(feature_map, self.grid_size, self.overlap_factor)

        # Run detection on each window
        detections = []
        for window in windows:
            det = self._run_detection_head(window)
            detections.append(det)

        # Merge overlapping detections
        final_preds = torch.cat(detections, dim=0)
        return final_preds
# ----------------------------
# 3. Define Loss Function
# ----------------------------

def generate_windowed_targets(
    bboxes_list, orig_height, orig_width, image_size=224, window_size=128, overlap_factor=0.1,
    num_classes=75, num_boxes=2
): # window_size 32, 64
    """
    Generate target tensors that align with sliding window patches and allow multiple boxes per window.
    
    Args:
    - bboxes_list: list of length B (batch), where each element is a list of (x, y, w, h, class_id), normalized [0, 1]
    - image_size: int, assuming square image
    - window_size: size of each sliding window
    - overlap_factor: amount of overlap between windows
    - num_classes: total number of object classes
    - num_boxes: number of boxes to assign per window

    Returns:
    - target_tensor: shape [B * num_windows, num_boxes * (5 + num_classes)]
    """
    stride = int(window_size * (1 - overlap_factor))

    stride = max(1, stride)
    h_ratio = orig_height/image_size
    w_ratio = orig_width/image_size
    num_windows_per_axis = (image_size - window_size) // stride + 1
    target_tensor = []

    for bboxes in [bboxes_list]:  # Loop over batch
        for y_idx in range(num_windows_per_axis):
            for x_idx in range(num_windows_per_axis):
                # Window bounds in normalized coordinates from 0 to 1 relative to image size
                x_start = (x_idx * stride) / image_size
                #print(f"x_start: {x_start}")
                y_start = (y_idx * stride) / image_size
                #print(f"y_start: {y_start}")
                # converts back to pixel-space (x_start * image_size), adds the window size, and then renormalizes to [0,1]
                x_end = (x_start * image_size + window_size) / image_size
                #print(f"x_end: {x_end}")
                y_end = (y_start * image_size + window_size) / image_size
                #print(f"y_end: {y_end}")

                # Initialize empty window target
                window_target = torch.zeros(num_boxes * (5 + num_classes))

                box_count = 0
                for bbox in bboxes:
                    if box_count >= num_boxes:
                        break
                    x, y, w, h, class_id = bbox
                    x = x/w_ratio
                    #print(f"x:{x}")
                    y = y/h_ratio
                    #print(f"y:{y}")
                    w = w/w_ratio
                    h = h/h_ratio
                    x_norm = x/image_size
                    y_norm = y/image_size
                    if x_start <= x_norm <= x_end and y_start <= y_norm <= y_end:
                        start_idx = box_count * (5 + num_classes)
                        window_target[start_idx:start_idx+4] = torch.tensor([x, y, w, h])
                        #print(f"window_target bbox: {window_target[start_idx:start_idx+4]}")
                        window_target[start_idx + 4] = 1.0  # Confidence
                        class_offset = start_idx + 5 + (class_id - 1)
                        if class_offset < len(window_target):  # Prevent indexing error
                            window_target[class_offset] = 1.0
                        box_count += 1

                target_tensor.append(window_target)

    return torch.stack(target_tensor)  # Shape: [B * num_windows, num_boxes * (5 + num_classes)]

def loss_function(pred, target, num_boxes=2, num_classes=75):
    target = target.view(-1, target.shape[-1])

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    grid_size = pred.shape[1]  # Assuming shape [B, 7, 7, num_boxes * (5 + num_classes)]
    total_attrs = 5 + num_classes
    device = pred.device

    bbox_loss = 0.0
    conf_loss = 0.0
    class_loss = 0.0

    for b in range(num_boxes):
        start = b * total_attrs
        end = start + total_attrs

        pred_slice = pred[..., start:end]
        target_slice = target[..., start:end]

        pred_box = pred_slice[..., :4]
        target_box = target_slice[..., :4]

        # Objectness, probability of ANY class. Region of interest to reduce clutter.
        pred_conf = pred_slice[..., 4]
        target_conf = target_slice[..., 4]

        #Probabilities of a particular class
        pred_class = pred_slice[..., 5:]
        target_class = target_slice[..., 5:]

        conf_loss += bce_loss(pred_conf, target_conf)
        mask = target_conf == 1  # Only apply class loss where there's an object
        if mask.any():
            class_loss += ce_loss(pred_class, target_class.argmax(dim=-1))
            bbox_loss += mse_loss(pred_box, target_box)

    total_loss = bbox_loss + conf_loss + class_loss
    return total_loss


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def main():
    train_image_paths, train_labels = pj.parse_annotations('COCO_dog_subset/train/annotations/instances_dog_subset.json', 'COCO_dog_subset/train/images/')
    val_image_paths, val_labels = pj.parse_annotations('COCO_dog_subset/val/annotations/instances_dog_subset.json', 'COCO_dog_subset/val/images/')

    train_dataset = ObjectDetectionDataset(train_image_paths, train_labels, transform=data_transform)
    val_dataset = ObjectDetectionDataset(val_image_paths, val_labels, transform=data_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = SimpleObjectDetector()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 6

    for epoch in range(num_epochs):
        for images, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, targets in val_dataloader:
                outputs = model(images)
                val_loss += loss_function(outputs, targets).item()
            val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    torch.save({
    'model_state_dict': model.state_dict(),
    'overlap_factor': 0.1,
    'num_boxes': 2,
    'num_classes': 75,
    'grid_size': 128,
}, "object_detector_windows_dogs_128.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
