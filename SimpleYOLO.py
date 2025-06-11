import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=1, num_anchors=3, img_size=416):
        """
        Simplified YOLO implementation.
        Args:
            num_classes: number of classes for detection (default 1 for mouse pain)
            num_anchors: number of anchors (default 3)
            img_size: input image size (default 416 for YOLO)
        """
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.img_size = img_size
        
        # backbone - simplified CNN architecture
        self.backbone = nn.Sequential(
            # 416x416 -> 208x208
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 208x208 -> 104x104
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 104x104 -> 52x52
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 52x52 -> 26x26
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 26x26 -> 13x13
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 13x13 -> 13x13 (additional conv layer)
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # detection head - 13x13 grid
        self.detection_head = nn.Conv2d(1024, num_anchors * (5 + num_classes), 1)
        
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        
        batch_size = x.size(0)
        grid_size = detections.size(2)  # should be 13
        
        # reshaping: [B, A*(5+C), G, G] -> [B, A, G, G, 5+C]
        detections = detections.view(
            batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size
        )
        detections = detections.permute(0, 1, 3, 4, 2).contiguous()
        
        return detections

class SimpleYOLOLoss(nn.Module):
    def __init__(self, anchors=None, img_size=416, lambda_coord=5.0, lambda_noobj=0.5):
        super(SimpleYOLOLoss, self).__init__()
        
        # default anchors if not provided (quite small objects) will be tuned later
        if anchors is None:
            self.anchors = torch.tensor([[10, 10], [20, 20], [30, 30]], dtype=torch.float32)
        else:
            self.anchors = torch.tensor(anchors, dtype=torch.float32)
        
        self.img_size = img_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        num_anchors = predictions.size(1)
        device = predictions.device
        
        # move anchors to the same device as predictions
        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)
        
        # get the components of predictions
        pred_boxes = predictions[..., :4]     # x, y, w, h
        pred_conf = predictions[..., 4]       # objectness
        pred_class = predictions[..., 5:]     # classes
        
        # losses calculation
        coord_loss = 0
        conf_loss = 0
        class_loss = 0
        
        for b in range(batch_size):
            target = targets[b]  # [grid_size, grid_size, 7]
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # is the object in this cell?
                    if target[i, j, 4] == 1:  # yes!
                        # find the best anchor
                        target_w = target[i, j, 2] * self.img_size
                        target_h = target[i, j, 3] * self.img_size
                        
                        best_anchor = 0
                        best_iou = 0
                        
                        # find the best anchor
                        for a in range(num_anchors):
                            anchor_w, anchor_h = self.anchors[a]
                            
                            # calculate IoU and find the best anchor
                            min_w = min(target_w, anchor_w)
                            min_h = min(target_h, anchor_h)
                            intersection = min_w * min_h
                            union = target_w * target_h + anchor_w * anchor_h - intersection
                            iou = intersection / (union + 1e-6)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_anchor = a
                        
                        # get the loss for the best anchor
                        pred_box = pred_boxes[b, best_anchor, i, j]
                        pred_c = pred_conf[b, best_anchor, i, j]
                        pred_cls = pred_class[b, best_anchor, i, j]
                        
                        target_box = target[i, j, :4]
                        target_cls = target[i, j, 5:]
                        
                        coord_loss += self.mse_loss(pred_box, target_box)
                        conf_loss += self.bce_loss(pred_c.unsqueeze(0), torch.ones(1, device=device))
                        
                        if pred_cls.size(0) > 0 and target_cls.size(0) > 0:
                            class_loss += self.bce_loss(pred_cls, target_cls)
                    
                    else:
                        # confidence loss for anchors without objects
                        for a in range(num_anchors):
                            pred_c = pred_conf[b, a, i, j]
                            conf_loss += self.lambda_noobj * self.bce_loss(
                                pred_c.unsqueeze(0), torch.zeros(1, device=device)
                            )
        
        total_loss = (
            self.lambda_coord * coord_loss + 
            conf_loss + 
            class_loss
        ) / batch_size
        
        return total_loss

class YOLODataset(Dataset):
    """
    Dataset for YOLO training. Is used to load images and their corresponding bounding boxes.
    """
    def __init__(self, samples, transform=None, grid_size=13, num_anchors=3, num_classes=1):
        self.samples = samples
        self.transform = transform
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['img_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # one target per grid cell, for each cell is stored: [grid_size, grid_size, 5 + num_classes]
        target = torch.zeros(self.grid_size, self.grid_size, 5 + self.num_classes)
        
        for bbox in sample['bboxes']:
            class_id, center_x, center_y, width, height = bbox
            
            # find the grid cell for the center of the bounding box
            grid_x = int(center_x * self.grid_size)
            grid_y = int(center_y * self.grid_size)
            
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                # calculate relative coordinates
                rel_x = center_x * self.grid_size - grid_x
                rel_y = center_y * self.grid_size - grid_y
                
                # store the bounding box in the target
                target[grid_y, grid_x, 0] = rel_x
                target[grid_y, grid_x, 1] = rel_y
                target[grid_y, grid_x, 2] = width
                target[grid_y, grid_x, 3] = height
                target[grid_y, grid_x, 4] = 1.0  # confidence
                
                # one-hot encoding for class
                if int(class_id) < self.num_classes:
                    target[grid_y, grid_x, 5 + int(class_id)] = 1.0
        
        return image, target