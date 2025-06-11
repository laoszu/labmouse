import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MouseEyeDataset(Dataset):
    def __init__(self, samples, img_size=416, mode='detection'):
        """
        Datasets for mouse eye detection and pain classification.
        Args:
            samples: list of samples from MousePainDataLoader
            img_size: output image size (default 416 for YOLO, 224 for classification)
            mode: 'detection' for YOLO, 'classification' for pain classifier
        """
        self.samples = samples
        self.img_size = img_size
        self.mode = mode
        
        if mode == 'classification':
            self.samples = [s for s in samples if s['pain_level'] is not None]
            print(f"Classification: {len(self.samples)} samples")
        else:
            print(f"Detection: {len(self.samples)} samples")
        
        # transformations, adjusted for each mode
        if mode == 'detection':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        else:  # classification
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Couldn't load {sample['image_path']}: {e}")
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        
        bbox = sample['bbox']
        x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
        
        if self.mode == 'classification':
            cropped_image = image.crop((x, y, x + w, y + h))
            cropped_image = self.transform(cropped_image)
            pain_level = sample['pain_level']
            return cropped_image, pain_level
        
        else:
            # detection mode
            original_w, original_h = image.size
            image = self.transform(image)
            
            # normalize bounding box coordinates
            x_center = (x + w/2) / original_w
            y_center = (y + h/2) / original_h
            bbox_w = w / original_w
            bbox_h = h / original_h
            
            # yolo format: [class, x_center, y_center, width, height]
            # class 0 for all objects, for simplicity
            target = torch.tensor([0, x_center, y_center, bbox_w, bbox_h], dtype=torch.float32)
            
            return image, target