from torch.utils.data import Dataset
from PIL import Image

class MousePainDataset(Dataset):
    def __init__(self, samples, transform=None, use_bbox=False):
        """
        Dataset for mouse pain detection and classification.
        Args:
            samples: list of samples, each sample is a dict with keys 'img_path', 'pain_level', opt. 'bbox'
            transform: some transforms to apply to the images
            use_bbox: whether to use bounding boxes for cropping images
        """
        self.samples = samples
        self.transform = transform
        self.use_bbox = use_bbox
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['img_path']).convert('RGB')
            # cut out bounding box if available
            if self.use_bbox and 'bbox' in sample:
                bbox = sample['bbox']
                image = image.crop((
                    bbox['x'], 
                    bbox['y'], 
                    bbox['x'] + bbox['w'], 
                    bbox['y'] + bbox['h']
                ))
            if self.transform:
                image = self.transform(image)
            pain_level = sample['pain_level']
            return image, pain_level
            
        except Exception as e:
            print(f"Couldn {idx}: {e}")
            # if image loading fails, return a dummy image and label
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, 0