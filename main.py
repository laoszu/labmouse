import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

from MouseDataLoader import MousePainDataLoader
from SimpleYOLO import SimpleYOLO, YOLODataset
from MousePainDataset import MousePainDataset
from PainClassifier import PainClassifier

def load_json_safely(json_path):
    try:
        if not os.path.exists(json_path):
            print(f"File doesn't exist: {json_path}")
            return {}
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded from {json_path} - {len(data)} images")
        return data
    except Exception as e:
        print(f"An error ocurred with {json_path}: {e}")
        return {}

def prepare_yolo_dataset(json_data, img_dir, dataset_type='detection'):
    samples = []
    
    for filename, annotations in json_data.items():
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            continue
            
        bboxes = []
        for annotation in annotations:
            if 'Boundingbox' in annotation:
                bbox = annotation['Boundingbox']
                img = Image.open(img_path)
                img_w, img_h = img.size
                
                x = int(bbox['x']) / img_w
                y = int(bbox['y']) / img_h  
                w = int(bbox['w']) / img_w
                h = int(bbox['h']) / img_h

                center_x = x + w/2
                center_y = y + h/2
                class_id = 0 if dataset_type == 'faces' else 1
                bboxes.append([class_id, center_x, center_y, w, h])
        if bboxes:
            samples.append({
                'filename': filename,
                'img_path': img_path,
                'bboxes': bboxes
            })

    return samples

def prepare_classification_dataset(json_data, img_dir, dataset_type='classification'):
    samples = []
    for filename, annotations in json_data.items():
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            continue
        for annotation in annotations:
            sample = {
                'filename': filename,
                'img_path': img_path,
                'annotation': annotation
            }
            if 'GrimaceScale' in annotation:
                sample['pain_level'] = int(annotation['GrimaceScale'])
            else:
                sample['pain_level'] = 0
            if 'Boundingbox' in annotation:
                bbox = annotation['Boundingbox']
                sample['bbox'] = {
                    'x': int(bbox['x']),
                    'y': int(bbox['y']),
                    'w': int(bbox['w']),
                    'h': int(bbox['h'])
                }
            samples.append(sample)
    return samples

def main():
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("metrics/plots", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_paths = {
        'eyes': {
            'json': 'data/eyes_detection_data.json',
            'images': 'Eyes_detection/'
        },
        'faces': {
            'json': 'data/face_detection_data.json', 
            'images': 'Face_detection/'
        }
    }

    data_loader = MousePainDataLoader()
    
    
    yolo_samples = []
    classification_samples = []
    
    for data_type, paths in data_paths.items():
        json_data = load_json_safely(paths['json'])
        
        if json_data:
            yolo_data = prepare_yolo_dataset(json_data, paths['images'], data_type)
            yolo_samples.extend(yolo_data)
            
            if data_type == 'eyes':
                class_data = prepare_classification_dataset(json_data, paths['images'])
                classification_samples.extend(class_data)
    
    transform_train = transforms.Compose([
        transforms.Resize((416, 416)),  # YOLO input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 1: YOLO
    if yolo_samples:
        print("\n=== PREPARING YOLO ===")
        yolo_dataset = YOLODataset(
            yolo_samples, 
            transform=transform_train, 
            grid_size=13, 
            num_anchors=3, 
            num_classes=2
        )
        yolo_loader = DataLoader(yolo_dataset, batch_size=4, shuffle=True)
        
        yolo_model = SimpleYOLO(num_classes=2).to(device)
        yolo_losses = data_loader.train_yolo(yolo_model, yolo_loader, device, num_epochs=5)
    
    # 2: CLASSIFICATION
    if classification_samples:
        print("\n=== PREPARING CLASSIFICATOR ===")
        pain_levels = [s["pain_level"] for s in classification_samples]
        unique_levels = sorted(set(pain_levels))
        print(f"Pain scale: {unique_levels}")
        
        train_samples, val_samples = train_test_split(
            classification_samples, test_size=0.2, random_state=42,
            stratify=pain_levels if len(unique_levels) > 1 else None
        )
        
        train_dataset = MousePainDataset(train_samples, transform_train, use_bbox=True)
        val_dataset = MousePainDataset(val_samples, transform_train, use_bbox=True)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        classifier = PainClassifier(num_classes=len(unique_levels)).to(device)
        classifier_metrics = data_loader.train_classifier(
            classifier, train_loader, val_loader, device, num_epochs=10
        )
    
    final_metrics = {
        "yolo_final_loss": yolo_losses[-1] if yolo_samples else None,
        "classifier_best_val_acc": classifier_metrics["best_val_accuracy"] if classification_samples else None
    }
    
    with open("metrics/final_metrics.json", "w") as f:
        json.dump(final_metrics, f)

if __name__ == "__main__":
    main()