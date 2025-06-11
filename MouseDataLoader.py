import json
import os
import glob
from pathlib import Path

class MousePainDataLoader:
    def __init__(self):
        """
        Class to load and process mouse pain detection datasets.
        """
        self.DATA_PATHS = {
            'face': ('data/face_detection_data.json', 'Face_detection/'),
            'eyes': ('data/eyes_detection_data.json', 'Eyes_detection/')
        }
    
    def load_json_data(self, json_path):
        if not os.path.exists(json_path):
            print(f"File doesn't exist: {json_path}")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"File loaded: {json_path}")
            return data
        except Exception as e:
            print(f"File couldn't be loaded: {e}")
            return None
    
    def extract_dataset_info(self, dataset_type='eyes'):
        """Get dataset information for a specific type (eyes or face)"""
        if dataset_type not in self.DATA_PATHS:
            print(f"Unknown dataset type: {dataset_type}")
            return None
        
        json_path, images_folder = self.DATA_PATHS[dataset_type]
        
        data = self.load_json_data(json_path)
        if data is None:
            return None
        
        samples = []
        
        for image_filename, annotations in data.items():
            image_path = os.path.join(images_folder, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Image is missing: {image_path}")
                continue
            
            for annotation in annotations:
                sample = {
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'animal_number': annotation.get('AnimalNumber', '1'),
                    'bbox': annotation['Boundingbox']
                }
                
                # only for eyes dataset (GrimmaceScale)
                if dataset_type == 'eyes' and 'GrimaceScale' in annotation:
                    sample['pain_level'] = int(annotation['GrimaceScale'])
                else:
                    sample['pain_level'] = None
                
                samples.append(sample)
        
        print(f"Found {len(samples)} samples in the {dataset_type}")
        return samples
    
    def get_all_samples(self):
        """Get all samples (images) from both datasets"""
        all_samples = []
        
        for dataset_type in ['eyes', 'face']:
            samples = self.extract_dataset_info(dataset_type)
            if samples:
                for sample in samples:
                    sample['dataset_type'] = dataset_type
                all_samples.extend(samples)
        
        return all_samples