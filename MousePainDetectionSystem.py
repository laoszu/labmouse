import json
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from MouseDataLoader import MousePainDataLoader
from MouseEyeDataset import MouseEyeDataset
from SimpleYOLO import SimpleYOLO, SimpleYOLOLoss
from PainClassifier import PainClassifier

class MousePainDetectionSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the detection system. Handles data loading, model training, and inference.
        Args:
            device: 'cuda' for GPU, 'cpu' for CPU
        """
        self.device = device
        self.data_loader = MousePainDataLoader()
        
        self.yolo_model = SimpleYOLO(num_classes=1).to(device)
        self.pain_classifier = PainClassifier(num_classes=3).to(device)
        
        self.classification_loss = nn.CrossEntropyLoss()
        
        self.yolo_optimizer = torch.optim.Adam(self.yolo_model.parameters(), lr=1e-4)
        self.classifier_optimizer = torch.optim.Adam(self.pain_classifier.parameters(), lr=1e-4)
        
    def prepare_data(self, batch_size=8):
        # prepare datasets and data loaders
        all_samples = self.data_loader.get_all_samples()
        
        if not all_samples:
            print("No samples found")
            return
        
        # for detection, training on images with bounding boxes
        self.detection_dataset = MouseEyeDataset(all_samples, img_size=416, mode='detection')
        
        # for classification, training on images with pain levels
        self.classification_dataset = MouseEyeDataset(all_samples, img_size=224, mode='classification')
        self.det_loader = DataLoader(self.detection_dataset, batch_size=batch_size, shuffle=True)
        self.cls_loader = DataLoader(self.classification_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Prepared datasets:")
        print(f"\tDetection: {len(self.detection_dataset)} samples")
        print(f"\tClassification: {len(self.classification_dataset)} samples")
    
    def train_yolo(model, train_loader, device, num_epochs=10):
        """Trenuj model YOLO z właściwą funkcją straty"""
        print("=== TRENING YOLO ===")
        
        os.makedirs("metrics", exist_ok=True)
        os.makedirs("metrics/plots", exist_ok=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = SimpleYOLOLoss()
        all_losses = []
        batch_losses = []

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            epoch_losses = []
            
            print(f"YOLO Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_losses.append(loss.item())
                batch_losses.append(loss.item())
                
                if batch_idx % 5 == 0:
                    print(f"\tBatch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            all_losses.append(avg_loss)
            print(f"YOLO Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        metrics = {
            "epoch_losses": all_losses,
            "batch_losses": batch_losses
        }
        with open("metrics/yolo_metrics.json", "w") as f:
            json.dump(metrics, f)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs+1), all_losses, "o-")
        plt.title("YOLO: Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(batch_losses[:100])
        plt.title("YOLO: Training loss per batch (first 100)")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("metrics/plots/yolo_training_loss.png")
        plt.close()
        
        torch.save(model.state_dict(), "yolo_model.pth")
        print("Model YOLO saved as 'yolo_model.pth'")
        return all_losses

    def train_classifier(model, train_loader, val_loader, device, num_epochs=10):
        """Train the pain classifier"""
        print("=== TRAINING CLASSIFIER ===")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        
        train_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            epoch_train_loss /= len(train_loader)
            epoch_train_acc = 100 * train_correct / train_total
            
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            epoch_val_acc = 100 * val_correct / val_total
            
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)
            val_accuracies.append(epoch_val_acc)

            print(f"Classifier Epoch {epoch+1}/{num_epochs}")
            print(f"\tTrain Loss: {epoch_train_loss:.4f}")
            print(f"\tTrain Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%")

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                torch.save(model.state_dict(), "best_classifier.pth")
        
        metrics = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "best_val_accuracy": best_acc
        }
        with open("metrics/classifier_metrics.json", "w") as f:
            json.dump(metrics, f)
        
        epochs = range(1, num_epochs+1)
        
        plt.figure()
        plt.plot(epochs, train_losses, "bo-")
        plt.title("Classifier: Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("metrics/plots/classifier_loss.png")
        plt.close()
        
        plt.figure()
        plt.plot(epochs, train_accuracies, "bo-", label="Training")
        plt.plot(epochs, val_accuracies, "ro-", label="Validation")
        plt.title("Classifier: Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig("metrics/plots/classifier_accuracy.png")
        plt.close()

        print(f"Best accuracy of classifier: {best_acc:.2f}%")
        return metrics
    
    def predict_pain_level(self, image_path):
        """Predict pain level for an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            self.pain_classifier.eval()
            with torch.no_grad():
                output = self.pain_classifier(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            return predicted_class, probabilities.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Error predicting for {image_path}: {e}")
            return None, None
    
    def save_models(self, classifier_path='pain_classifier.pth'):
        torch.save(self.pain_classifier.state_dict(), classifier_path)
        print(f"Model saved in: {classifier_path}")
    
    def load_models(self, classifier_path='pain_classifier.pth'):
        if os.path.exists(classifier_path):
            self.pain_classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            print(f"Model loaded from: {classifier_path}")
        else:
            print(f"Model couldn't be found: {classifier_path}")