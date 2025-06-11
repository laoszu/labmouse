import torch.nn as nn
import torchvision.models as models

class PainClassifier(nn.Module):
    def __init__(self, num_classes=3):  # 0, 1, 2 pain levels
        """
        Initializes the PainClassifier model.
        Args:
            num_classes: number of pain levels to classify (default 3)
        """
        super(PainClassifier, self).__init__()
        
        self.backbone = models.resnet18(pretrained=True)
        
        # last fully connected layer is replaced with a new one that outputs num_classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)