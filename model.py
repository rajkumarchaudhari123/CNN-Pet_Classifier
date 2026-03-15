# model.py - Pre-trained ResNet (Transfer Learning)

import torch
import torch.nn as nn
import torchvision.models as models

class PetCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(PetCNN, self).__init__()
        
        # 🔥 Pre-trained ResNet18 (ImageNet pe trained)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Last layer freeze karo (pehle se seekha hua)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Sirf last layer train karo
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)  # Cat vs Dog
        )
        
    def forward(self, x):
        return self.backbone(x)

# Test
if __name__ == "__main__":
    model = PetCNN(pretrained=True)
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"✅ Pre-trained model ready! Output: {output.shape}")