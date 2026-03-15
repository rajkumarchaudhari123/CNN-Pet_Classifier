# train.py - Real CNN Training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import PetCNN
import os

# 🔧 Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 🔧 Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 🔧 Dataset Load (Aapko data chahiye!)
# Option A: Kaggle Cats vs Dogs
# Option B: Custom folder structure

# Dummy dataset example (replace with real data):
try:
    train_dataset = datasets.ImageFolder('data/training_set/training_set', transform=train_transform)
    test_dataset = datasets.ImageFolder('data/test_set/test_set', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
except:
    print("❌ Data nahi mila! Folder structure banao:")
    print("data/train/cats/, data/train/dogs/")
    print("data/test/cats/, data/test/dogs/")
    exit()

# 🔧 Model
model = PetCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 🔥 Training Loop
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    
    # Testing
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_pet_model.pth')
        print(f"💾 Best model saved! ({best_acc:.2f}%)")
    
    scheduler.step()

print(f"\n✅ Training complete! Best accuracy: {best_acc:.2f}%")