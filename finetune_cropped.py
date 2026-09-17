import os
import time
import copy
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.file_paths[idx]).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return torch.zeros((3, 224, 224)), self.labels[idx]

def train_model():
    source_path = Path("d:/VLM/gadiya_cropped")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_paths = {
        "army_vehicle": source_path / "Military",
        "normal_vehicle": source_path / "Normal Indian Vehicle"
    }

    split_paths = {"train": [], "val": []}
    split_labels = {"train": [], "val": []}
    
    for cls_idx, (cls_name, cls_path) in enumerate(class_paths.items()):
        valid_files = [str(p) for p in cls_path.rglob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        train_idx = int(0.8 * len(valid_files))
        
        split_paths["train"].extend(valid_files[:train_idx])
        split_labels["train"].extend([cls_idx] * train_idx)
        split_paths["val"].extend(valid_files[train_idx:])
        split_labels["val"].extend([cls_idx] * (len(valid_files) - train_idx))
        print(f"Class {cls_name}: {len(valid_files)} images.")

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    datasets = {x: CustomImageDataset(split_paths[x], split_labels[x], transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=32, shuffle=(x == 'train'), num_workers=4, pin_memory=True) 
        for x in ['train', 'val']
    }

    # Load existing best model
    model_dir = Path("d:/VLM/models/vehicle_classifier")
    num_classes = 2
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 3
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(best_model_wts, model_dir / "best.pt")
    print("Fine-tuning complete. Saved new best.pt")

if __name__ == "__main__":
    train_model()
