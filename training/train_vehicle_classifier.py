import os
import sys
import time
import json
import random
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

# For metrics (implemented manually)

class VehicleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fallback for corrupt images
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms

def prepare_dataset(source_dir, dest_dir, fast_mode=False):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    classes = ["army_vehicle", "normal_vehicle"]
    class_paths = {
        "army_vehicle": source_path / "Military",
        "normal_vehicle": source_path / "Normal Indian Vehicle"
    }
    
    print("--- 1. Inspecting Dataset ---")
    data_dict = {"army_vehicle": [], "normal_vehicle": []}
    
    for cls_name, cls_path in class_paths.items():
        if not cls_path.exists():
            print(f"Directory {cls_path} does not exist!")
            continue
        valid_files = [str(p) for p in cls_path.rglob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(valid_files)
        
        if fast_mode:
            valid_files = valid_files[:400] # Subsample heavily for testing
            
        data_dict[cls_name] = valid_files
        print(f"Found {len(valid_files)} valid {cls_name} images.")
        
    # Split
    print("\n--- 2. Splitting Dataset (70% Train, 20% Val, 10% Test) ---")
    splits = {"train": (0, 0.7), "val": (0.7, 0.9), "test": (0.9, 1.0)}
    split_paths = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    
    for cls_idx, cls_name in enumerate(classes):
        files = data_dict[cls_name]
        total = len(files)
        
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        split_paths["train"].extend(train_files)
        split_labels["train"].extend([cls_idx] * len(train_files))
        
        split_paths["val"].extend(val_files)
        split_labels["val"].extend([cls_idx] * len(val_files))
        
        split_paths["test"].extend(test_files)
        split_labels["test"].extend([cls_idx] * len(test_files))
        
    return split_paths, split_labels, classes

def train_model(epochs=10, fast_mode=False):
    source_dir = "d:/VLM/gadiya"
    dest_dir = "d:/VLM/gadiya_split"
    models_dir = Path("d:/VLM/models/vehicle_classifier")
    
    # Create timestamped version dir
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = models_dir / f"v_{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset Prep
    split_paths, split_labels, classes = prepare_dataset(source_dir, dest_dir, fast_mode=fast_mode)
    
    train_tx, val_tx = get_transforms()
    
    train_dataset = VehicleDataset(split_paths["train"], split_labels["train"], transform=train_tx)
    val_dataset = VehicleDataset(split_paths["val"], split_labels["val"], transform=val_tx)
    test_dataset = VehicleDataset(split_paths["test"], split_labels["test"], transform=val_tx)
    
    # Class weights for imbalance
    class_counts = [split_labels["train"].count(0), split_labels["train"].count(1)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / c if c > 0 else 0 for c in class_counts]
    sample_weights = [class_weights[l] for l in split_labels["train"]]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    best_model_path = version_dir / "best.pt"
    
    print("\n--- 3. Training Model ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        val_acc = val_correct / len(all_labels) if all_labels else 0
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            
    # Save final model
    torch.save(model.state_dict(), version_dir / "final.pt")
    # Save classes
    with open(version_dir / "classes.json", "w") as f:
        json.dump({str(i): cls for i, cls in enumerate(classes)}, f)
        
    print("\n--- 4. Evaluating on Test Set ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
            
    # Calculate metrics manually to avoid sklearn DLL block
    correct = sum(1 for p, l in zip(test_preds, test_labels) if p == l)
    test_acc = correct / len(test_labels) if test_labels else 0
    
    cm = [[0, 0], [0, 0]]
    for p, l in zip(test_preds, test_labels):
        cm[l][p] += 1
        
    def get_prf(cls_idx):
        tp = cm[cls_idx][cls_idx]
        fp = sum(cm[i][cls_idx] for i in range(2)) - tp
        fn = sum(cm[cls_idx][i] for i in range(2)) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
        
    p0, r0, f1_0 = get_prf(0)
    p1, r1, f1_1 = get_prf(1)
    
    print("\n============================================================")
    print("6. MODEL QUALITY REPORT")
    print("============================================================")
    print(f"Total Dataset: {len(split_paths['train']) + len(split_paths['val']) + len(split_paths['test'])}")
    print(f"Army: {split_labels['train'].count(0) + split_labels['val'].count(0) + split_labels['test'].count(0)}")
    print(f"Normal: {split_labels['train'].count(1) + split_labels['val'].count(1) + split_labels['test'].count(1)}")
    print(f"Training: {len(split_paths['train'])}")
    print(f"Validation: {len(split_paths['val'])}")
    print(f"Test: {len(split_paths['test'])}")
    print(f"Accuracy: {test_acc*100:.2f}%")
    
    print(f"\nArmy Precision: {p0*100:.2f}%")
    print(f"Army Recall: {r0*100:.2f}%")
    print(f"Army F1: {f1_0*100:.2f}%")
    
    print(f"\nNormal Precision: {p1*100:.2f}%")
    print(f"Normal Recall: {r1*100:.2f}%")
    print(f"Normal F1: {f1_1*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Army   Normal")
    print(f"Actual Army    {cm[0][0]:<5}   {cm[0][1]}")
    print(f"Actual Normal  {cm[1][0]:<5}   {cm[1][1]}")
    
    # Save training config & metrics
    end_time = time.time()
    
    config_metadata = {
        "model": "MobileNetV2",
        "classes": classes,
        "epochs": epochs,
        "dataset": "gadiya",
        "image_size": "224x224",
        "timestamp": version,
        "validation_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "inference_device": str(device),
        "training_time_seconds": end_time - start_time
    }
    with open(version_dir / "training_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
        
    metrics = {
        "accuracy": float(test_acc),
        "army": {"precision": float(p0), "recall": float(r0), "f1": float(f1_0)},
        "normal": {"precision": float(p1), "recall": float(r1), "f1": float(f1_1)},
        "confusion_matrix": cm
    }
    with open(version_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Copy best model to the root of models_dir for live inference
    shutil.copy(best_model_path, models_dir / "best.pt")
    shutil.copy(version_dir / "classes.json", models_dir / "classes.json")
    
    print(f"\nModel saved to {best_model_path}")
    print("Main inference endpoints updated with the best model.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fast", action="store_true", help="Run on a tiny subset for testing")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, fast_mode=args.fast)
