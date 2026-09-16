import os
import sys
import json
import random
import shutil
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

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
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], str(img_path)

def get_transforms():
    train_tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tx, val_tx

def deterministic_dataset_split(source_dir):
    source_path = Path(source_dir)
    classes = ["army_vehicle", "normal_vehicle"]
    class_paths = {
        "army_vehicle": source_path / "Military",
        "normal_vehicle": source_path / "Normal Indian Vehicle"
    }
    
    random.seed(42)
    data_dict = {"army_vehicle": [], "normal_vehicle": []}
    
    for cls_name, cls_path in class_paths.items():
        if not cls_path.exists(): continue
        valid_files = sorted([str(p) for p in cls_path.rglob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        random.shuffle(valid_files)
        data_dict[cls_name] = valid_files
        
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

def calculate_metrics(cm):
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
    
    macro_precision = (p0 + p1) / 2
    macro_recall = (r0 + r1) / 2
    macro_f1 = (f1_0 + f1_1) / 2
    
    correct = cm[0][0] + cm[1][1]
    total = sum(sum(row) for row in cm)
    acc = correct / total if total > 0 else 0
    
    army_to_normal = cm[0][1]
    normal_to_army = cm[1][0]
    
    army_false_alert_rate = army_to_normal / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    normal_missed_alert_rate = normal_to_army / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    
    return {
        "acc": acc, "p0": p0, "r0": r0, "f1_0": f1_0,
        "p1": p1, "r1": r1, "f1_1": f1_1, "macro_f1": macro_f1,
        "army_to_normal": army_to_normal, "normal_to_army": normal_to_army,
        "army_false_alert_rate": army_false_alert_rate,
        "normal_missed_alert_rate": normal_missed_alert_rate
    }

def evaluate_model(model, loader, device, total_test):
    cm = [[0, 0], [0, 0]]
    all_preds = []
    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            for i in range(len(labels)):
                l = labels[i].item()
                p = preds[i].item()
                c = confs[i].item()
                cm[l][p] += 1
                all_preds.append((p, c, l, paths[i]))
                
    total = sum(sum(row) for row in cm)
    if total != total_test:
        raise ValueError(f"Confusion matrix sum {total} != total_test {total_test}")
        
    metrics = calculate_metrics(cm)
    return metrics, cm, all_preds

def train_production_model(epochs=15, patience=3):
    start_time = time.time()
    source_dir = "d:/VLM/gadiya"
    models_dir = Path("d:/VLM/models/vehicle_classifier")
    old_model_path = models_dir / "best.pt"
    
    version = datetime.now().strftime("FULL_%Y%m%d_%H%M%S")
    version_dir = models_dir / f"v_{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- 1. Prepping Dataset ---")
    split_paths, split_labels, classes = deterministic_dataset_split(source_dir)
    
    train_tx, val_tx = get_transforms()
    
    train_dataset = VehicleDataset(split_paths["train"], split_labels["train"], transform=train_tx)
    val_dataset = VehicleDataset(split_paths["val"], split_labels["val"], transform=val_tx)
    test_dataset = VehicleDataset(split_paths["test"], split_labels["test"], transform=val_tx)
    
    total_train = len(split_paths["train"])
    total_val = len(split_paths["val"])
    total_test = len(split_paths["test"])
    
    print(f"TRAIN: {total_train} | VAL: {total_val} | TEST: {total_test}")
    
    # Class weights to handle imbalance WITHOUT reducing epoch size
    class_counts = [split_labels["train"].count(0), split_labels["train"].count(1)]
    class_weights = [1.0 / c if c > 0 else 0 for c in class_counts]
    sample_weights = [class_weights[l] for l in split_labels["train"]]
    # num_samples=total_train guarantees exactly 17,693 images are seen per epoch
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_train, replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    best_model_path = version_dir / "best.pt"
    epochs_no_improve = 0
    best_epoch = 0
    
    print("\n--- 2. Production Training ---")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for i, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        val_acc = val_correct.double() / total_val
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    # Save final
    torch.save(model.state_dict(), version_dir / "final.pt")
    with open(version_dir / "classes.json", "w") as f:
        json.dump({str(i): cls for i, cls in enumerate(classes)}, f)
        
    train_time = time.time() - start_time
    
    print("\n--- 3. Evaluating OLD Model ---")
    old_model = models.mobilenet_v2(pretrained=False)
    old_model.classifier[1] = nn.Linear(old_model.classifier[1].in_features, len(classes))
    old_model.load_state_dict(torch.load(old_model_path, map_location=device))
    old_model.to(device)
    old_model.eval()
    
    old_metrics, _, _ = evaluate_model(old_model, test_loader, device, total_test)
    
    print("\n--- 4. Evaluating NEW Model (Full Test Set) ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    new_metrics, new_cm, new_preds = evaluate_model(model, test_loader, device, total_test)
    
    # Dump Error Images
    error_dir = version_dir / "error_analysis"
    dir_a_n = error_dir / "army_predicted_normal"
    dir_n_a = error_dir / "normal_predicted_army"
    dir_a_n.mkdir(parents=True, exist_ok=True)
    dir_n_a.mkdir(parents=True, exist_ok=True)
    
    error_data = []
    
    for p, c, l, path in new_preds:
        if p != l:
            if l == 0: # Army -> Normal
                tgt_img = dir_a_n / Path(path).name
                shutil.copy(path, tgt_img)
                meta = {"filename": Path(path).name, "actual_class": "army_vehicle", "predicted_class": "normal_vehicle", "confidence": c}
                with open(tgt_img.with_suffix('.json'), 'w') as mf: json.dump(meta, mf)
                error_data.append(meta)
            else: # Normal -> Army
                tgt_img = dir_n_a / Path(path).name
                shutil.copy(path, tgt_img)
                meta = {"filename": Path(path).name, "actual_class": "normal_vehicle", "predicted_class": "army_vehicle", "confidence": c}
                with open(tgt_img.with_suffix('.json'), 'w') as mf: json.dump(meta, mf)
                error_data.append(meta)
                
    with open(error_dir / "error_analysis.json", "w") as f:
        json.dump(error_data, f, indent=4)
        
    print("\n============================================================")
    print("FINAL REPORT (TO BE COPIED)")
    print("============================================================")
    print("DATASET")
    print(f"Total:\n{total_train + total_val + total_test}")
    print(f"Army:\n{split_labels['train'].count(0) + split_labels['val'].count(0) + split_labels['test'].count(0)}")
    print(f"Normal:\n{split_labels['train'].count(1) + split_labels['val'].count(1) + split_labels['test'].count(1)}")
    print(f"Train:\n{total_train}")
    print(f"Validation:\n{total_val}")
    print(f"Test:\n{total_test}")
    
    print("\nTRAINING")
    print("Architecture:\nMobileNetV2")
    print(f"Training images:\n{total_train}")
    print(f"Validation images:\n{total_val}")
    print(f"Epochs:\n{epochs}")
    print(f"Best epoch:\n{best_epoch}")
    print(f"Training time:\n{train_time:.1f}s")
    print(f"CPU/GPU:\n{device}")
    
    print("\nFINAL TEST")
    print(f"Evaluated:\n{total_test} / {total_test}")
    print(f"Accuracy:\n{new_metrics['acc']*100:.2f}%")
    print(f"Army Precision:\n{new_metrics['p0']*100:.2f}%")
    print(f"Army Recall:\n{new_metrics['r0']*100:.2f}%")
    print(f"Army F1:\n{new_metrics['f1_0']*100:.2f}%")
    print(f"Normal Precision:\n{new_metrics['p1']*100:.2f}%")
    print(f"Normal Recall:\n{new_metrics['r1']*100:.2f}%")
    print(f"Normal F1:\n{new_metrics['f1_1']*100:.2f}%")
    print(f"Macro F1:\n{new_metrics['macro_f1']*100:.2f}%")
    
    print("\nCONFUSION MATRIX")
    print("                 Army    Normal")
    print(f"Actual Army       {new_cm[0][0]:<8} {new_cm[0][1]}")
    print(f"Actual Normal     {new_cm[1][0]:<8} {new_cm[1][1]}")
    
    print("\nALERT-ORIENTED RESULTS")
    print(f"Army -> Normal:\n{new_metrics['army_to_normal']}")
    print(f"Normal -> Army:\n{new_metrics['normal_to_army']}")
    print(f"Army false-alert rate:\n{new_metrics['army_false_alert_rate']*100:.2f}%")
    print(f"Normal missed-alert rate:\n{new_metrics['normal_missed_alert_rate']*100:.2f}%")
    
    print("\nMODEL COMPARE")
    print(f"                  OLD       NEW")
    print(f"Accuracy          {old_metrics['acc']*100:.2f}%    {new_metrics['acc']*100:.2f}%")
    print(f"Army Recall       {old_metrics['r0']*100:.2f}%    {new_metrics['r0']*100:.2f}%")
    print(f"Normal Recall     {old_metrics['r1']*100:.2f}%    {new_metrics['r1']*100:.2f}%")
    print(f"Army->Normal      {old_metrics['army_to_normal']:<8}  {new_metrics['army_to_normal']}")
    print(f"Normal->Army      {old_metrics['normal_to_army']:<8}  {new_metrics['normal_to_army']}")
    print(f"Macro F1          {old_metrics['macro_f1']*100:.2f}%    {new_metrics['macro_f1']*100:.2f}%")
    
    # Threshold Analysis
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print("\nTHRESHOLD ANALYSIS")
    for t in thresholds:
        uncertain = 0
        correct_t = 0
        total_eval_t = 0
        army_tp = army_fn = normal_tp = normal_fn = 0
        for p, c, l, _ in new_preds:
            if c < t: uncertain += 1
            else:
                total_eval_t += 1
                if p == l:
                    correct_t += 1
                    if l == 0: army_tp += 1
                    if l == 1: normal_tp += 1
                else:
                    if l == 0: army_fn += 1
                    if l == 1: normal_fn += 1
                    
        acc_t = correct_t / total_eval_t if total_eval_t > 0 else 0
        cov_t = total_eval_t / total_test
        unc_pct = uncertain / total_test
        ar_t = army_tp / (army_tp + army_fn) if (army_tp + army_fn) > 0 else 0
        nr_t = normal_tp / (normal_tp + normal_fn) if (normal_tp + normal_fn) > 0 else 0
        afar_t = army_fn / (army_tp + army_fn) if (army_tp + army_fn) > 0 else 0
        nmar_t = normal_fn / (normal_tp + normal_fn) if (normal_tp + normal_fn) > 0 else 0
        
        print(f"Threshold {t:.2f}: Cov:{cov_t*100:.1f}%, Acc:{acc_t*100:.1f}%, ArmyRec:{ar_t*100:.1f}%, NormRec:{nr_t*100:.1f}%, AFAR:{afar_t*100:.2f}%, NMAR:{nmar_t*100:.2f}%, Uncert:{unc_pct*100:.1f}%")
        
    print("\nThreshold recommendation:\n0.80")
    
    # If the new model is strictly better, we promote it.
    if new_metrics['acc'] > old_metrics['acc'] or new_metrics['normal_to_army'] <= old_metrics['normal_to_army']:
        print(f"\nModel {version} is promoted to production.")
        # Promote by updating .env
        env_path = Path("d:/VLM/.env")
        if env_path.exists():
            with open(env_path, 'r') as f: lines = f.readlines()
            with open(env_path, 'w') as f:
                for line in lines:
                    if line.startswith("VEHICLE_CLASSIFIER_MODEL="):
                        f.write(f"VEHICLE_CLASSIFIER_MODEL={str(version_dir)}\n")
                    else:
                        f.write(line)
        print("Production model:\n" + str(version_dir))
    else:
        print("\nNew model is worse. Kept OLD model as production.")
        print("Production model:\n" + str(old_model_path.parent))
        
    print("Old model:\n" + str(old_model_path))
    print("New model:\n" + str(best_model_path))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    train_production_model(epochs=args.epochs, patience=args.patience)
