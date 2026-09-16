import os
import sys
import json
import random
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

class VehicleTestDataset(Dataset):
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
        except Exception:
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], str(img_path)

def deterministic_dataset_split(source_dir):
    """Replicates the 70/20/10 split on the full dataset with a fixed seed."""
    source_path = Path(source_dir)
    classes = ["army_vehicle", "normal_vehicle"]
    class_paths = {
        "army_vehicle": source_path / "Military",
        "normal_vehicle": source_path / "Normal Indian Vehicle"
    }
    
    random.seed(42) # Seed to ensure reproducible split
    data_dict = {"army_vehicle": [], "normal_vehicle": []}
    
    for cls_name, cls_path in class_paths.items():
        if not cls_path.exists():
            print(f"Directory {cls_path} does not exist!")
            continue
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
        
    return split_paths, split_labels

def main():
    model_dir = Path("d:/VLM/models/vehicle_classifier")
    best_model_path = model_dir / "best.pt"
    classes_path = model_dir / "classes.json"
    
    if not best_model_path.exists() or not classes_path.exists():
        print("Model files not found.")
        sys.exit(1)
        
    with open(classes_path, "r") as f:
        class_mapping = json.load(f)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_mapping)
    
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model Architecture: MobileNetV2")
    print(f"Model Path: {best_model_path}")
    print(f"Class Mapping: {class_mapping}")
    
    source_dir = "d:/VLM/gadiya"
    split_paths, split_labels = deterministic_dataset_split(source_dir)
    
    test_paths = split_paths["test"]
    test_labels = split_labels["test"]
    total_test = len(test_paths)
    
    print(f"\nExpected: 2539 test images")
    print(f"Actual Test Images Found: {total_test}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = VehicleTestDataset(test_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    error_dir = model_dir / "error_analysis"
    if error_dir.exists():
        shutil.rmtree(error_dir)
    
    dir_a_n = error_dir / "army_predicted_normal"
    dir_n_a = error_dir / "normal_predicted_army"
    dir_a_n.mkdir(parents=True, exist_ok=True)
    dir_n_a.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating every test image exactly once...")
    
    all_preds = []
    all_confs = []
    
    cm = [[0, 0], [0, 0]]
    
    army_to_normal_err = 0
    normal_to_army_err = 0
    
    conf_correct_army = []
    conf_incorrect_army = []
    conf_correct_normal = []
    conf_incorrect_normal = []
    
    evaluated_count = 0
    
    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            for i in range(len(labels)):
                l = labels[i].item()
                p = preds[i].item()
                c = confs[i].item()
                path = paths[i]
                
                all_preds.append((p, c, l))
                cm[l][p] += 1
                
                # Confidence tracking and Error Image Export
                if l == 0: # Actual Army
                    if p == 0:
                        conf_correct_army.append(c)
                    else:
                        conf_incorrect_army.append(c)
                        army_to_normal_err += 1
                        tgt_img = dir_a_n / Path(path).name
                        shutil.copy(path, tgt_img)
                        with open(tgt_img.with_suffix('.json'), 'w') as mf:
                            json.dump({"filename": Path(path).name, "actual": "army", "predicted": "normal", "confidence": c}, mf)
                else: # Actual Normal
                    if p == 1:
                        conf_correct_normal.append(c)
                    else:
                        conf_incorrect_normal.append(c)
                        normal_to_army_err += 1
                        tgt_img = dir_n_a / Path(path).name
                        shutil.copy(path, tgt_img)
                        with open(tgt_img.with_suffix('.json'), 'w') as mf:
                            json.dump({"filename": Path(path).name, "actual": "normal", "predicted": "army", "confidence": c}, mf)
                
                evaluated_count += 1
                
    print(f"\nEvaluated:\n{evaluated_count} / {total_test}")
    
    if (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]) != total_test:
        print("ERROR: Confusion matrix sum does not equal total test images!")
        sys.exit(1)
        
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
    acc = correct / total_test
    
    print(f"\nDATASET")
    print(f"Total:\n{len(split_paths['train']) + len(split_paths['val']) + len(split_paths['test'])}")
    print(f"Army:\n{split_labels['train'].count(0) + split_labels['val'].count(0) + split_labels['test'].count(0)}")
    print(f"Normal:\n{split_labels['train'].count(1) + split_labels['val'].count(1) + split_labels['test'].count(1)}")
    print(f"TRAIN:\n{len(split_paths['train'])}")
    print(f"VALIDATION:\n{len(split_paths['val'])}")
    print(f"TEST:\n{len(split_paths['test'])}")
    
    print("\nFULL TEST RESULTS")
    print(f"Evaluated:\n{evaluated_count} / {total_test}")
    print(f"Accuracy:\n{acc*100:.2f}%")
    print(f"Army Precision:\n{p0*100:.2f}%")
    print(f"Army Recall:\n{r0*100:.2f}%")
    print(f"Army F1:\n{f1_0*100:.2f}%")
    print(f"Normal Precision:\n{p1*100:.2f}%")
    print(f"Normal Recall:\n{r1*100:.2f}%")
    print(f"Normal F1:\n{f1_1*100:.2f}%")
    print(f"Macro F1:\n{macro_f1*100:.2f}%")
    
    print("\nCONFUSION MATRIX")
    print("             Predicted")
    print("           Army    Normal")
    print(f"Actual Army     {cm[0][0]:<8} {cm[0][1]}")
    print(f"Actual Normal   {cm[1][0]:<8} {cm[1][1]}")
    
    print("\nERRORS")
    print(f"Army -> Normal:\n{army_to_normal_err}")
    print(f"Normal -> Army:\n{normal_to_army_err}")
    
    print("\nCONFIDENCE DISTRIBUTION")
    print(f"Correct Army: {sum(conf_correct_army)/len(conf_correct_army) if conf_correct_army else 0:.4f}")
    print(f"Incorrect Army: {sum(conf_incorrect_army)/len(conf_incorrect_army) if conf_incorrect_army else 0:.4f}")
    print(f"Correct Normal: {sum(conf_correct_normal)/len(conf_correct_normal) if conf_correct_normal else 0:.4f}")
    print(f"Incorrect Normal: {sum(conf_incorrect_normal)/len(conf_incorrect_normal) if conf_incorrect_normal else 0:.4f}")
    
    print("\nUNCERTAIN ANALYSIS")
    thresholds = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    
    for t in thresholds:
        uncertain = 0
        correct_t = 0
        total_eval_t = 0
        army_tp = 0
        army_fn = 0
        normal_tp = 0
        normal_fn = 0
        
        for p, c, l in all_preds:
            if c < t:
                uncertain += 1
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
        
        print(f"Threshold {t:.2f}:")
        print(f"Coverage: {cov_t*100:.2f}%")
        print(f"Accuracy: {acc_t*100:.2f}%")
        print(f"Army Recall: {ar_t*100:.2f}%")
        print(f"Normal Recall: {nr_t*100:.2f}%")
        print(f"Uncertain %: {unc_pct*100:.2f}%\n")

if __name__ == "__main__":
    main()
