import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def train_model():
    dataset_root = r"D:\VLM\datasets\psna_bus_classifier"
    model_save_dir = r"D:\VLM\models\psna_bus_classifier"
    os.makedirs(model_save_dir, exist_ok=True)

    class_mapping = {"OTHER_VEHICLE": 0, "PSNA_BUS": 1}
    with open(os.path.join(model_save_dir, "classes.json"), "w") as f:
        json.dump(class_mapping, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, "train"), transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, "val"), transform=eval_transforms
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, "test"), transform=eval_transforms
    )

    train_dataset.class_to_idx = class_mapping
    val_dataset.class_to_idx = class_mapping
    test_dataset.class_to_idx = class_mapping

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    labels = [s[1] for s in train_dataset.samples]
    count_0 = labels.count(0)
    count_1 = labels.count(1)
    weight_0 = 1.0 / count_0 if count_0 > 0 else 1.0
    weight_1 = 1.0 / count_1 if count_1 > 0 else 1.0

    samples_weight = np.array([weight_0 if t == 0 else weight_1 for t in labels])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    def train_epoch(model, dataloader, opt, crit):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / total, correct / total

    def eval_model(model, dataloader, crit):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = crit(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return running_loss / total, correct / total, all_labels, all_preds, all_probs

    best_val_acc = 0.0
    training_history = []

    print("\n--- STAGE 1: Head Training (Frozen Backbone) ---")
    for epoch in range(15):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc, _, _, _ = eval_model(model, val_loader, criterion)
        training_history.append(
            {"epoch": epoch + 1, "train_acc": t_acc, "val_acc": v_acc}
        )

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best.pt"))

    print("\n--- STAGE 2: Fine-Tuning (Unfrozen Upper Layers) ---")
    model.load_state_dict(
        torch.load(os.path.join(model_save_dir, "best.pt"), weights_only=True)
    )

    for param in list(model.features.parameters())[-8:]:
        param.requires_grad = True

    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    for epoch in range(10):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer_ft, criterion)
        v_loss, v_acc, _, _, _ = eval_model(model, val_loader, criterion)
        training_history.append(
            {"epoch": epoch + 16, "train_acc": t_acc, "val_acc": v_acc}
        )

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best.pt"))

    model.load_state_dict(
        torch.load(os.path.join(model_save_dir, "best.pt"), weights_only=True)
    )
    print("\n--- TEST SET EVALUATION (Untouched Data) ---")
    te_loss, te_acc, te_labels, te_preds, te_probs = eval_model(
        model, test_loader, criterion
    )

    cm = confusion_matrix(te_labels, te_preds)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

    precision = precision_score(te_labels, te_preds, zero_division=0)
    recall = recall_score(te_labels, te_preds, zero_division=0)
    f1 = f1_score(te_labels, te_preds, zero_division=0)

    print("\nConfusion Matrix (Actual \\ Predicted):")
    print("                 OTHER (0)    PSNA (1)")
    print(f"Actual OTHER (0)   {tn}             {fp}")
    print(f"Actual PSNA (1)    {fn}             {tp}")

    psna_fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0
    other_fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_results = {}
    for t in thresholds:
        t_preds = [1 if p >= t else 0 for p in te_probs]
        t_prec = precision_score(te_labels, t_preds, zero_division=0)
        t_rec = recall_score(te_labels, t_preds, zero_division=0)

        t_cm = confusion_matrix(te_labels, t_preds)
        t_tn, t_fp, t_fn, t_tp = (
            t_cm.ravel() if len(t_cm.ravel()) == 4 else (0, 0, 0, 0)
        )
        t_fpr = (t_fp / (t_fp + t_tn)) if (t_fp + t_tn) > 0 else 0

        thresh_results[str(t)] = {"precision": t_prec, "recall": t_rec, "fpr": t_fpr}

    metrics = {
        "test_accuracy": te_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "psna_fnr": psna_fnr,
        "other_fpr": other_fpr,
        "threshold_analysis": thresh_results,
    }

    with open(os.path.join(model_save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(model_save_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=4)

    with open(os.path.join(model_save_dir, "config.json"), "w") as f:
        json.dump(
            {
                "model_architecture": "MobileNetV3-Small",
                "weights": "DEFAULT",
                "input_size": [224, 224],
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "classes": 2,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    train_model()
