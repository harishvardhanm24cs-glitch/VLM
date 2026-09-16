import os
import sys
import copy
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import WeightedRandomSampler, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def audit_military_dataset(data_dir):
    logger.info(f"Auditing military dataset at: {data_dir}")
    splits = ['train', 'validation', 'test']
    
    audit_results = {}
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    
    all_files = {'train': set(), 'validation': set(), 'test': set()}
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            logger.error(f"Missing split directory: {split_dir}")
            return False, {}
            
        audit_results[split] = {}
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                files = [f for f in os.listdir(class_dir) if Path(f).suffix.lower() in valid_exts]
                audit_results[split][class_name] = len(files)
                for f in files:
                    all_files[split].add(os.path.join(class_name, f))
                    
    # Leakage check
    train_val_leak = all_files['train'].intersection(all_files['validation'])
    train_test_leak = all_files['train'].intersection(all_files['test'])
    val_test_leak = all_files['validation'].intersection(all_files['test'])
    
    logger.info("--- Data Audit Results ---")
    for split in splits:
        logger.info(f"[{split.upper()}]")
        for cls, count in audit_results[split].items():
            logger.info(f"  {cls}: {count} images")
            
    logger.info("--- Leakage Check ---")
    logger.info(f"Train/Val intersection: {len(train_val_leak)}")
    logger.info(f"Train/Test intersection: {len(train_test_leak)}")
    logger.info(f"Val/Test intersection: {len(val_test_leak)}")
    
    if len(train_val_leak) > 0 or len(train_test_leak) > 0 or len(val_test_leak) > 0:
        logger.warning("WARNING: Possible data leakage detected between splits based on filename!")
    else:
        logger.info("No filename leakage detected between splits.")
        
    return True, audit_results

def get_weighted_sampler(dataset):
    class_counts = Counter([label for _, label in dataset.samples])
    total = len(dataset)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def calculate_macro_f1(all_preds, all_labels, num_classes):
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, y_hat in zip(all_labels, all_preds):
        cm[y][y_hat] += 1
        
    f1_per_class = []
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(num_classes) if j != i)
        fn = sum(cm[i][j] for j in range(num_classes) if j != i)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_per_class.append(f1)
        
    return sum(f1_per_class) / num_classes

def train_model(data_dir, base_model_out_dir, num_epochs=50, patience=8):
    logger.info("Setting up PRODUCTION training...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out_dir = os.path.join(base_model_out_dir, f"v_FULL_{timestamp}")
    os.makedirs(model_out_dir, exist_ok=True)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}
    
    # Use WeightedRandomSampler for train to combat class imbalance
    train_sampler = get_weighted_sampler(image_datasets['train'])
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, sampler=train_sampler, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=64, shuffle=False, num_workers=4)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    class_mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(os.path.join(model_out_dir, 'classes.json'), 'w') as f:
        json.dump(class_mapping, f, indent=4)
        
    config = {
        "model": "MobileNetV2",
        "pretrained": True,
        "input_size": 224,
        "batch_size": 64,
        "optimizer": "AdamW",
        "epochs": num_epochs,
        "patience": patience,
        "device": str(device)
    }
    with open(os.path.join(model_out_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Stage 1: Freeze backbone, train head
    logger.info("--- Stage 1: Training Classification Head ---")
    for param in model.features.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    metrics_history = []
    
    def train_epochs(start_epoch, end_epoch, opt, sched):
        nonlocal best_val_f1, best_val_acc, best_model_wts, best_epoch, epochs_no_improve
        for epoch in range(start_epoch, end_epoch):
            logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            logger.info('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                all_preds = []
                all_labels = []

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    opt.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            opt.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                if phase == 'train' and sched is not None:
                    sched.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_f1 = calculate_macro_f1(all_preds, all_labels, num_classes)

                logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Macro F1: {epoch_f1:.4f}')

                if phase == 'validation':
                    metrics_history.append({
                        "epoch": epoch,
                        "val_loss": epoch_loss,
                        "val_acc": epoch_acc.item(),
                        "val_macro_f1": epoch_f1
                    })
                    
                    if epoch_f1 > best_val_f1:
                        best_val_f1 = epoch_f1
                        best_val_acc = epoch_acc.item()
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), os.path.join(model_out_dir, 'best.pt'))
                    else:
                        epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                return True
                
        return False
        
    # Run Stage 1 for 10 epochs
    early_stopped = train_epochs(0, 10, optimizer, scheduler)
    
    if not early_stopped:
        # Stage 2: Unfreeze backbone, fine-tune
        logger.info("--- Stage 2: Fine-Tuning Entire Model ---")
        for param in model.features.parameters():
            param.requires_grad = True
            
        optimizer_ft = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler_ft = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs - 10)
        
        train_epochs(10, num_epochs, optimizer_ft, scheduler_ft)

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best Epoch: {best_epoch}')
    logger.info(f'Best Val Acc: {best_val_acc:.4f}')
    logger.info(f'Best Val Macro F1: {best_val_f1:.4f}')
    
    with open(os.path.join(model_out_dir, 'metrics.json'), 'w') as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_f1,
            "training_time_seconds": time_elapsed,
            "history": metrics_history
        }, f, indent=4)
        
    logger.info(f"Model and artifacts saved to {model_out_dir}")
    
    # Save a pointer to the newest model in the base dir so evaluate script can find it easily
    with open(os.path.join(base_model_out_dir, 'latest_model.txt'), 'w') as f:
        f.write(model_out_dir)

if __name__ == '__main__':
    data_dir = r'd:\VLM\gadiya\Military'
    base_model_out_dir = r'd:\VLM\models\military_subtype'
    
    success, _ = audit_military_dataset(data_dir)
    if success:
        train_model(data_dir, base_model_out_dir, num_epochs=50, patience=8)
    else:
        logger.error("Audit failed. Aborting training.")
