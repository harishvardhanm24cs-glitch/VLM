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

def resume_training(data_dir, base_model_out_dir, start_epoch=31, num_epochs=50, patience=8):
    logger.info(f"Resuming training from epoch {start_epoch}...")
    
    # Find the latest model dir
    latest_file = os.path.join(base_model_out_dir, 'latest_model.txt')
    if os.path.exists(latest_file):
        with open(latest_file, 'r') as f:
            model_out_dir = f.read().strip()
    else:
        model_out_dir = max([os.path.join(base_model_out_dir, d) for d in os.listdir(base_model_out_dir) if os.path.isdir(os.path.join(base_model_out_dir, d))], key=os.path.getmtime)
        
    logger.info(f"Resuming from model directory: {model_out_dir}")
    model_path = os.path.join(model_out_dir, 'best.pt')
    
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
    
    # Use WeightedRandomSampler for train
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
        
    # Recreate model and load weights
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    logger.info(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # We are in Stage 2 (Fine-tuning entire model)
    logger.info("--- Stage 2: Resuming Fine-Tuning Entire Model ---")
    for param in model.features.parameters():
        param.requires_grad = True
        
    optimizer_ft = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_ft = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=(num_epochs - start_epoch))

    since = time.time()
    
    # Load previous best metrics if possible
    metrics_path = os.path.join(model_out_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            prev_metrics = json.load(f)
            best_val_f1 = prev_metrics.get('best_val_macro_f1', 0.9845)
            best_val_acc = prev_metrics.get('best_val_acc', 0.9850)
            best_epoch = prev_metrics.get('best_epoch', 29)
            metrics_history = prev_metrics.get('history', [])
    else:
        best_val_f1 = 0.9845
        best_val_acc = 0.9850
        best_epoch = 29
        metrics_history = []
        
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
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
                        logger.info(f"New best model saved at epoch {epoch}")
                    else:
                        epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                return True
                
        return False
        
    train_epochs(start_epoch, num_epochs, optimizer_ft, scheduler_ft)

    time_elapsed = time.time() - since
    logger.info(f'Resume Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best Epoch: {best_epoch}')
    logger.info(f'Best Val Acc: {best_val_acc:.4f}')
    logger.info(f'Best Val Macro F1: {best_val_f1:.4f}')
    
    with open(os.path.join(model_out_dir, 'metrics.json'), 'w') as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_f1,
            "history": metrics_history
        }, f, indent=4)

if __name__ == '__main__':
    data_dir = r'd:\VLM\gadiya\Military'
    base_model_out_dir = r'd:\VLM\models\military_subtype'
    
    resume_training(data_dir, base_model_out_dir, start_epoch=31, num_epochs=50, patience=8)
