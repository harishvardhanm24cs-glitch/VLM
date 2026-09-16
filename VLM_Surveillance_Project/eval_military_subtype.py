import os
import sys
import json
import logging
import math
import random
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(data_dir, base_model_dir):
    # Find the newest model using the pointer
    latest_file = os.path.join(base_model_dir, 'latest_model.txt')
    if os.path.exists(latest_file):
        with open(latest_file, 'r') as f:
            model_dir = f.read().strip()
    else:
        # Fallback if no pointer
        model_dir = max([os.path.join(base_model_dir, d) for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))], key=os.path.getmtime)
        
    logger.info(f"Evaluating model at: {model_dir}")
    test_dir = os.path.join(data_dir, 'test')
    
    with open(os.path.join(model_dir, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)
        
    num_classes = len(class_mapping)
    class_names = [class_mapping[str(i)] for i in range(num_classes)]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(test_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    logger.info(f"Evaluating model on full test dataset ({len(dataset)} images)...")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    correct = sum(1 for y, y_hat in zip(all_labels, all_preds) if y == y_hat)
    acc = correct / len(all_labels) if all_labels else 0
    
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, y_hat in zip(all_labels, all_preds):
        cm[y][y_hat] += 1
        
    prec_per_class = []
    rec_per_class = []
    f1_per_class = []
    support_per_class = []
    
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(num_classes) if j != i)
        fn = sum(cm[i][j] for j in range(num_classes) if j != i)
        support = sum(cm[i])
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        prec_per_class.append(prec)
        rec_per_class.append(rec)
        f1_per_class.append(f1)
        support_per_class.append(support)
        
    macro_prec = sum(prec_per_class) / num_classes if num_classes > 0 else 0
    macro_rec = sum(rec_per_class) / num_classes if num_classes > 0 else 0
    macro_f1 = sum(f1_per_class) / num_classes if num_classes > 0 else 0
    
    total_samples = len(all_labels)
    weighted_f1 = sum(f1_per_class[i] * support_per_class[i] for i in range(num_classes)) / total_samples if total_samples > 0 else 0
    
    # Save Classification Report
    report_path = os.path.join(model_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("--- MILITARY SUBTYPE CLASSIFICATION REPORT ---\n\n")
        f.write(f"Overall Accuracy:  {acc:.4f}\n")
        f.write(f"Macro Precision:   {macro_prec:.4f}\n")
        f.write(f"Macro Recall:      {macro_rec:.4f}\n")
        f.write(f"Macro F1:          {macro_f1:.4f}\n")
        f.write(f"Weighted F1:       {weighted_f1:.4f}\n\n")
        
        f.write(f"{'Class':<35} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'Support'}\n")
        f.write("-" * 75 + "\n")
        for i, name in enumerate(class_names):
            f.write(f"{name:<35} | {prec_per_class[i]:.4f}    | {rec_per_class[i]:.4f} | {f1_per_class[i]:.4f} | {support_per_class[i]}\n")
            
    logger.info(f"Classification report saved to {report_path}")
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Military Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    cm_np = np.array(cm)
    thresh = np.max(cm_np) / 2.
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            plt.text(j, i, format(cm_np[i][j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm_np[i][j] > thresh else "black")
                     
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    cm_path = os.path.join(model_dir, 'military_confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Report confused classes
    logger.info("--- Most Confused Classes ---")
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i][j]))
                
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    for pair in confused_pairs[:5]:
        logger.info(f"True '{pair[0]}' misclassified as '{pair[1]}': {pair[2]} times")
        
    # Real Image Sanity Test
    logger.info("--- REAL IMAGE SANITY TEST ---")
    
    # Group dataset by class
    class_images = {i: [] for i in range(num_classes)}
    for idx, (path, label) in enumerate(dataset.samples):
        class_images[label].append(path)
        
    for i in range(num_classes):
        selected_paths = random.sample(class_images[i], min(3, len(class_images[i])))
        for path in selected_paths:
            img = Image.open(path).convert('RGB')
            input_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_t)
                prob = torch.nn.functional.softmax(out, dim=1)[0]
                conf, pred = torch.max(prob, 0)
                
            logger.info(f"Image: {os.path.basename(path)}")
            logger.info(f"Actual class: {class_names[i]}")
            logger.info(f"Predicted class: {class_names[pred.item()]}")
            logger.info(f"Confidence: {conf.item():.4f}")
            logger.info("-" * 20)
            
    # Model Quality Gate
    logger.info("--- MODEL QUALITY GATE ---")
    if acc >= 0.90 and macro_f1 >= 0.90:
        logger.info("MILITARY SUBTYPE MODEL READY FOR INTEGRATION")
    else:
        logger.warning("TRAINING INSUFFICIENT")
        logger.warning(f"Failed criteria: Accuracy ({acc:.4f}) or Macro F1 ({macro_f1:.4f}) < 0.90")
        for i in range(num_classes):
            if rec_per_class[i] < 0.8:
                logger.warning(f"Class '{class_names[i]}' has poor recall: {rec_per_class[i]:.4f}")

if __name__ == '__main__':
    evaluate_model(r'd:\VLM\gadiya\Military', r'd:\VLM\models\military_subtype')
