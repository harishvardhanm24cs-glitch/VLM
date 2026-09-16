import argparse
import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

def load_classifier(model_dir):
    model_path = model_dir / "best.pt"
    classes_path = model_dir / "classes.json"
    
    if not model_path.exists() or not classes_path.exists():
        print(f"Error: Model files not found in {model_dir}")
        sys.exit(1)
        
    with open(classes_path, "r") as f:
        class_mapping = json.load(f)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_mapping)
    
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, class_mapping, device

def test_image(image_path, min_confidence=0.80):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)
        
    model_dir = Path("d:/VLM/models/vehicle_classifier")
    
    # Try to find the version if training_config exists
    config_path = model_dir / "training_config.json"
    version_info = "Unknown"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)
            version_info = cfg.get("timestamp", "Unknown")
            
    print(f"Loading Model Version: {version_info} from {model_dir}")
    
    model, class_mapping, device = load_classifier(model_dir)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(img_path).convert("RGB")
        img_t = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
        conf = conf.item()
        pred_idx = str(preds.item())
        predicted_class = class_mapping.get(pred_idx, "unknown")
        
        print("\n## Vehicle Classification")
        if conf < min_confidence:
            print(f"Class: UNCERTAIN")
            print(f"Confidence: {conf*100:.1f}%")
        elif predicted_class == "army_vehicle":
            print(f"Class: ARMY VEHICLE")
            print(f"Confidence: {conf*100:.1f}%")
        elif predicted_class == "normal_vehicle":
            print(f"Class: NORMAL VEHICLE")
            print(f"Confidence: {conf*100:.1f}%")
        else:
            print(f"Class: {predicted_class.upper()}")
            print(f"Confidence: {conf*100:.1f}%")
            
    except Exception as e:
        print(f"Failed to process image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image to classify")
    args = parser.parse_args()
    
    import os
    from dotenv import load_dotenv
    load_dotenv("d:/VLM/.env")
    min_conf = float(os.getenv("VEHICLE_CLASSIFICATION_MIN_CONFIDENCE", "0.80"))
    
    test_image(args.image, min_confidence=min_conf)
