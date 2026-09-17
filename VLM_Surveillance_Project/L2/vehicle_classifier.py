import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from collections import Counter

logger = logging.getLogger(__name__)

class HierarchicalVehicleClassifier:
    def __init__(self, 
                 top_level_dir="d:/VLM/models/vehicle_classifier", 
                 subtype_dir=None,
                 min_confidence=0.80, 
                 interval_frames=5, 
                 min_obs=3, 
                 stability_ratio=0.67):
                 
        self.top_level_dir = Path(top_level_dir) if top_level_dir else None
        self.subtype_dir = Path(subtype_dir) if subtype_dir else None
        self.min_confidence = min_confidence
        self.interval_frames = interval_frames
        self.min_obs = min_obs
        self.stability_ratio = stability_ratio
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.top_model = None
        self.subtype_model = None
        
        self.top_mapping = {}
        self.sub_mapping = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # {track_id: {"predictions": [], "stable_top": None, "stable_top_conf": None, "stable_sub": None, "stable_sub_conf": None, "frame_counter": 0}}
        self.track_history = {}
        
        self.load_models()

    def load_models(self):
        # Load Top-level (Binary) Model
        if (self.top_level_dir / "best.pt").exists() and (self.top_level_dir / "classes.json").exists():
            try:
                with open(self.top_level_dir / "classes.json", "r") as f:
                    self.top_mapping = json.load(f)
                    
                num_classes = len(self.top_mapping)
                self.top_model = models.mobilenet_v2(pretrained=False)
                self.top_model.classifier[1] = nn.Linear(self.top_model.classifier[1].in_features, num_classes)
                self.top_model.load_state_dict(torch.load(self.top_level_dir / "best.pt", map_location=self.device))
                self.top_model.to(self.device)
                self.top_model.eval()
                logger.info(f"Loaded top-level classifier: {self.top_mapping}")
            except Exception as e:
                logger.error(f"Failed to load top-level classifier: {e}")
                
        # Load Subtype Model
        if self.subtype_dir and (self.subtype_dir / "best.pt").exists() and (self.subtype_dir / "classes.json").exists():
            try:
                with open(self.subtype_dir / "classes.json", "r") as f:
                    self.sub_mapping = json.load(f)
                    
                num_classes = len(self.sub_mapping)
                self.subtype_model = models.mobilenet_v2(pretrained=False)
                self.subtype_model.classifier[1] = nn.Linear(self.subtype_model.classifier[1].in_features, num_classes)
                self.subtype_model.load_state_dict(torch.load(self.subtype_dir / "best.pt", map_location=self.device))
                self.subtype_model.to(self.device)
                self.subtype_model.eval()
                logger.info(f"Loaded subtype classifier: {self.sub_mapping}")
            except Exception as e:
                logger.error(f"Failed to load subtype classifier: {e}")

    def classify_crop(self, cv2_image):
        default_res = {"top_class": "uncertain", "top_conf": 0.0, "sub_class": None, "sub_conf": 0.0}
        
        if self.top_model is None or cv2_image is None or cv2_image.size == 0:
            return default_res

        try:
            import cv2
            img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.top_model(img_t)
                probs = F.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, 1)
                
            conf = conf.item()
            if conf < self.min_confidence:
                return default_res
                
            pred_idx = str(preds.item())
            top_class = self.top_mapping.get(pred_idx, "uncertain")
            
            # Get probabilities by mapping 
            # Assuming '0' is Military and '1' is Normal according to self.top_mapping
            mil_idx = next((int(k) for k, v in self.top_mapping.items() if v == "army_vehicle"), 0)
            nor_idx = next((int(k) for k, v in self.top_mapping.items() if v == "normal_vehicle"), 1)
            
            res = {
                "top_class": top_class, 
                "top_conf": conf, 
                "sub_class": None, 
                "sub_conf": 0.0,
                "mil_prob": probs[0][mil_idx].item(),
                "nor_prob": probs[0][nor_idx].item()
            }
            
            # Map legacy 'army_vehicle' to 'Military' and 'normal_vehicle' to 'Normal'
            clean_top_class = "Military" if top_class == "army_vehicle" else "Normal" if top_class == "normal_vehicle" else "uncertain"
            res["top_class"] = clean_top_class
            
            if clean_top_class == "Military" and self.subtype_model is not None:
                with torch.no_grad():
                    sub_outputs = self.subtype_model(img_t)
                    sub_probs = F.softmax(sub_outputs, dim=1)
                    sub_conf, sub_preds = torch.max(sub_probs, 1)
                
                sub_conf_val = sub_conf.item()
                if sub_conf_val >= self.min_confidence:
                    sub_pred_idx = str(sub_preds.item())
                    res["sub_class"] = self.sub_mapping.get(sub_pred_idx, "Unknown")
                    res["sub_conf"] = sub_conf_val
                    
            return res
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return default_res

    def process_track(self, track_id, cv2_image):
        """
        Processes a tracked vehicle. Aggregates predictions across frames.
        Returns stable top-level and subtype classes.
        """
        if len(self.track_history) > 1000:
            self.track_history.clear()
            
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                "predictions": [], 
                "stable_top": None, 
                "stable_top_conf": 0.0, 
                "stable_sub": None, 
                "stable_sub_conf": 0.0, 
                "frame_counter": 0
            }
            
        history = self.track_history[track_id]
        
        # If stable, return cached result
        if history["stable_top"] is not None:
            return {
                "class": history["stable_top"], 
                "confidence": history["stable_top_conf"],
                "subtype": history["stable_sub"],
                "subtype_confidence": history["stable_sub_conf"]
            }
            
        history["frame_counter"] += 1
        
        reason = None
        
        if history["frame_counter"] % self.interval_frames != 0:
            reason = f"Frame counter {history['frame_counter']} not divisible by {self.interval_frames}"
            return {"class": "uncertain", "confidence": 0.0, "subtype": None, "subtype_confidence": 0.0}
            
        if len(history["predictions"]) >= self.min_obs * 5:
            reason = f"Too many predictions ({len(history['predictions'])})"
            return {"class": "uncertain", "confidence": 0.0, "subtype": None, "subtype_confidence": 0.0}
            
        res = self.classify_crop(cv2_image)
        if res["top_class"] != "uncertain":
            history["predictions"].append(res)
        
        preds = history["predictions"]
        
        if len(preds) < self.min_obs:
            reason = f"Observations {len(preds)}/{self.min_obs}"
        
        if len(preds) >= self.min_obs:
            total_valid = len(preds)
            military_count = sum(1 for p in preds if p["top_class"] == "Military")
            normal_count = sum(1 for p in preds if p["top_class"] == "Normal")
            
            military_ratio = military_count / total_valid
            normal_ratio = normal_count / total_valid
            
            # --- DEBUG LOGGING ---
            last_mil_prob = res.get("mil_prob", 0.0)
            last_nor_prob = res.get("nor_prob", 0.0)
            print(f"\n[VEHICLE CLASSIFIER]")
            print(f"Track ID: {track_id}")
            print(f"Crop dimensions: {cv2_image.shape[1]}x{cv2_image.shape[0]}")
            print(f"Binary predicted class: {res['top_class']}")
            print(f"Binary confidence: {res['top_conf']:.4f}")
            print(f"Military probability: {last_mil_prob:.4f}")
            print(f"Normal probability: {last_nor_prob:.4f}")
            print(f"Uncertain threshold: {self.min_confidence}")
            print(f"Temporal observation count: {len(preds)}")
            print(f"Military ratio: {military_ratio:.2f}")
            print(f"Stability ratio: {self.stability_ratio}")
            
            if military_ratio >= self.stability_ratio:
                history["stable_top"] = "Military"
                history["stable_top_conf"] = sum(p["top_conf"] for p in preds if p["top_class"] == "Military") / military_count
                
                print("Final category: Military")
                print("Subtype classifier: CALLED")
                # Subtype voting
                subtypes = [p["sub_class"] for p in preds if p["top_class"] == "Military" and p["sub_class"] is not None]
                if subtypes:
                    from collections import Counter
                    counter = Counter(subtypes)
                    most_common, count = counter.most_common(1)[0]
                    if count / len(subtypes) >= 0.5: # Simple majority for subtype
                        history["stable_sub"] = most_common
                        history["stable_sub_conf"] = sum(p["sub_conf"] for p in preds if p["sub_class"] == most_common) / count
                        print(f"Subtype: {most_common} = {history['stable_sub_conf']:.4f}")
                        
            elif normal_ratio >= self.stability_ratio:
                history["stable_top"] = "Normal"
                history["stable_top_conf"] = sum(p["top_conf"] for p in preds if p["top_class"] == "Normal") / normal_count
                history["stable_sub"] = "Unknown Civilian" if False else None # Optional generic label or None
                print("Final category: Normal")
                print("Subtype classifier: NOT RUN")
                
            else:
                reason = f"Military ratio {military_ratio:.2f} and Normal ratio {normal_ratio:.2f} < threshold {self.stability_ratio}"
                print("Final category: UNCERTAIN")
                print(f"Reason: {reason}")
                print("Subtype classifier: NOT RUN")
                
            if history["stable_top"] is not None:
                return {
                    "class": history["stable_top"], 
                    "confidence": history["stable_top_conf"],
                    "subtype": history["stable_sub"],
                    "subtype_confidence": history["stable_sub_conf"]
                }
                
        # If we reach here, it's uncertain.
        # Check if we should print debug even if observations are too few
        if reason and len(preds) < self.min_obs:
            last_mil_prob = res.get("mil_prob", 0.0)
            last_nor_prob = res.get("nor_prob", 0.0)
            print(f"\n[VEHICLE CLASSIFIER]")
            print(f"Track ID: {track_id}")
            print(f"Crop dimensions: {cv2_image.shape[1]}x{cv2_image.shape[0]}")
            print(f"Binary predicted class: {res['top_class']}")
            print(f"Binary confidence: {res['top_conf']:.4f}")
            print(f"Military probability: {last_mil_prob:.4f}")
            print(f"Normal probability: {last_nor_prob:.4f}")
            print(f"Uncertain threshold: {self.min_confidence}")
            print(f"Temporal observation count: {len(preds)}/{self.min_obs}")
            print(f"Final category: UNCERTAIN")
            print(f"Reason: {reason}")
            print("Subtype classifier: NOT RUN")
            
        return {"class": "uncertain", "confidence": 0.0, "subtype": None, "subtype_confidence": 0.0}

classifier = None

def init_classifier(model_dir="d:/VLM/models/vehicle_classifier", subtype_dir=None, min_confidence=0.80, interval_frames=5, min_obs=3, stability_ratio=0.67):
    global classifier
    classifier = HierarchicalVehicleClassifier(
        top_level_dir=model_dir,
        subtype_dir=subtype_dir,
        min_confidence=min_confidence, 
        interval_frames=interval_frames, 
        min_obs=min_obs, 
        stability_ratio=stability_ratio
    )
    
def get_classifier():
    return classifier
