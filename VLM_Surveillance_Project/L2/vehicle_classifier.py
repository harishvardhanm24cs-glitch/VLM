import json
import logging
import os
import sys
from pathlib import Path

vlm_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if vlm_root not in sys.path:
    sys.path.append(vlm_root)

try:
    from L_PSNABus.psna_bus_classifier import PSNABusClassifier
except ImportError as e:
    PSNABusClassifier = None

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


def letterbox_image(img_rgb, desired_size=(224, 224)):
    h, w = img_rgb.shape[:2]
    dw, dh = desired_size
    scale = min(dw / w, dh / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (dh - new_h) // 2
    bottom = dh - new_h - top
    left = (dw - new_w) // 2
    right = dw - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded


class HierarchicalVehicleClassifier:
    def __init__(
        self,
        old_top_dir="d:/VLM/models/vehicle_classifier/v_20260916_235302",
        new_top_dir="d:/VLM/models/vehicle_classifier",
        subtype_dir=None,
        interval_frames=5,
    ):

        self.old_top_dir = Path(old_top_dir) if old_top_dir else None
        self.new_top_dir = Path(new_top_dir) if new_top_dir else None
        self.subtype_dir = Path(subtype_dir) if subtype_dir else None
        self.interval_frames = interval_frames

        self.MIN_MILITARY_PROB = 0.90
        self.MIN_MILITARY_MARGIN = 0.25
        self.MIN_NORMAL_PROB = 0.80

        self.MIN_OBSERVATIONS = 4
        self.MILITARY_STABILITY_RATIO = 0.75
        self.NORMAL_STABILITY_RATIO = 0.70

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.old_top_model = None
        self.new_top_model = None
        self.subtype_model = None

        self.old_mil_idx = -1
        self.old_nor_idx = -1
        self.new_mil_idx = -1
        self.new_nor_idx = -1

        self.sub_mapping = {}

        try:
            if PSNABusClassifier:
                self.psna_classifier = PSNABusClassifier()
            else:
                self.psna_classifier = None
        except Exception as e:
            self.psna_classifier = None

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.track_history = {}

        self.load_models()

    def _load_single_model(self, model_dir):
        mapping = {}
        mil_idx, nor_idx = -1, -1

        with open(model_dir / "classes.json", "r") as f:
            content = f.read()
            mapping = json.loads(content)

        mil_aliases = ["army_vehicle", "military", "Military", "army"]
        nor_aliases = [
            "normal_vehicle",
            "normal",
            "Normal",
            "civilian",
            "civilian_vehicle",
        ]

        for k, v in mapping.items():
            if v in mil_aliases:
                mil_idx = int(k)
            elif v in nor_aliases:
                nor_idx = int(k)

        if mil_idx == -1 or nor_idx == -1:
            err_msg = f"Failed to resolve class indexes. Military: {mil_idx}, Normal: {nor_idx}. Mapping: {mapping}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        num_classes = len(mapping)
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(
            torch.load(model_dir / "best.pt", map_location=self.device)
        )
        model.to(self.device)
        model.eval()

        return model, mil_idx, nor_idx

    def load_models(self):

        if self.old_top_dir and (self.old_top_dir / "best.pt").exists():
            try:
                self.old_top_model, self.old_mil_idx, self.old_nor_idx = (
                    self._load_single_model(self.old_top_dir)
                )
            except Exception as e:
                logger.error(f"Failed to load old top-level classifier: {e}")
                raise e
        else:
            raise FileNotFoundError(f"Old model not found in {self.old_top_dir}")

        if self.new_top_dir and (self.new_top_dir / "best.pt").exists():
            try:
                self.new_top_model, self.new_mil_idx, self.new_nor_idx = (
                    self._load_single_model(self.new_top_dir)
                )
            except Exception as e:
                logger.error(f"Failed to load new top-level classifier: {e}")
                raise e
        else:
            raise FileNotFoundError(f"New model not found in {self.new_top_dir}")

        if (
            self.subtype_dir
            and (self.subtype_dir / "best.pt").exists()
            and (self.subtype_dir / "classes.json").exists()
        ):
            try:
                with open(self.subtype_dir / "classes.json", "r") as f:
                    content = f.read()
                    self.sub_mapping = json.loads(content)

                num_classes = len(self.sub_mapping)
                self.subtype_model = models.mobilenet_v2(pretrained=False)
                self.subtype_model.classifier[1] = nn.Linear(
                    self.subtype_model.classifier[1].in_features, num_classes
                )
                self.subtype_model.load_state_dict(
                    torch.load(self.subtype_dir / "best.pt", map_location=self.device)
                )
                self.subtype_model.to(self.device)
                self.subtype_model.eval()
            except Exception as e:
                logger.error(f"Failed to load subtype classifier: {e}")
                raise e

    def classify_crop(self, cv2_image):
        default_res = {
            "valid": False,
            "reason": "invalid_image",
            "old_mil_prob": 0.0,
            "new_mil_prob": 0.0,
            "old_nor_prob": 0.0,
            "new_nor_prob": 0.0,
            "old_margin": 0.0,
            "new_margin": 0.0,
            "predicted_class": "UNCERTAIN",
        }

        if (
            self.old_top_model is None
            or self.new_top_model is None
            or cv2_image is None
            or cv2_image.size == 0
        ):
            return default_res

        h, w = cv2_image.shape[:2]
        area = w * h
        if w < 20 or h < 20 or area < 400:
            default_res["reason"] = f"crop_too_small_{w}x{h}"
            return default_res

        try:
            img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            padded = letterbox_image(img_rgb)
            img_pil = Image.fromarray(padded)
            img_t = self.transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                old_outputs = self.old_top_model(img_t)
                old_probs = F.softmax(old_outputs, dim=1)[0]

                new_outputs = self.new_top_model(img_t)
                new_probs = F.softmax(new_outputs, dim=1)[0]

            old_mil_prob = old_probs[self.old_mil_idx].item()
            old_nor_prob = old_probs[self.old_nor_idx].item()
            old_margin = old_mil_prob - old_nor_prob

            new_mil_prob = new_probs[self.new_mil_idx].item()
            new_nor_prob = new_probs[self.new_nor_idx].item()
            new_margin = new_mil_prob - new_nor_prob

            predicted_class = "UNCERTAIN"

            if (
                old_mil_prob >= self.MIN_MILITARY_PROB
                and old_margin >= self.MIN_MILITARY_MARGIN
            ) and (
                new_mil_prob >= self.MIN_MILITARY_PROB
                and new_margin >= self.MIN_MILITARY_MARGIN
            ):
                predicted_class = "Military"

            elif (old_nor_prob >= self.MIN_NORMAL_PROB) or (
                new_nor_prob >= self.MIN_NORMAL_PROB
            ):
                predicted_class = "Normal"

            res = {
                "valid": True,
                "reason": "ok",
                "old_mil_prob": old_mil_prob,
                "old_nor_prob": old_nor_prob,
                "old_margin": old_margin,
                "new_mil_prob": new_mil_prob,
                "new_nor_prob": new_nor_prob,
                "new_margin": new_margin,
                "predicted_class": predicted_class,
                "crop_width": w,
                "crop_height": h,
                "crop_area": area,
            }
            return res

        except Exception as e:
            logger.error(f"Classification error: {e}")
            default_res["reason"] = f"exception_{e}"
            return default_res

    def remove_track(self, track_id):
        if track_id in self.track_history:
            del self.track_history[track_id]

    def process_track(self, track_id, cv2_image, yolo_class="unknown", yolo_conf=0.0):
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                "predictions": [],
                "stable_top": None,
                "stable_top_conf": 0.0,
                "stable_sub": None,
                "stable_sub_conf": 0.0,
                "frame_counter": 0,
            }

        history = self.track_history[track_id]

        if history["stable_top"] is not None:
            return {
                "class": history["stable_top"],
                "confidence": history["stable_top_conf"],
                "subtype": history["stable_sub"],
                "subtype_confidence": history["stable_sub_conf"],
            }

        history["frame_counter"] += 1

        if history["frame_counter"] % self.interval_frames != 0:
            return {
                "class": "uncertain",
                "confidence": 0.0,
                "subtype": None,
                "subtype_confidence": 0.0,
            }

        res = self.classify_crop(cv2_image)
        if not res["valid"]:
            return {
                "class": "uncertain",
                "confidence": 0.0,
                "subtype": None,
                "subtype_confidence": 0.0,
            }

        history["predictions"].append(
            {
                "old_mil_prob": res["old_mil_prob"],
                "old_nor_prob": res["old_nor_prob"],
                "old_margin": res["old_margin"],
                "new_mil_prob": res["new_mil_prob"],
                "new_nor_prob": res["new_nor_prob"],
                "new_margin": res["new_margin"],
                "predicted_class": res["predicted_class"],
                "crop_width": res["crop_width"],
                "crop_height": res["crop_height"],
                "crop_area": res["crop_area"],
                "frame_number": history["frame_counter"],
            }
        )

        preds = history["predictions"]

        if len(preds) >= self.MIN_OBSERVATIONS:
            total_valid = len(preds)
            military_count = sum(1 for p in preds if p["predicted_class"] == "Military")
            normal_count = sum(1 for p in preds if p["predicted_class"] == "Normal")

            military_ratio = military_count / total_valid
            normal_ratio = normal_count / total_valid

            avg_old_mil = sum(p["old_mil_prob"] for p in preds) / total_valid
            avg_new_mil = sum(p["new_mil_prob"] for p in preds) / total_valid

            avg_old_nor = sum(p["old_nor_prob"] for p in preds) / total_valid
            avg_new_nor = sum(p["new_nor_prob"] for p in preds) / total_valid

            avg_old_margin = sum(p["old_margin"] for p in preds) / total_valid
            avg_new_margin = sum(p["new_margin"] for p in preds) / total_valid

            if (
                military_ratio >= self.MILITARY_STABILITY_RATIO
                and avg_old_mil >= self.MIN_MILITARY_PROB
                and avg_old_margin >= self.MIN_MILITARY_MARGIN
                and avg_new_mil >= self.MIN_MILITARY_PROB
                and avg_new_margin >= self.MIN_MILITARY_MARGIN
            ):

                history["stable_top"] = "Military"
                history["stable_top_conf"] = (avg_old_mil + avg_new_mil) / 2.0

                if self.subtype_model is not None:
                    try:
                        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                        padded = letterbox_image(img_rgb)
                        img_pil = Image.fromarray(padded)
                        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            sub_outputs = self.subtype_model(img_t)
                            sub_probs = F.softmax(sub_outputs, dim=1)
                            sub_conf, sub_preds = torch.max(sub_probs, 1)

                        sub_conf_val = sub_conf.item()
                        sub_pred_idx = str(sub_preds.item())
                        history["stable_sub"] = self.sub_mapping.get(
                            sub_pred_idx, "Unknown"
                        )
                        history["stable_sub_conf"] = sub_conf_val
                    except Exception as e:
                        logger.error(f"Subtype error: {e}")

            elif normal_ratio >= self.NORMAL_STABILITY_RATIO and (
                avg_old_nor >= self.MIN_NORMAL_PROB
                or avg_new_nor >= self.MIN_NORMAL_PROB
            ):
                history["stable_top"] = "Normal"
                history["stable_top_conf"] = (avg_old_nor + avg_new_nor) / 2.0

                if self.psna_classifier is not None and yolo_class in [
                    "bus",
                    "truck",
                    "unknown",
                ]:
                    try:
                        psna_res = self.psna_classifier.classify(cv2_image)
                        if psna_res["class"] == "PSNA_BUS":
                            history["stable_sub"] = "PSNA_BUS"
                            history["stable_sub_conf"] = psna_res["confidence"]
                    except Exception as e:
                        logger.error(f"PSNA Bus error: {e}")

            if history["stable_top"] is not None:
                return {
                    "class": history["stable_top"],
                    "confidence": history["stable_top_conf"],
                    "subtype": history["stable_sub"],
                    "subtype_confidence": history["stable_sub_conf"],
                }

        return {
            "class": "uncertain",
            "confidence": 0.0,
            "subtype": None,
            "subtype_confidence": 0.0,
        }


classifier = None


def init_classifier(
    old_model_dir="d:/VLM/models/vehicle_classifier/v_20260916_235302",
    new_model_dir="d:/VLM/models/vehicle_classifier",
    subtype_dir=None,
    interval_frames=5,
):
    global classifier
    classifier = HierarchicalVehicleClassifier(
        old_top_dir=old_model_dir,
        new_top_dir=new_model_dir,
        subtype_dir=subtype_dir,
        interval_frames=interval_frames,
    )


def get_classifier():
    return classifier
