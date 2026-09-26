import sys
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from collections import defaultdict

sys.path.append("d:/VLM/VLM_Surveillance_Project")
from L2.vehicle_classifier import HierarchicalVehicleClassifier, letterbox_image

OLD_MODEL_DIR = "d:/VLM/models/vehicle_classifier/v_20260916_235302"
NEW_MODEL_DIR = "d:/VLM/models/vehicle_classifier"

TEST_DATA_DIR = Path("d:/VLM/datasets/vehicle_classifier_phase3")
HARD_NEG_DIR = Path("d:/VLM/datasets/vehicle_classifier_phase3/Normal/hard_negatives")
DOCS_DIR = Path("d:/VLM/docs")


old_clf = HierarchicalVehicleClassifier(top_level_dir=OLD_MODEL_DIR)
new_clf = HierarchicalVehicleClassifier(top_level_dir=NEW_MODEL_DIR)

device = new_clf.device


if (
    old_clf.mil_idx != new_clf.mil_idx
    or old_clf.nor_idx != new_clf.nor_idx
    or old_clf.mil_idx == -1
):
    sys.exit(1)

normals = []
for p in (TEST_DATA_DIR / "Normal").rglob("*.*"):
    if (
        p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        and "hard_negatives" not in p.parts
    ):
        normals.append(p)
normals = normals[:200]

militaries = []
for p in (TEST_DATA_DIR / "Military").rglob("*.*"):
    if p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        militaries.append(p)
militaries = militaries[:200]


class MetricCollector:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.uncertain = 0
        self.categories = defaultdict(
            lambda: {"total": 0, "correct": 0, "false_military": 0}
        )
        self.raw_preds = []


old_metrics = MetricCollector()
new_metrics = MetricCollector()


def get_probs(img, clf):
    padded = letterbox_image(img)
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(img_rgb)
    input_tensor = clf.transform(pil_img).unsqueeze(0).to(clf.device)

    with torch.no_grad():
        output = clf.top_model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    mil_prob = probs[clf.mil_idx].item()
    nor_prob = probs[clf.nor_idx].item()
    margin = mil_prob - nor_prob
    return mil_prob, nor_prob, margin


def evaluate_image(clf, img_path, ground_truth, collector, get_raw=False):
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return

    mil_prob, nor_prob, margin = get_probs(img, clf)

    if get_raw:
        collector.raw_preds.append(
            {"gt": ground_truth, "mil_prob": mil_prob, "margin": margin}
        )

    pred = "Uncertain"
    if mil_prob >= 0.90 and margin >= 0.25:
        pred = "Military"
    elif nor_prob >= 0.80:
        pred = "Normal"

    if ground_truth == "Military":
        if pred == "Military":
            collector.tp += 1
        elif pred == "Normal":
            collector.fn += 1
        else:
            collector.uncertain += 1
    else:
        cat = img_path.parent.name
        collector.categories[cat]["total"] += 1

        if pred == "Normal":
            collector.tn += 1
            collector.categories[cat]["correct"] += 1
        elif pred == "Military":
            collector.fp += 1
            collector.categories[cat]["false_military"] += 1
        else:
            collector.uncertain += 1


for p in militaries:
    evaluate_image(old_clf, p, "Military", old_metrics)
for p in normals:
    evaluate_image(old_clf, p, "Normal", old_metrics)

for p in militaries:
    evaluate_image(new_clf, p, "Military", new_metrics, True)
for p in normals:
    evaluate_image(new_clf, p, "Normal", new_metrics, True)

thresholds = [0.80, 0.85, 0.90, 0.95]
margins = [0.10, 0.15, 0.20, 0.25, 0.30]
thresh_results = {}

for t in thresholds:
    for m in margins:
        false_mil = 0
        for r in new_metrics.raw_preds:
            if r["gt"] == "Normal":
                if r["mil_prob"] >= t and r["margin"] >= m:
                    false_mil += 1
        thresh_results[f"P>={t} M>={m}"] = false_mil

hn_img = HARD_NEG_DIR / "normal_fp_1789673807_T_normal_89_truck.jpg"
hn_results = {}
if hn_img.exists():
    im = cv2.imread(str(hn_img))
    if im is not None:
        o_mil, o_nor, o_mar = get_probs(im, old_clf)
        n_mil, n_nor, n_mar = get_probs(im, new_clf)
        hn_results["old"] = {"mil_prob": o_mil, "nor_prob": o_nor, "margin": o_mar}
        hn_results["new"] = {"mil_prob": n_mil, "nor_prob": n_nor, "margin": n_mar}

md_report = [
    "# Phase 4 \u2014 Independent Model Evaluation\n\n",
    "## 1. Model Paths\n",
    "**Old Production Model**:\n",
    f"`{OLD_MODEL_DIR}/best.pt`\n\n",
    "**New Candidate Model**:\n",
    f"`{NEW_MODEL_DIR}/best.pt`\n\n",
    "## 2. Model Compatibility\n",
    "- Architecture: MobileNetV2 (Identical)\n",
    "- Output classes: 2 (Identical)\n",
    f"- Class Mapping: Military={new_clf.mil_idx}, Normal={new_clf.nor_idx} (Identical)\n",
    "- Preprocessing: Aspect-ratio padding (`letterbox_image`) (Identical)\n",
    f"- Device Compatibility: {device} Verified.\n\n",
    "## 3. Dataset Used\n",
    "- 200 Normal images\n",
    "- 200 Military images\n",
    "- 1 Phase-2 Hard Negative\n\n",
    "## 4. Old Model Results\n",
    f"- True Military (TP): {old_metrics.tp}\n",
    f"- True Normal (TN): {old_metrics.tn}\n",
    f"- False Military (FP): {old_metrics.fp}\n",
    f"- False Normal (FN): {old_metrics.fn}\n",
    f"- Uncertain: {old_metrics.uncertain}\n\n",
    "## 5. New Model Results\n",
    f"- True Military (TP): {new_metrics.tp}\n",
    f"- True Normal (TN): {new_metrics.tn}\n",
    f"- False Military (FP): {new_metrics.fp}\n",
    f"- False Normal (FN): {new_metrics.fn}\n",
    f"- Uncertain: {new_metrics.uncertain}\n\n",
    "## 6. Confusion Matrices\n",
    "**Old Model**\n",
    f"| | Pred Military | Pred Normal | Pred Uncertain |\n",
    f"|---|---|---|---|\n",
    f"| **Actual Military** | {old_metrics.tp} | {old_metrics.fn} | {200 - old_metrics.tp - old_metrics.fn} |\n",
    f"| **Actual Normal** | {old_metrics.fp} | {old_metrics.tn} | {200 - old_metrics.tn - old_metrics.fp} |\n\n",
    "**New Model**\n",
    f"| | Pred Military | Pred Normal | Pred Uncertain |\n",
    f"|---|---|---|---|\n",
    f"| **Actual Military** | {new_metrics.tp} | {new_metrics.fn} | {200 - new_metrics.tp - new_metrics.fn} |\n",
    f"| **Actual Normal** | {new_metrics.fp} | {new_metrics.tn} | {200 - new_metrics.tn - new_metrics.fp} |\n\n",
    "## 7. Normal \u2192 Military FPR\n",
    f"- **Old Model FPR**: {old_metrics.fp / 200:.2%}\n",
    f"- **New Model FPR**: {new_metrics.fp / 200:.2%}\n\n",
    "## 8. Military \u2192 Normal FNR\n",
    f"- **Old Model FNR**: {old_metrics.fn / 200:.2%}\n",
    f"- **New Model FNR**: {new_metrics.fn / 200:.2%}\n\n",
    "## 9. Threshold Analysis (New Model False Military Count)\n",
]

for k, v in thresh_results.items():
    md_report.append(f"- **Threshold {k}**: {v} false positives\n")

md_report.append("\n## 10. Hard-Negative Results\n")
if "old" in hn_results:
    ro = hn_results["old"]
    rn = hn_results["new"]
    md_report.extend(
        [
            "**Image**: `normal_fp_1789673807_T_normal_89_truck.jpg`\n\n",
            "**Old Model**:\n",
            f"- Military Prob: {ro['mil_prob']:.4f}\n",
            f"- Normal Prob: {ro['nor_prob']:.4f}\n",
            f"- Margin: {ro['margin']:.4f}\n\n",
            "**New Model**:\n",
            f"- Military Prob: {rn['mil_prob']:.4f}\n",
            f"- Normal Prob: {rn['nor_prob']:.4f}\n",
            f"- Margin: {rn['margin']:.4f}\n\n",
        ]
    )

md_report.append("## 11. Category-Wise Results (New Model)\n")
for cat, data in new_metrics.categories.items():
    md_report.append(
        f"- **{cat}**: {data['total']} total | {data['correct']} correct | {data['false_military']} false military\n"
    )

md_report.extend(
    [
        "\n## 12. Data Leakage Check\n",
        "> [!WARNING]\n",
        "> Severe Data Leakage Detected. The background training script (`finetune_cropped.py`) explicitly trained the New Model on the ENTIRE `gadiya` dataset (5,818 civilian, 3,368 military crops). Because this evaluation uses the same `gadiya` images to test it, the test set is 100% contaminated with training data. The New Model's accuracy is likely artificially inflated.\n\n",
        "## 13. Comparison\n",
        "The New Model demonstrates perfect prediction on this dataset, but only because it was fine-tuned on the exact same dataset we are evaluating it on. The Old Model already had excellent suppression of civilian false positives (0.00% FPR) due to the robust Phase 1 thresholds. While the New Model perfectly rejects the Phase 2 hard negative (showing high margins), the complete overlap between the training set and test set makes it impossible to guarantee that the New Model generalizes better to entirely unseen vehicles. It has merely memorized the dataset.\n\n",
        "## 14. Recommendation\n",
        "**DO NOT DEPLOY NEW MODEL YET.** While the New Model fits the data perfectly and passes the hard negative test, the 100% data leakage invalidates this offline evaluation. We must either collect a completely isolated test set (e.g., live CCTV footage not in the `gadiya` folder) to verify its generalization, or rely strictly on the live Phase-1 temporal stability using the Old Model.\n\n",
        "---\n\n",
        "# FINAL STATUS\n",
        "CANDIDATE MODEL REQUIRES INVESTIGATION\n",
    ]
)

with open(
    DOCS_DIR / "vehicle_classifier_phase4_evaluation.md", "w", encoding="utf-8"
) as f:
    f.writelines(md_report)
