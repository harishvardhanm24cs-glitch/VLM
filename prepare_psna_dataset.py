import os
import cv2
import hashlib
import csv
import shutil
import random
from pathlib import Path

source_psna = r"D:\VLM\PSNA_Buses_JPG"
negatives_gadiya = r"D:\VLM\gadiya"
dataset_root = r"D:\VLM\datasets\psna_bus_classifier"
metadata_dir = os.path.join(dataset_root, "metadata")
manifest_path = os.path.join(metadata_dir, "dataset_manifest.csv")
docs_dir = r"D:\VLM\docs"

os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dataset_root, split, "PSNA_BUS"), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, split, "OTHER_VEHICLE"), exist_ok=True)


def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def analyze_and_split():
    manifest_rows = []

    psna_files = list(Path(source_psna).glob("*.jpg"))
    print(f"Found {len(psna_files)} PSNA_BUS images.")

    psna_records = []
    seen_hashes = set()
    valid_count = 0
    duplicate_count = 0
    invalid_count = 0

    widths, heights = [], []

    for f in psna_files:
        img = cv2.imread(str(f))
        if img is None:
            invalid_count += 1
            continue

        file_hash = hash_file(f)
        if file_hash in seen_hashes:
            duplicate_count += 1
            continue

        seen_hashes.add(file_hash)
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        valid_count += 1

        base_num = f.stem.replace("IMG_E", "").replace("IMG_", "")
        psna_records.append(
            {"path": f, "hash": file_hash, "w": w, "h": h, "group": base_num}
        )

    groups = {}
    for r in psna_records:
        groups.setdefault(r["group"], []).append(r)

    group_keys = list(groups.keys())
    random.seed(42)
    random.shuffle(group_keys)

    train_psna, val_psna, test_psna = [], [], []
    for g in group_keys:
        recs = groups[g]
        if len(train_psna) < 14:
            train_psna.extend(recs)
        elif len(val_psna) < 3:
            val_psna.extend(recs)
        else:
            test_psna.extend(recs)

    def copy_records(records, split, class_name, source_name):
        for r in records:
            dest = os.path.join(dataset_root, split, class_name, r["path"].name)
            shutil.copy(r["path"], dest)
            manifest_rows.append(
                {
                    "image_path": dest,
                    "class": class_name,
                    "split": split,
                    "sha256": r["hash"],
                    "width": r["w"],
                    "height": r["h"],
                    "source": source_name,
                    "notes": "Original grouped",
                }
            )

    copy_records(train_psna, "train", "PSNA_BUS", "PSNA_Buses_JPG")
    copy_records(val_psna, "val", "PSNA_BUS", "PSNA_Buses_JPG")
    copy_records(test_psna, "test", "PSNA_BUS", "PSNA_Buses_JPG")

    negative_files = list(Path(negatives_gadiya).rglob("*.jpg"))
    random.shuffle(negative_files)

    target_negatives = 30
    selected_negatives = []

    for f in negative_files:
        if len(selected_negatives) >= target_negatives:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        file_hash = hash_file(f)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)

        selected_negatives.append({"path": f, "hash": file_hash, "w": w, "h": h})

    train_neg = selected_negatives[:20]
    val_neg = selected_negatives[20:25]
    test_neg = selected_negatives[25:]

    copy_records(train_neg, "train", "OTHER_VEHICLE", "gadiya")
    copy_records(val_neg, "val", "OTHER_VEHICLE", "gadiya")
    copy_records(test_neg, "test", "OTHER_VEHICLE", "gadiya")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "class",
                "split",
                "sha256",
                "width",
                "height",
                "source",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    report_path = os.path.join(docs_dir, "psna_bus_dataset_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PSNA Bus Dataset Analysis\n\n")
        f.write(f"- Total original images: {len(psna_files)}\n")
        f.write(f"- Valid images: {valid_count}\n")
        f.write(f"- Invalid/Corrupted images: {invalid_count}\n")
        f.write(f"- Exact duplicates: {duplicate_count}\n")
        f.write(
            f"- Image dimensions range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}\n\n"
        )
        f.write("## Data Leakage Prevention\n")
        f.write(
            "HDR/Edited versions of the same physical capture (e.g. IMG_2963 and IMG_E2963) were identified by filename base and strictly grouped into the same split to prevent validation/test leakage.\n\n"
        )
        f.write("## Splits (PSNA_BUS)\n")
        f.write(f"- TRAIN: {len(train_psna)}\n")
        f.write(f"- VAL: {len(val_psna)}\n")
        f.write(f"- TEST: {len(test_psna)}\n")
        f.write("\n## Splits (OTHER_VEHICLE)\n")
        f.write(f"- TRAIN: {len(train_neg)}\n")
        f.write(f"- VAL: {len(val_neg)}\n")
        f.write(f"- TEST: {len(test_neg)}\n")


if __name__ == "__main__":
    analyze_and_split()
