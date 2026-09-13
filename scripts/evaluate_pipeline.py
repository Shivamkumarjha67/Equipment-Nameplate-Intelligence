import sys
from pathlib import Path

# Add project root to sys.path so src imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
import cv2
from src.ocr.extractor import NameplateExtractor
from src.vision.preprocessor import NameplatePreprocessor

ANNOTATIONS_PATH = Path("data/synthetic/annotations.jsonl")
IMAGE_DIR = Path("data/synthetic/images")

def run_evaluation(num_samples: int = 100):
    preprocessor = NameplatePreprocessor(blur_threshold=100.0)
    extractor = NameplateExtractor(gpu=False)

    field_metrics = {
        "voltage": {"correct": 0, "total": 0},
        "current": {"correct": 0, "total": 0},
        "power": {"correct": 0, "total": 0},
        "frequency": {"correct": 0, "total": 0},
        "manufacturer": {"correct": 0, "total": 0},
        "serial_number": {"correct": 0, "total": 0},
    }

    print(f"Starting evaluation on {num_samples} samples...")
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break

            record = json.loads(line)
            image_path = IMAGE_DIR / record["image_file"]
            gt_specs = record["specs"]

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            # 1. OpenCV Preprocessing
            prep_res = preprocessor.process(image)

            # 2. OCR Extraction
            tokens = extractor.run_ocr(prep_res["enhanced_gray"])
            preds = extractor.parse_fields(tokens)

            # 3. Score against Ground Truth
            for field in field_metrics.keys():
                field_metrics[field]["total"] += 1
                gt_val = gt_specs.get(field)
                pred_val = preds[field]["value"]

                if gt_val is not None and pred_val is not None:
                    if str(gt_val).strip() == str(pred_val).strip():
                        field_metrics[field]["correct"] += 1

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{num_samples} samples...")

    print("\n" + "=" * 45)
    print(f"{'FIELD':<16} | {'ACCURACY':<10} | {'CORRECT / TOTAL'}")
    print("-" * 45)
    for field, data in field_metrics.items():
        acc = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"{field:<16} | {acc:>8.2f}% | {data['correct']}/{data['total']}")
    print("=" * 45)

if __name__ == "__main__":
    run_evaluation(num_samples=50)