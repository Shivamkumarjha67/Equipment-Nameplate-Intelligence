import sys
from pathlib import Path
import json
import time
import io

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import joblib
import numpy as np

from src.config import ANNOTATIONS_PATH, MODELS_DIR, SYNTHETIC_IMAGES
from src.ml.feature_extractor import NameplateFeatureExtractor
from src.ocr.extractor import NameplateExtractor
from src.vision.preprocessor import NameplatePreprocessor


def run_rigorous_evaluation(num_samples: int = 60):
    preprocessor = NameplatePreprocessor(blur_threshold=100.0)
    extractor = NameplateExtractor(gpu=False)
    feat_extractor = NameplateFeatureExtractor()
    rf_model = joblib.load(MODELS_DIR / "review_classifier.joblib")
    iso_model = joblib.load(MODELS_DIR / "isolation_forest.joblib")

    fields = ["manufacturer", "voltage", "current", "power", "frequency", "serial_number"]

    # Metrics containers
    raw_scores = {f: {"correct": 0, "total": 0} for f in fields}
    pipeline_scores = {f: {"correct": 0, "total": 0} for f in fields}

    # Triage metrics
    triage_tp = 0  # True positive: Needed review and flagged as review
    triage_fp = 0  # False positive: Acceptable plate flagged as review
    triage_tn = 0  # True negative: Acceptable plate accepted
    triage_fn = 0  # False negative: Needed review but falsely accepted (critical safety risk!)

    # Latencies in milliseconds
    latencies = {"cv": [], "ocr": [], "ml": []}

    print(f"Executing empirical ablation evaluation on {num_samples} samples...")
    print("Comparing: Baseline (Raw Image -> OCR) vs Pipeline (CV Homography + CLAHE -> OCR -> ML Gate)")

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break

            record = json.loads(line)
            img_path = SYNTHETIC_IMAGES / record["image_file"]
            gt = record["specs"]

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Ground truth determination: does this sample require manual review?
            # A sample needs review if any key field is corrupt or heavily blurred/glared
            is_ground_truth_flawed = record.get("difficulty") == "severe"

            # ---------------------------------------------------------
            # 1. Baseline: Raw Image (No homography, no CLAHE) -> OCR
            # ---------------------------------------------------------
            gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            raw_tokens = extractor.run_ocr(gray_raw)
            raw_preds = extractor.parse_fields(raw_tokens)

            for f_name in fields:
                raw_scores[f_name]["total"] += 1
                gt_val = str(gt.get(f_name, "")).strip()
                pred_val = str(raw_preds[f_name].get("value", "")).strip()
                if gt_val and pred_val and gt_val == pred_val:
                    raw_scores[f_name]["correct"] += 1

            # ---------------------------------------------------------
            # 2. Pipeline: Full CV Homography + Bilateral + CLAHE
            # ---------------------------------------------------------
            t0 = time.perf_counter()
            prep_res = preprocessor.process(image)
            t1 = time.perf_counter()

            pipeline_tokens = extractor.run_ocr(prep_res["enhanced_gray"])
            t2 = time.perf_counter()

            pipeline_preds = extractor.parse_fields(pipeline_tokens)

            for f_name in fields:
                pipeline_scores[f_name]["total"] += 1
                gt_val = str(gt.get(f_name, "")).strip()
                pred_val = str(pipeline_preds[f_name].get("value", "")).strip()
                if gt_val and pred_val and gt_val == pred_val:
                    pipeline_scores[f_name]["correct"] += 1

            # ---------------------------------------------------------
            # 3. ML Quality Gate Triage Evaluation
            # ---------------------------------------------------------
            feats = feat_extractor.extract_features(prep_res, pipeline_tokens, pipeline_preds)
            x_vec = [feat_extractor.to_feature_vector(feats)]
            triage_pred = rf_model.predict(x_vec)[0]  # 0 = Accepted, 1 = Review
            t3 = time.perf_counter()

            latencies["cv"].append((t1 - t0) * 1000)
            latencies["ocr"].append((t2 - t1) * 1000)
            latencies["ml"].append((t3 - t2) * 1000)

            # Confusion matrix for Review detection
            if is_ground_truth_flawed:
                if triage_pred == 1:
                    triage_tp += 1
                else:
                    triage_fn += 1
            else:
                if triage_pred == 0:
                    triage_tn += 1
                else:
                    triage_fp += 1

            if (idx + 1) % 15 == 0:
                print(f"  Processed {idx + 1}/{num_samples} evaluation images...")

    # Print Markdown Benchmark Table
    print("\n" + "=" * 70)
    print("### EMPIRICAL ABLATION RESULTS (Formatted for README.md)")
    print("=" * 70)
    print("| Parameter | Baseline (Raw OCR) | Full Pipeline (CV + CLAHE) | Accuracy Lift |")
    print("| :--- | :---: | :---: | :---: |")
    for f_name in fields:
        raw_acc = (raw_scores[f_name]["correct"] / raw_scores[f_name]["total"]) * 100
        pipe_acc = (pipeline_scores[f_name]["correct"] / pipeline_scores[f_name]["total"]) * 100
        lift = pipe_acc - raw_acc
        sign = "+" if lift >= 0 else ""
        print(f"| **{f_name.replace('_', ' ').title()}** | {raw_acc:.1f}% | **{pipe_acc:.1f}%** | {sign}{lift:.1f}% |")

    # Triage performance
    precision = (triage_tp / (triage_tp + triage_fp)) * 100 if (triage_tp + triage_fp) > 0 else 0
    recall = (triage_tp / (triage_tp + triage_fn)) * 100 if (triage_tp + triage_fn) > 0 else 0
    print("\n### ML QUALITY GATE (TRIAGE) PERFORMANCE")
    print(f"- Safety Recall (Corrupted Plates Flagged): {recall:.1f}%")
    print(f"- Triage Precision: {precision:.1f}%")
    print(f"- Critical False Acceptance Rate (FN): {(triage_fn / max(1, triage_tp + triage_fn)) * 100:.1f}%")

    # Latencies
    print("\n### LATENCY PROFILE (CPU, Single Thread)")
    print(f"- OpenCV Homography & CLAHE: P50 = {np.median(latencies['cv']):.1f} ms | P95 = {np.percentile(latencies['cv'], 95):.1f} ms")
    print(f"- EasyOCR Detection & Recognition: P50 = {np.median(latencies['ocr']):.1f} ms | P95 = {np.percentile(latencies['ocr'], 95):.1f} ms")
    print(f"- ML Quality Gate & Physics Triplet: P50 = {np.median(latencies['ml']):.1f} ms | P95 = {np.percentile(latencies['ml'], 95):.1f} ms")
    print("=" * 70)

    # Persist metrics to JSON
    eval_summary = {
        "fields": {
            f_name: {
                "raw_acc": round((raw_scores[f_name]["correct"] / raw_scores[f_name]["total"]) * 100, 1),
                "pipeline_acc": round((pipeline_scores[f_name]["correct"] / pipeline_scores[f_name]["total"]) * 100, 1),
                "lift": round(((pipeline_scores[f_name]["correct"] / pipeline_scores[f_name]["total"]) - (raw_scores[f_name]["correct"] / raw_scores[f_name]["total"])) * 100, 1),
            } for f_name in fields
        },
        "triage": {
            "safety_recall": round(recall, 1),
            "precision": round(precision, 1),
            "false_acceptance_rate": round((triage_fn / max(1, triage_tp + triage_fn)) * 100, 1),
        },
        "latency_ms": {
            "cv_p50": round(float(np.median(latencies['cv'])), 1),
            "cv_p95": round(float(np.percentile(latencies['cv'], 95)), 1),
            "ocr_p50": round(float(np.median(latencies['ocr'])), 1),
            "ocr_p95": round(float(np.percentile(latencies['ocr'], 95)), 1),
            "ml_p50": round(float(np.median(latencies['ml'])), 1),
            "ml_p95": round(float(np.percentile(latencies['ml'], 95)), 1),
        }
    }
    with open(ROOT_DIR / "data/evaluation_results.json", "w", encoding="utf-8") as out_f:
        json.dump(eval_summary, out_f, indent=2)
    print(f"[OK] Evaluation results persisted to {ROOT_DIR / 'data/evaluation_results.json'}")


if __name__ == "__main__":
    run_rigorous_evaluation(num_samples=15)