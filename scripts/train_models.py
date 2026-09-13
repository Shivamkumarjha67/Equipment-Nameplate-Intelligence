from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.ml.feature_extractor import NameplateFeatureExtractor
from src.ocr.extractor import NameplateExtractor
from src.vision.preprocessor import NameplatePreprocessor

ANNOTATIONS_PATH = ROOT_DIR / "data/synthetic/annotations.jsonl"
IMAGE_DIR = ROOT_DIR / "data/synthetic/images"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train():
  prep = NameplatePreprocessor()
  ocr = NameplateExtractor(gpu=False)
  feat_extractor = NameplateFeatureExtractor()

  X = []
  y = []  # 0 = ACCEPTED, 1 = MANUAL_REVIEW_REQUIRED
  plausibility_features = (
      []
  )  # [voltage, current, power] for IsolationForest fitting

  print("Extracting features from dataset to train ML models...")
  with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
      if idx >= 150:  # Train on first 150 samples for speed
        break

      rec = json.loads(line)
      img_path = IMAGE_DIR / rec["image_file"]
      image = cv2.imread(str(img_path))
      if image is None:
        continue

      gt_specs = rec["specs"]
      prep_res = prep.process(image)
      tokens = ocr.run_ocr(prep_res["enhanced_gray"])
      preds = ocr.parse_fields(tokens)

      feats = feat_extractor.extract_features(prep_res, tokens, preds)
      X.append(feat_extractor.to_feature_vector(feats))

      # Ground truth label: if any numerical field has >10% error or is None, label = 1 (Review)
      review_needed = 0
      for field in ["voltage", "frequency", "power", "current"]:
        gt_v = gt_specs.get(field)
        pr_v = preds.get(field, {}).get("value")
        if pr_v is None:
          review_needed = 1
          break
        if abs(float(gt_v) - float(pr_v)) > (0.1 * float(gt_v)):
          review_needed = 1
          break
      y.append(review_needed)

      # Collect physical parameter triplets for anomaly fitting
      v_val = preds.get("voltage", {}).get("value") or 400.0
      i_val = preds.get("current", {}).get("value") or 10.0
      p_val = preds.get("power", {}).get("value") or 5.5
      plausibility_features.append([v_val, i_val, p_val])

      if (idx + 1) % 25 == 0:
        print(f"  Extracted {idx + 1} training vectors...")

  X = np.array(X)
  y = np.array(y)

  print(
      f"\nDataset prepared: {len(y)} samples. Class distribution: Accepted (0):"
      f" {np.sum(y == 0)}, Needs Review (1): {np.sum(y == 1)}"
  )

  # Train/Val Split for the Classifier
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # 1. Train Random Forest Review Classifier
  rf_model = RandomForestClassifier(
      n_estimators=50, max_depth=5, random_state=42
  )
  rf_model.fit(X_train, y_train)

  preds_rf = rf_model.predict(X_test)
  print("\n--- Review Classifier Validation Performance ---")
  print(classification_report(y_test, preds_rf, zero_division=0))

  # 2. Train Isolation Forest for Physical Anomaly Detection
  iso_forest = IsolationForest(contamination=0.1, random_state=42)
  iso_forest.fit(np.array(plausibility_features))

  # Save trained models
  joblib.dump(rf_model, MODELS_DIR / "review_classifier.joblib")
  joblib.dump(iso_forest, MODELS_DIR / "isolation_forest.joblib")
  print(f"\n[OK] Models successfully trained and serialized into {MODELS_DIR}")


if __name__ == "__main__":
  train()