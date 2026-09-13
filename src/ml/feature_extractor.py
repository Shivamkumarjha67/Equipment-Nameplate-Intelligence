from __future__ import annotations

import math
import numpy as np


class NameplateFeatureExtractor:

  def extract_features(
      self, prep_res: dict, tokens: list[dict], parsed_fields: dict
  ) -> dict:
    """Extracts tabular features from preprocessor output and OCR fields."""
    # 1. Image Quality Metrics
    blur_score = float(prep_res.get("blur_score", 0.0))
    corners_detected = 1 if prep_res.get("corners_detected", False) else 0

    # 2. OCR Token Level Confidence & Entropy
    confidences = [t["confidence"] for t in tokens] if tokens else [0.0]
    mean_ocr_conf = float(np.mean(confidences))
    min_ocr_conf = float(np.min(confidences))
    total_tokens = len(tokens)
    total_chars = sum(len(t["text"]) for t in tokens)

    # 3. Field Completeness
    critical_fields = [
        "voltage",
        "current",
        "power",
        "frequency",
        "manufacturer",
        "serial_number",
    ]
    extracted_count = sum(
        1 for f in critical_fields if parsed_fields.get(f, {}).get("value")
    )
    completeness_ratio = extracted_count / len(critical_fields)

    # 4. Electro-Mechanical Plausibility Ratio: P = sqrt(3) * V * I * pf * eff
    # Theoretical Power (kW) approx = 1.732 * V * I * 0.85 * 0.90 / 1000 ≈ (V * I) / 755
    v = parsed_fields.get("voltage", {}).get("value")
    i = parsed_fields.get("current", {}).get("value")
    p = parsed_fields.get("power", {}).get("value")

    if v and i and p and v > 0 and i > 0 and p > 0:
      expected_power_kw = (v * i * math.sqrt(3) * 0.85 * 0.88) / 1000.0
      # Relative discrepancy ratio between extracted power and theoretical power
      power_discrepancy = abs(p - expected_power_kw) / max(
          expected_power_kw, 0.1
      )
    else:
      power_discrepancy = 5.0  # High penalty if critical fields missing

    return {
        "blur_score": blur_score,
        "corners_detected": corners_detected,
        "mean_ocr_conf": mean_ocr_conf,
        "min_ocr_conf": min_ocr_conf,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "extracted_count": extracted_count,
        "completeness_ratio": completeness_ratio,
        "power_discrepancy": min(power_discrepancy, 10.0),
    }

  def to_feature_vector(self, features: dict) -> list[float]:
    """Returns an ordered feature list for model input."""
    return [
        features["blur_score"],
        features["corners_detected"],
        features["mean_ocr_conf"],
        features["min_ocr_conf"],
        features["total_tokens"],
        features["total_chars"],
        features["extracted_count"],
        features["completeness_ratio"],
        features["power_discrepancy"],
    ]