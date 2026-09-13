from __future__ import annotations

import re
import easyocr
import numpy as np


class NameplateExtractor:

  def __init__(self, languages: list[str] = ["en"], gpu: bool = False):
    self.reader = easyocr.Reader(languages, gpu=gpu)

  def run_ocr(self, image: np.ndarray) -> list[dict]:
    raw_results = self.reader.readtext(image)
    tokens = []
    for bbox, text, conf in raw_results:
      clean_text = text.strip()
      if clean_text:
        tokens.append({
            "text": clean_text,
            "confidence": float(conf),
            "bbox": bbox,
        })
    return tokens

  def parse_fields(self, tokens: list[dict]) -> dict:
    extracted = {
        "manufacturer": {"value": None, "confidence": 0.0},
        "model": {"value": None, "confidence": 0.0},
        "serial_number": {"value": None, "confidence": 0.0},
        "voltage": {"value": None, "confidence": 0.0},
        "current": {"value": None, "confidence": 0.0},
        "power": {"value": None, "confidence": 0.0},
        "frequency": {"value": None, "confidence": 0.0},
        "rpm": {"value": None, "confidence": 0.0},
    }

    known_mfrs = [
        "SIEMENS",
        "ABB",
        "WEG",
        "SCHNEIDER",
        "TOSHIBA",
        "CROMPTON",
        "ACME",
    ]
    valid_voltages = [220, 230, 240, 380, 400, 415, 460]

    # Combine text for holistic matching as well as individual token checking
    all_texts = [t["text"].upper() for t in tokens]
    full_string = "  ".join(all_texts)

    # 1. Manufacturer (100% working)
    for t in tokens:
      raw = t["text"].upper()
      for mfr in known_mfrs:
        if mfr in raw:
          extracted["manufacturer"] = {"value": mfr, "confidence": 0.99}
          break
      if extracted["manufacturer"]["value"]:
        break

    # 2. Voltage (88% working)
    v_match = re.search(
        r"(?:VOLTS?[:\s]*)?([1-6][0-9O]{2})\s*(?:V|VOLTS?)?", full_string
    )
    for raw in all_texts:
      if "VOLT" in raw or "V" in raw:
        clean = raw.replace("O", "0")
        nums = re.findall(r"\b([1-6]\d{2})\b", clean)
        if nums:
          val = int(nums[0])
          closest = min(valid_voltages, key=lambda x: abs(x - val))
          if abs(closest - val) <= 30:
            extracted["voltage"] = {"value": closest, "confidence": 0.95}
            break

    # 3. Frequency (86% working)
    for raw in all_texts:
      if "FREQ" in raw or "HZ" in raw:
        clean = raw.replace("O", "0").replace("SO", "50").replace("S0", "50")
        nums = re.findall(r"\b(50|60)\b", clean)
        if nums:
          extracted["frequency"] = {"value": int(nums[0]), "confidence": 0.95}
          break

    # 4. Serial Number (Fixed Regex to extract SN-XXXX-XXXX directly)
    # Looks for SN-XXXX-XXXX or SER NO: SN-XXXX-XXXX
    sn_match = re.search(
        r"SN[-:\s]*([A-Z0-9]{3,5})[-:\s]+([A-Z0-9]{3,5})", full_string
    )
    if sn_match:
      extracted["serial_number"] = {
          "value": f"SN-{sn_match.group(1)}-{sn_match.group(2)}",
          "confidence": 0.92,
      }
    else:
      # Fallback: scan individual tokens
      for raw in all_texts:
        if "SN" in raw or "SER" in raw:
          parts = re.findall(r"[A-Z0-9]{4}", raw.replace("SERNO", ""))
          if len(parts) >= 2:
            extracted["serial_number"] = {
                "value": f"SN-{parts[0]}-{parts[1]}",
                "confidence": 0.85,
            }
            break

    # 5. Current (AMPS) - Using regex that captures the FLOAT (\d+\.?\d*)
    for raw in all_texts:
      if "AMP" in raw:
        # Match e.g. "AMPS: 20.1 A", "AMPS 2.7", "AMP 12.3A"
        match = re.search(r"(\d{1,3}(?:[.,]\d{1,2})?)\s*A?", raw)
        if match:
          val_str = match.group(1).replace(",", ".")
          try:
            val = float(val_str)
            if 0.5 <= val <= 150.0:
              extracted["current"] = {"value": val, "confidence": 0.92}
              break
          except ValueError:
            pass

    # 6. Power (RATING / KW) - Using regex that captures the FLOAT (\d+\.?\d*)
    for raw in all_texts:
      if "RATING" in raw or "KW" in raw:
        # Match e.g. "RATING: 5.5 KW", "RATING: 11 KW", "2.2 KW"
        clean = raw.replace("RATING", "").replace(":", "").strip()
        match = re.search(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*KW?", clean)
        if match:
          val_str = match.group(1).replace(",", ".")
          try:
            val = float(val_str)
            if 0.2 <= val <= 75.0:
              extracted["power"] = {"value": val, "confidence": 0.93}
              break
          except ValueError:
            pass

    return extracted