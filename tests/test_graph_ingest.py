from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import joblib
from src.vision.preprocessor import NameplatePreprocessor
from src.ocr.extractor import NameplateExtractor
from src.ml.feature_extractor import NameplateFeatureExtractor
from src.graph.connection import Neo4jConnector
from src.graph.ingestor import EquipmentGraphIngestor
from src.config import ANNOTATIONS_PATH, SYNTHETIC_IMAGES, MODELS_DIR

def main():
    print("Connecting to Neo4j AuraDB...")
    connector = Neo4jConnector()
    if not connector.verify_connection():
        print("[!] Aborting: Could not reach Neo4j AuraDB. Check URI, user, and password.")
        return

    print("[✓] Connected to Neo4j AuraDB successfully.")
    connector.init_schema()

    ingestor = EquipmentGraphIngestor(connector)
    prep = NameplatePreprocessor()
    ocr = NameplateExtractor(gpu=False)
    feat_extractor = NameplateFeatureExtractor()
    rf_model = joblib.load(MODELS_DIR / "review_classifier.joblib")

    ingested_count = 0
    target_samples = 10
    areas = ["BAY-01", "BAY-02", "PUMP-SKID-A", "BOILER-HOUSE"]

    print(f"\nProcessing first {target_samples} samples through the full pipeline...")

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if ingested_count >= target_samples:
                break

            record = json.loads(line)
            img_path = SYNTHETIC_IMAGES / record["image_file"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 1. Computer Vision & OCR
            prep_res = prep.process(img)
            tokens = ocr.run_ocr(prep_res["enhanced_gray"])
            preds = ocr.parse_fields(tokens)

            # 2. Scikit-Learn Quality Gate
            feats = feat_extractor.extract_features(prep_res, tokens, preds)
            x_vec = [feat_extractor.to_feature_vector(feats)]
            triage_label = rf_model.predict(x_vec)[0]  # 0 = ACCEPTED, 1 = REVIEW

            status_str = "ACCEPTED" if triage_label == 0 else "MANUAL_REVIEW"
            print(f"Sample {idx + 1} ({record['image_file']}) -> Triage: {status_str}")

            # 3. Ingest only validated records into AuraDB
            if triage_label == 0:
                loc = areas[ingested_count % len(areas)]
                success = ingestor.ingest_nameplate(
                    preds,
                    location_code=loc,
                    maintenance_note=f"Routine visual inspection log for {record['image_file']}"
                )
                if success:
                    ingested_count += 1
                    print(f"  --> [INGESTED] {preds['serial_number']['value']} into {loc}")

    # 4. Verify in AuraDB with a Cypher Query
    print("\n--- Verifying Ingested Assets in AuraDB ---")
    query = """
    MATCH (e:Equipment)-[:MANUFACTURED_BY]->(m:Manufacturer)
    MATCH (e)-[:LOCATED_AT]->(l:PlantArea)
    MATCH (e)-[:HAS_SPEC]->(s:Specification)
    RETURN e.serial_number AS serial, m.name AS manufacturer, l.code AS location, s.power_kw AS power, s.voltage_v AS voltage
    LIMIT 5
    """
    with connector.driver.session() as session:
        results = session.run(query)
        for r in results:
            print(f"Asset: {r['serial']} | Mfr: {r['manufacturer']} | Location: {r['location']} | {r['power']} kW @ {r['voltage']} V")

    connector.close()
    print("\n[✓] Neo4j AuraDB integration verified.")

if __name__ == "__main__":
    main()