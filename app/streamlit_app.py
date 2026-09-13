from __future__ import annotations

from pathlib import Path
import sys

# Ensure root directory is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from src.config import MODELS_DIR
from src.rag.query_engine import GraphQueryEngine
from src.graph.connection import Neo4jConnector
from src.graph.ingestor import EquipmentGraphIngestor
from src.ml.feature_extractor import NameplateFeatureExtractor
from src.ocr.extractor import NameplateExtractor
from src.vision.preprocessor import NameplatePreprocessor
import streamlit as st

st.set_page_config(
    page_title="Equipment Nameplate Intelligence",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def load_components():
  prep = NameplatePreprocessor()
  ocr = NameplateExtractor(gpu=False)
  feat_extractor = NameplateFeatureExtractor()
  rf_model = joblib.load(MODELS_DIR / "review_classifier.joblib")
  iso_model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
  connector = Neo4jConnector()
  ingestor = EquipmentGraphIngestor(connector)
  rag_engine = GraphQueryEngine(connector)
  return prep, ocr, feat_extractor, rf_model, iso_model, connector, ingestor, rag_engine


prep, ocr, feat_extractor, rf_model, iso_model, connector, ingestor, rag_engine = (
    load_components()
)

st.title("⚙️ Equipment Nameplate Intelligence System")
st.markdown(
    "**Offline-first Multimodal Asset Intelligence** combining geometric"
    " OpenCV, OCR, Scikit-Learn validation, and Neo4j Knowledge Graph."
)

tab1, tab2 = st.tabs(
    ["🔍 Live Nameplate Inspector", "📊 Plant Asset Knowledge Graph"]
)

# ==========================================
# TAB 1: INSPECTOR & INGESTION PIPELINE
# ==========================================
with tab1:
  col_upload, col_meta = st.columns([2, 1])

  with col_upload:
    uploaded_file = st.file_uploader(
        "Upload equipment nameplate photograph (.jpg, .png)",
        type=["jpg", "jpeg", "png"],
    )

  with col_meta:
    location_code = st.selectbox(
        "Plant Asset Location",
        ["BAY-01", "BAY-02", "PUMP-SKID-A", "BOILER-HOUSE", "SUBSTATION-3"],
    )
    maint_note = st.text_input(
        "Maintenance Log Note (Optional)", "Routine scheduled visual audit"
    )

  if uploaded_file is not None:
    # Convert uploaded file to OpenCV BGR format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("1. Computer Vision & Geometric Rectification")
    with st.spinner("Executing OpenCV Homography & CLAHE Enhancement..."):
      prep_res = prep.process(img_bgr)

    col_raw, col_warp, col_enh = st.columns(3)
    with col_raw:
      st.image(
          cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
          caption="Original Raw Capture",
          width="stretch",
      )
    with col_warp:
      st.image(
          cv2.cvtColor(prep_res["warped_color"], cv2.COLOR_BGR2RGB),
          caption=(
              "Rectified Homography Patch"
              if prep_res["corners_detected"]
              else "Fallback (Full Image)"
          ),
          width="stretch",
      )
    with col_enh:
      st.image(
          prep_res["enhanced_gray"],
          caption="CLAHE Contrast Normalization",
          width="stretch",
          clamp=True,
      )

    # ==========================================
    # STEP 2: OCR & EXTRACTION
    # ==========================================
    st.subheader("2. OCR Tokenization & Technical Field Extraction")
    with st.spinner("Extracting Tokens & Parsing Industrial Parameters..."):
        tokens = ocr.run_ocr(prep_res["enhanced_gray"])
        preds = ocr.parse_fields(tokens)

    # 1. Show the extracted matrix directly under Step 2
    specs_display = []
    for field, data in preds.items():
        specs_display.append({
            "Parameter": field.upper(),
            "Extracted Value": data.get("value"),
            "Model Confidence": f"{data.get('confidence', 0.0) * 100:.1f}%",
            "Status": "Extracted" if data.get("value") is not None else "Missing",
        })
    st.dataframe(pd.DataFrame(specs_display), width='stretch')

    # 2. Add an expander showing raw OCR tokens detected in Step 2
    with st.expander("📝 View Raw OCR Detections & Token Bounding Boxes"):
        raw_tokens_df = [
            {"Detected Text": t["text"], "Confidence": f"{t['confidence']*100:.1f}%"}
            for t in tokens
        ]
        st.dataframe(pd.DataFrame(raw_tokens_df), width='stretch')

    # ==========================================
    # STEP 3: ML QUALITY GATE & TRIAGE
    # ==========================================
    st.subheader("3. Scikit-Learn Triage & Plausibility Validation")

    feats = feat_extractor.extract_features(prep_res, tokens, preds)
    x_vec = [feat_extractor.to_feature_vector(feats)]
    triage_pred = rf_model.predict(x_vec)[0]  # 0 = Accepted, 1 = Review

    v_val = preds.get("voltage", {}).get("value") or 400.0
    i_val = preds.get("current", {}).get("value") or 10.0
    p_val = preds.get("power", {}).get("value") or 5.5
    anomaly_score = iso_model.predict([[v_val, i_val, p_val]])[0]

    col_triage, col_anomaly, col_blur, col_conf = st.columns(4)
    with col_triage:
        if triage_pred == 0:
            st.success("✅ TRIAGE: ACCEPTED")
        else:
            st.error("⚠️ TRIAGE: MANUAL REVIEW REQUIRED")

    with col_anomaly:
        if anomaly_score == 1:
            st.info("⚡ PHYSICAL SPECS: PLAUSIBLE")
        else:
            st.warning("⚠️ PHYSICAL SPECS: ANOMALOUS RATIO")

    with col_blur:
        st.metric(label="Laplacian Blur Score", value=f"{feats['blur_score']:.1f}")

    with col_conf:
        st.metric(label="Mean OCR Confidence", value=f"{feats['mean_ocr_conf'] * 100:.1f}%")

    # Graph Ingestion Action
    st.subheader("4. Knowledge Graph Mutation")
    if triage_pred == 0:
      if st.button("🚀 Ingest Asset into Neo4j AuraDB"):
        with st.spinner("Executing Cypher MERGE transactions..."):
          success = ingestor.ingest_nameplate(
              preds, location_code=location_code, maintenance_note=maint_note
          )
          if success:
            st.success(
                f"Asset {preds.get('serial_number', {}).get('value')} linked to"
                f" {location_code} in Neo4j AuraDB."
            )
          else:
            st.error("Missing Serial Number or Manufacturer to ingest.")
    else:
      st.warning(
          "Automated ingestion blocked due to low confidence or missing fields."
          " Manual verification required."
      )

# ==========================================
# TAB 2: NEO4J ASSET EXPLORER & QUERY
# ==========================================
with tab2:
  st.subheader("Plant Asset Topology & Natural Language Query")

  # --- SECTION A: Natural Language Graph-RAG (Gemini) ---
  st.markdown("### 💬 Ask the Plant Knowledge Graph (Powered by Gemini)")

  default_questions = [
      "Which 5.5 kW motors are located in BAY-01?",
      "List all equipment manufactured by SIEMENS or ABB with their locations.",
      "Show all equipment operating above 400 Volts.",
      "What are the serial numbers of machines in PUMP-SKID-A?",
  ]
  selected_sample = st.selectbox(
      "Quick Example Prompts:", ["Custom Question"] + default_questions
  )

  user_nl_query = st.text_input(
      "Enter your question:",
      value="" if selected_sample == "Custom Question" else selected_sample,
      placeholder="e.g. Which 3.7 kW assets are located in BAY-02?",
  )

  if st.button("🔎 Run Graph-RAG Query", key="btn_rag"):
    if not user_nl_query.strip():
      st.warning("Please enter a valid question.")
    else:
      with st.spinner("Gemini is formulating Cypher query & querying AuraDB..."):
        try:
          rag_output = rag_engine.answer_question(user_nl_query)

          st.success("### Answer:")
          st.write(rag_output["summary"])

          with st.expander("🛠️ View Graph-RAG Execution Trace"):
            st.markdown("**Generated Cypher Query:**")
            st.code(rag_output["cypher_query"], language="cypher")
            st.markdown("**Retrieved Graph Records:**")
            st.json(rag_output["raw_records"])
        except Exception as err:
          st.error(f"Query failed: {err}")

  st.divider()

  # --- SECTION B: Full Inventory Table ---
  st.markdown("### 📋 Complete Asset Inventory")
  col_refresh, _ = st.columns([1, 4])
  with col_refresh:
    refresh_btn = st.button("🔄 Refresh Inventory Table")

  query = """
    MATCH (e:Equipment)-[:MANUFACTURED_BY]->(m:Manufacturer)
    MATCH (e)-[:LOCATED_AT]->(l:PlantArea)
    MATCH (e)-[:HAS_SPEC]->(s:Specification)
    RETURN e.serial_number AS `Serial Number`,
           e.model AS `Model`,
           m.name AS `Manufacturer`,
           l.code AS `Plant Location`,
           s.power_kw AS `Power (kW)`,
           s.voltage_v AS `Voltage (V)`,
           s.current_a AS `Current (A)`,
           s.frequency_hz AS `Frequency (Hz)`
    """

  try:
    with connector.driver.session() as session:
      results = session.run(query)
      records = [r.data() for r in results]

    if records:
      df_graph = pd.DataFrame(records)
      st.dataframe(df_graph, width='stretch')

      col_m1, col_m2, col_m3 = st.columns(3)
      col_m1.metric("Total Equipment Registered", len(df_graph))
      col_m2.metric(
          "Active Manufacturers", df_graph["Manufacturer"].nunique()
      )
      col_m3.metric(
          "Plant Locations Monitored", df_graph["Plant Location"].nunique()
      )
    else:
      st.info(
          "No records found in Neo4j AuraDB yet. Ingest samples from Tab 1."
      )
  except Exception as e:
    st.error(f"Failed to query Neo4j: {e}")