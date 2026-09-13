from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
SYNTHETIC_IMAGES = SYNTHETIC_DIR / "images"
ANNOTATIONS_PATH = SYNTHETIC_DIR / "annotations.jsonl"
MODELS_DIR = BASE_DIR / "models"

# Quality & Extraction Constants
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 100.0))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.75))

# Neo4j Database Credentials
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# GEMINI API Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")