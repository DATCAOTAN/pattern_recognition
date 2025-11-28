"""
Configuration settings for EcoSort AI Backend
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "runs" / "exp3_final_p100" / "weights" / "best.pt"
FALLBACK_MODEL_PATH = BASE_DIR / "yolov8n.pt"

# Model settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# Class names mapping (from data_kaggle.yaml)
CLASS_NAMES = {
    0: "bag",
    1: "banana_peel", 
    2: "bottle",
    3: "can",
    4: "eggshell",
    5: "leaves"
}

# Vietnamese class names
CLASS_NAMES_VI = {
    "bag": "Túi",
    "banana_peel": "Vỏ chuối",
    "bottle": "Chai",
    "can": "Lon",
    "eggshell": "Vỏ trứng",
    "leaves": "Lá cây"
}

# Category mapping (Business Logic)
INORGANIC_CLASSES = ["bag", "bottle", "can"]
ORGANIC_CLASSES = ["banana_peel", "eggshell", "leaves"]

# Color coding for visualization (BGR format for OpenCV)
COLORS = {
    "inorganic": (0, 255, 0),   # Green
    "organic": (0, 165, 255),    # Orange
}

# Color coding for web (RGB format)
WEB_COLORS = {
    "inorganic": "#00FF00",  # Green
    "organic": "#FF6600",     # Orange
}

# Logs directory
LOGS_DIR = BASE_DIR / "Backend" / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Upload directory
UPLOAD_DIR = BASE_DIR / "Backend" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Snapshot directory
SNAPSHOT_DIR = BASE_DIR / "Backend" / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)
