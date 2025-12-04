"""
Configuration settings for EcoSort AI Backend
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "runs" / "yolov8n_waste" / "weights" / "best.pt"
FALLBACK_MODEL_PATH = BASE_DIR / "yolov8n.pt"

# Model settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# ========== 39 CLASSES MAPPING ==========
# Class names mapping (from data_final)
CLASS_NAMES = {
    # Organic Waste (0-31) - 32 classes
    0: "Apple", 1: "Apple-core", 2: "Apple-peel", 3: "Bone", 4: "Bone-fish",
    5: "Bread", 6: "Bun", 7: "Egg-hard", 8: "Egg-scramble", 9: "Egg-shell",
    10: "Egg-steam", 11: "Egg-yolk", 12: "Fish", 13: "Meat", 14: "Mussel",
    15: "Mussel-shell", 16: "Noodle", 17: "Orange", 18: "Orange-peel",
    19: "Other-waste", 20: "Pancake", 21: "Pasta", 22: "Pear", 23: "Pear-core",
    24: "Pear-peel", 25: "Potato", 26: "Rice", 27: "Shrimp", 28: "Shrimp-shell",
    29: "Tofu", 30: "Tomato", 31: "Vegetable",
    # Inorganic (32-33) - 2 classes
    32: "plastic_bag", 33: "styrofoam",
    # Recyclable (34-38) - 5 classes
    34: "Cardboard", 35: "Glass", 36: "Metal", 37: "Paper", 38: "Plastic"
}

# Vietnamese class names
CLASS_NAMES_VI = {
    # Organic
    "Apple": "Táo", "Apple-core": "Lõi táo", "Apple-peel": "Vỏ táo",
    "Bone": "Xương", "Bone-fish": "Xương cá", "Bread": "Bánh mì", "Bun": "Bánh bao",
    "Egg-hard": "Trứng luộc", "Egg-scramble": "Trứng chiên", "Egg-shell": "Vỏ trứng",
    "Egg-steam": "Trứng hấp", "Egg-yolk": "Lòng đỏ trứng", "Fish": "Cá", "Meat": "Thịt",
    "Mussel": "Trai", "Mussel-shell": "Vỏ trai", "Noodle": "Mì", "Orange": "Cam",
    "Orange-peel": "Vỏ cam", "Other-waste": "Rác khác", "Pancake": "Bánh kếp",
    "Pasta": "Mì ống", "Pear": "Lê", "Pear-core": "Lõi lê", "Pear-peel": "Vỏ lê",
    "Potato": "Khoai tây", "Rice": "Cơm", "Shrimp": "Tôm", "Shrimp-shell": "Vỏ tôm",
    "Tofu": "Đậu hũ", "Tomato": "Cà chua", "Vegetable": "Rau củ",
    # Inorganic
    "plastic_bag": "Túi nhựa", "styrofoam": "Xốp",
    # Recyclable
    "Cardboard": "Bìa carton", "Glass": "Thủy tinh", "Metal": "Kim loại",
    "Paper": "Giấy", "Plastic": "Nhựa"
}

# ========== CATEGORY MAPPING (3 CATEGORIES) ==========
# Organic classes (Hữu cơ) - index 0-31
ORGANIC_CLASSES = [
    "Apple", "Apple-core", "Apple-peel", "Bone", "Bone-fish", "Bread", "Bun",
    "Egg-hard", "Egg-scramble", "Egg-shell", "Egg-steam", "Egg-yolk", "Fish",
    "Meat", "Mussel", "Mussel-shell", "Noodle", "Orange", "Orange-peel",
    "Other-waste", "Pancake", "Pasta", "Pear", "Pear-core", "Pear-peel",
    "Potato", "Rice", "Shrimp", "Shrimp-shell", "Tofu", "Tomato", "Vegetable"
]

# Inorganic classes (Vô cơ) - index 32-33
INORGANIC_CLASSES = ["plastic_bag", "styrofoam"]

# Recyclable classes (Tái chế) - index 34-38
RECYCLABLE_CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]

# Function to get category from class name
def get_category(class_name):
    """Get category (organic/inorganic/recyclable) from class name"""
    if class_name in ORGANIC_CLASSES:
        return "organic"
    elif class_name in INORGANIC_CLASSES:
        return "inorganic"
    elif class_name in RECYCLABLE_CLASSES:
        return "recyclable"
    return "unknown"

# Color coding for visualization (BGR format for OpenCV)
COLORS = {
    "organic": (0, 165, 255),      # Orange
    "inorganic": (128, 128, 128),  # Gray
    "recyclable": (0, 255, 0),     # Green
}

# Color coding for web (RGB/Hex format)
WEB_COLORS = {
    "organic": "#FF6600",      # Orange
    "inorganic": "#808080",    # Gray
    "recyclable": "#00FF00",   # Green
}

# Category Vietnamese names
CATEGORY_NAMES_VI = {
    "organic": "Hữu cơ",
    "inorganic": "Vô cơ", 
    "recyclable": "Tái chế"
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
