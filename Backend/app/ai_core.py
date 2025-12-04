"""
AI Core Module - YOLO Model Loading and Inference
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
import torch

from .config import (
    MODEL_PATH, FALLBACK_MODEL_PATH, CLASS_NAMES,
    INORGANIC_CLASSES, ORGANIC_CLASSES, RECYCLABLE_CLASSES,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, IMG_SIZE, COLORS, WEB_COLORS,
    get_category
)


class AICore:
    """Core AI module for waste detection and classification"""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path: Optional[Path] = None
        self.is_loaded: bool = False
        
    def load_model(self) -> bool:
        """Smart model loading - loads best.pt or fallback to yolov8n.pt"""
        try:
            # Try loading best.pt first
            if MODEL_PATH.exists():
                self.model = YOLO(str(MODEL_PATH))
                self.model_path = MODEL_PATH
            elif FALLBACK_MODEL_PATH.exists():
                self.model = YOLO(str(FALLBACK_MODEL_PATH))
                self.model_path = FALLBACK_MODEL_PATH
            else:
                print("No model file found!")
                return False
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.is_loaded = True
            print(f"Model loaded successfully from {self.model_path} on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(
        self, 
        image: np.ndarray, 
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Perform inference on image
        
        Args:
            image: Input image in BGR format (OpenCV)
            conf_threshold: Confidence threshold for filtering detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection results with bboxes, labels, confidence, category
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=IMG_SIZE,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Get class name
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                # Determine category using get_category function
                category = get_category(class_name)
                color = WEB_COLORS.get(category, "#FFFFFF")
                
                detection = {
                    "id": i,
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                    "category": category.capitalize(),  # "Organic", "Inorganic", "Recyclable"
                    "color": color
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image in BGR format
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        img_copy = image.copy()
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            
            # Get color based on category (Organic, Inorganic, Recyclable)
            category_lower = det["category"].lower()
            color = COLORS.get(category_lower, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label = f"{det['class_name'].replace('_', ' ').replace('-', ' ').title()} ({det['category']}) {int(det['confidence']*100)}%"
            
            # Calculate label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(img_copy, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)
            
            # Draw label text
            cv2.putText(img_copy, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        return img_copy
    
    def get_sorting_decision(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine sorting decision based on detections
        
        Returns:
            Sorting decision with signal info for 3 categories:
            - RED: Organic (hữu cơ)
            - GREEN: Inorganic (vô cơ)
            - BLUE: Recyclable (tái chế)
        """
        organic_count = sum(1 for d in detections if d["category"] == "Organic")
        inorganic_count = sum(1 for d in detections if d["category"] == "Inorganic")
        recyclable_count = sum(1 for d in detections if d["category"] == "Recyclable")
        
        if len(detections) == 0:
            decision = "NO_DETECTION"
            signal = "IDLE"
        elif (organic_count > 0 and inorganic_count > 0) or \
             (organic_count > 0 and recyclable_count > 0) or \
             (inorganic_count > 0 and recyclable_count > 0):
            decision = "SEPARATE_STREAMS"
            signal = "MIXED"
        elif organic_count > 0:
            decision = "ORGANIC_STREAM"
            signal = "RED"  # Red/Orange for organic
        elif inorganic_count > 0:
            decision = "INORGANIC_STREAM"
            signal = "GREEN"  # Green for inorganic
        else:
            decision = "RECYCLABLE_STREAM"
            signal = "BLUE"  # Blue for recyclable
            
        return {
            "decision": decision,
            "signal": signal,
            "organic_count": organic_count,
            "inorganic_count": inorganic_count,
            "recyclable_count": recyclable_count,
            "total_count": len(detections)
        }


# Singleton instance
ai_core = AICore()
