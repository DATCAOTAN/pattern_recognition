"""
Image Processing Module - Resize, Normalize, Format Conversion
"""
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional

from .config import IMG_SIZE


def resize_image(image: np.ndarray, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio
    
    Args:
        image: Input image (BGR format)
        target_size: Target size for width and height
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create square canvas and center the image
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to 0-1 range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR format to RGB format for web display
    
    Args:
        image: BGR image
        
    Returns:
        RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB format to OpenCV BGR format
    
    Args:
        image: RGB image
        
    Returns:
        BGR image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to OpenCV format
    
    Args:
        image_bytes: Image data in bytes
        
    Returns:
        OpenCV image (BGR format)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def cv2_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
    """
    Convert OpenCV image to base64 string for web display
    
    Args:
        image: OpenCV image (BGR format)
        format: Output format (jpeg, png)
        
    Returns:
        Base64 encoded image string
    """
    # Convert BGR to RGB for proper color display
    rgb_image = bgr_to_rgb(image)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format.upper())
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/{format};base64,{img_base64}"


def base64_to_cv2(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        OpenCV image (BGR format)
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode base64
    img_bytes = base64.b64decode(base64_string)
    
    # Convert to OpenCV
    return bytes_to_cv2(img_bytes)


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for model input
    
    Args:
        image: Raw input image
        
    Returns:
        Preprocessed image ready for model
    """
    # Resize to model input size
    processed = resize_image(image, IMG_SIZE)
    
    return processed


def extract_frame_from_video(video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
    """
    Extract a specific frame from video file
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract
        
    Returns:
        Frame as OpenCV image or None
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None


def get_video_info(video_path: str) -> dict:
    """
    Get video file information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video info
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "Cannot open video"}
    
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return info
