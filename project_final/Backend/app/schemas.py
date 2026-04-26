"""
Pydantic Schemas for API Request/Response
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    """Single detection result"""
    id: int
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    category: str  # Organic or Inorganic
    color: str  # Hex color for visualization


class SortingDecision(BaseModel):
    """Sorting decision for detected items"""
    decision: str
    signal: str  # GREEN, RED, MIXED, IDLE
    inorganic_count: int
    organic_count: int
    total_count: int


class PredictResponse(BaseModel):
    """Response for prediction endpoint"""
    success: bool
    detections: List[Detection]
    sorting_decision: SortingDecision
    processed_image: Optional[str] = None  # Base64 encoded image
    inference_time_ms: float


class SystemStatus(BaseModel):
    """System status response"""
    health: str
    model_status: str
    model_path: Optional[str]
    cpu_percent: float
    ram: Dict[str, Any]
    gpu: Dict[str, Any]
    system_info: Dict[str, Any]


class LogEntry(BaseModel):
    """Single log entry"""
    id: int
    timestamp: str
    formatted_time: str
    class_name: str
    category: str
    confidence: float


class LogsResponse(BaseModel):
    """Response for logs endpoint"""
    logs: List[LogEntry]
    total_count: int


class StatisticsResponse(BaseModel):
    """Response for statistics endpoint"""
    total_detections: int
    inorganic_count: int
    organic_count: int
    class_counts: Dict[str, int]
    inorganic_percentage: float
    organic_percentage: float
    session_id: str


class VideoFrameRequest(BaseModel):
    """Request for video frame processing"""
    frame_data: str  # Base64 encoded frame
    confidence_threshold: float = 0.5
    class_filter: Optional[List[str]] = None


class ConfigUpdate(BaseModel):
    """Configuration update request"""
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    class_filter: Optional[List[str]] = None
