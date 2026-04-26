"""
EcoSort AI - Main FastAPI Application
Waste Classification System with YOLO
"""
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .ai_core import ai_core
from .image_processing import (
    bytes_to_cv2, cv2_to_base64, base64_to_cv2,
    preprocess_for_model, get_video_info
)
from .log_manager import log_manager
from .system_monitor import system_monitor
from .config import (
    UPLOAD_DIR, SNAPSHOT_DIR, LOGS_DIR, CLASS_NAMES,
    CONFIDENCE_THRESHOLD, CLASS_NAMES_VI
)
from .schemas import (
    PredictResponse, SystemStatus, LogsResponse,
    StatisticsResponse, Detection, SortingDecision, BoundingBox
)

# Create FastAPI app
app = FastAPI(
    title="EcoSort AI API",
    description="Intelligent Waste Classification System powered by YOLOv8",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections for real-time streaming
active_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("=" * 50)
    print("🌿 EcoSort AI - Starting up...")
    print("=" * 50)
    
    # Load AI model
    success = ai_core.load_model()
    if success:
        system_monitor.set_model_status("Ready", str(ai_core.model_path))
        print("✅ AI Model loaded successfully")
    else:
        system_monitor.set_model_status("Failed to load")
        print("❌ Failed to load AI model")
    
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    log_manager.save_session()
    print("🌿 EcoSort AI - Shutting down...")


# ============== PREDICTION ENDPOINTS ==============

@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    draw_boxes: bool = Query(default=True)
):
    """
    Predict waste classification from uploaded image
    
    - **file**: Image file (jpg, png, jpeg)
    - **confidence**: Confidence threshold (0.0-1.0)
    - **draw_boxes**: Whether to return image with drawn bounding boxes
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPG, PNG, or JPEG.")
    
    try:
        # Read and decode image
        contents = await file.read()
        image = bytes_to_cv2(contents)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Perform inference
        start_time = time.time()
        detections = ai_core.predict(image, conf_threshold=confidence)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get sorting decision
        sorting = ai_core.get_sorting_decision(detections)
        
        # Log detections
        log_manager.add_batch_logs(detections)
        
        # Prepare response
        processed_image = None
        if draw_boxes and detections:
            drawn_image = ai_core.draw_detections(image, detections)
            processed_image = cv2_to_base64(drawn_image)
        elif draw_boxes:
            processed_image = cv2_to_base64(image)
        
        return PredictResponse(
            success=True,
            detections=[Detection(**d) for d in detections],
            sorting_decision=SortingDecision(**sorting),
            processed_image=processed_image,
            inference_time_ms=round(inference_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    frame_skip: int = Query(default=5, ge=1, description="Process every N frames")
):
    """
    Process video file and return detection results for each processed frame
    
    - **file**: Video file (mp4, avi)
    - **confidence**: Confidence threshold
    - **frame_skip**: Process every N frames (for performance)
    """
    allowed_types = ["video/mp4", "video/avi", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Use MP4 or AVI.")
    
    try:
        # Save uploaded video temporarily
        video_path = UPLOAD_DIR / f"temp_{uuid.uuid4()}.mp4"
        contents = await file.read()
        with open(video_path, "wb") as f:
            f.write(contents)
        
        # Get video info
        video_info = get_video_info(str(video_path))
        
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        frame_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                detections = ai_core.predict(frame, conf_threshold=confidence)
                sorting = ai_core.get_sorting_decision(detections)
                
                frame_results.append({
                    "frame_number": frame_count,
                    "detections": detections,
                    "sorting_decision": sorting
                })
                
                # Log detections
                log_manager.add_batch_logs(detections)
            
            frame_count += 1
        
        cap.release()
        
        # Cleanup temp file
        os.remove(video_path)
        
        return {
            "success": True,
            "video_info": video_info,
            "total_frames_processed": len(frame_results),
            "frame_results": frame_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/frame")
async def predict_frame(
    frame_data: str,
    confidence: float = Query(default=0.5, ge=0.0, le=1.0)
):
    """
    Predict from base64 encoded frame (for real-time camera stream)
    
    - **frame_data**: Base64 encoded image frame
    - **confidence**: Confidence threshold
    """
    try:
        # Decode base64 frame
        image = base64_to_cv2(frame_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode frame")
        
        # Perform inference
        start_time = time.time()
        detections = ai_core.predict(image, conf_threshold=confidence)
        inference_time = (time.time() - start_time) * 1000
        
        # Get sorting decision
        sorting = ai_core.get_sorting_decision(detections)
        
        # Draw boxes on image
        drawn_image = ai_core.draw_detections(image, detections)
        processed_image = cv2_to_base64(drawn_image)
        
        # Log detections
        if detections:
            log_manager.add_batch_logs(detections)
        
        return {
            "success": True,
            "detections": detections,
            "sorting_decision": sorting,
            "processed_image": processed_image,
            "inference_time_ms": round(inference_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== WEBSOCKET FOR REAL-TIME STREAMING ==============

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera stream processing
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            frame_data = data.get("frame")
            confidence = data.get("confidence", 0.5)
            
            if frame_data:
                # Decode and process frame
                image = base64_to_cv2(frame_data)
                
                if image is not None:
                    # Perform inference
                    detections = ai_core.predict(image, conf_threshold=confidence)
                    sorting = ai_core.get_sorting_decision(detections)
                    
                    # Draw boxes
                    drawn_image = ai_core.draw_detections(image, detections)
                    processed_image = cv2_to_base64(drawn_image)
                    
                    # Log detections
                    if detections:
                        log_manager.add_batch_logs(detections)
                    
                    # Send response
                    await websocket.send_json({
                        "detections": detections,
                        "sorting_decision": sorting,
                        "processed_image": processed_image,
                        "statistics": log_manager.get_statistics()
                    })
                    
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# ============== SYSTEM STATUS ENDPOINTS ==============

@app.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system health status including GPU, RAM, and model status"""
    status = system_monitor.get_full_status()
    return SystemStatus(**status)


@app.get("/system/classes")
async def get_available_classes():
    """Get list of available waste classes"""
    return {
        "classes": list(CLASS_NAMES.values()),
        "classes_vi": CLASS_NAMES_VI,
        "organic": ["banana_peel", "eggshell", "leaves"],
        "inorganic": ["bag", "bottle", "can"]
    }


# ============== LOGGING ENDPOINTS ==============

@app.get("/logs", response_model=LogsResponse)
async def get_logs(
    limit: Optional[int] = Query(default=100, ge=1, le=1000),
    class_filter: Optional[str] = Query(default=None),
    category_filter: Optional[str] = Query(default=None)
):
    """Get detection logs with optional filtering"""
    class_list = class_filter.split(",") if class_filter else None
    logs = log_manager.get_logs(
        limit=limit,
        class_filter=class_list,
        category_filter=category_filter
    )
    
    return LogsResponse(
        logs=logs,
        total_count=len(log_manager.logs)
    )


@app.get("/logs/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Get session statistics"""
    stats = log_manager.get_statistics()
    return StatisticsResponse(**stats)


@app.get("/logs/export/csv")
async def export_logs_csv():
    """Export logs to CSV file"""
    filepath = log_manager.export_to_csv()
    return FileResponse(
        path=filepath,
        filename=os.path.basename(filepath),
        media_type="text/csv"
    )


@app.get("/logs/export/excel")
async def export_logs_excel():
    """Export logs to Excel file"""
    filepath = log_manager.export_to_excel()
    return FileResponse(
        path=filepath,
        filename=os.path.basename(filepath),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.delete("/logs/clear")
async def clear_logs():
    """Clear all logs and start new session"""
    log_manager.clear_logs()
    return {"success": True, "message": "Logs cleared", "new_session_id": log_manager.session_id}


# ============== SNAPSHOT ENDPOINTS ==============

@app.post("/snapshot")
async def save_snapshot(
    frame_data: str,
    include_detections: bool = Query(default=True)
):
    """
    Save current frame as snapshot
    
    - **frame_data**: Base64 encoded image
    - **include_detections**: Whether to include bounding boxes
    """
    try:
        image = base64_to_cv2(frame_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        if include_detections:
            # Get current detections and draw
            detections = ai_core.predict(image)
            image = ai_core.draw_detections(image, detections)
        
        # Save snapshot
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = SNAPSHOT_DIR / filename
        cv2.imwrite(str(filepath), image)
        
        return {
            "success": True,
            "filename": filename,
            "path": str(filepath)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshots")
async def list_snapshots():
    """List all saved snapshots"""
    snapshots = []
    for f in SNAPSHOT_DIR.glob("*.jpg"):
        snapshots.append({
            "filename": f.name,
            "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "size_kb": round(f.stat().st_size / 1024, 2)
        })
    return {"snapshots": sorted(snapshots, key=lambda x: x["created"], reverse=True)}


@app.get("/snapshots/{filename}")
async def get_snapshot(filename: str):
    """Download a specific snapshot"""
    filepath = SNAPSHOT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path=str(filepath), filename=filename)


# ============== HEALTH CHECK ==============

@app.get("/")
async def root():
    """API root - health check"""
    return {
        "name": "EcoSort AI API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": ai_core.is_loaded,
        "endpoints": {
            "predict_image": "POST /predict/image",
            "predict_video": "POST /predict/video",
            "predict_frame": "POST /predict/frame",
            "websocket_stream": "WS /ws/stream",
            "system_status": "GET /system/status",
            "logs": "GET /logs",
            "statistics": "GET /logs/statistics",
            "export_csv": "GET /logs/export/csv",
            "export_excel": "GET /logs/export/excel"
        }
    }
