
# 🌿 EcoSort AI - Waste Classification Model Report

## 📊 Evaluation Summary
- **Dataset**: Waste_organic_inorganic_recyclable
- **Classes**: 40 (Organic: 33, Inorganic: 2, Recyclable: 5)
- **Models Evaluated**: YOLOv8n, YOLO11n
- **Image Size**: 640x640

## 🏆 Best Model: YOLOv8n

### Performance Metrics
| Metric | Value |
|--------|-------|
| mAP@50 | 0.8337 |
| mAP@50-95 | 0.6244 |
| Precision | 0.9008 |
| Recall | 0.8364 |
| F1 Score | 0.8674 |
| Inference Speed | 92.9 FPS |

### Model Comparison

| Model   |    mAP50 |   mAP50-95 |   Precision |   Recall |       F1 |   Inference Time (ms) |     FPS |   Overall Score |
|:--------|---------:|-----------:|------------:|---------:|---------:|----------------------:|--------:|----------------:|
| YOLOv8n | 0.833709 |   0.624389 |    0.900765 | 0.836366 | 0.867371 |               10.7679 | 92.8687 |        0.809967 |
| YOLO11n | 0.832968 |   0.611034 |    0.887183 | 0.839548 | 0.862708 |               12.8705 | 77.6968 |        0.754215 |

## 🎯 Recommended Use Cases

1. **Real-time Camera Detection**: FPS > 30 suitable for live streaming
2. **Image Upload**: High accuracy for single image analysis
3. **Video Processing**: Batch processing of video files
4. **Edge Deployment**: ONNX/TensorRT for embedded systems

## 📁 Output Files

- `best_model.pt` - PyTorch model (recommended)
- `best_model.onnx` - ONNX format for cross-platform
- `best_model.torchscript` - TorchScript for production
- `model_comparison.csv` - Detailed comparison results

## 🚀 Quick Start

```python
from ultralytics import YOLO

# Load model
model = YOLO('best_model.pt')

# Image inference
results = model.predict('image.jpg', conf=0.25)

# Video inference
results = model.predict('video.mp4', stream=True)

# Camera inference
results = model.predict(source=0, stream=True)  # webcam
```

---
Generated on: 2025-12-03 13:23:11
