# 🌿 EcoSort AI - Waste Classification System

Hệ thống phân loại rác thông minh sử dụng YOLOv8s và AI, hỗ trợ 39 loại rác với độ chính xác cao.

## 📁 Cấu trúc Project

```
projectFinal/
├── Backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── ai_core.py       # YOLOv8s model & inference
│   │   ├── config.py        # 39 waste classes configuration
│   │   ├── image_processing.py
│   │   ├── log_manager.py   # Activity logging
│   │   ├── schemas.py       # Pydantic models
│   │   └── system_monitor.py
│   ├── requirements.txt
│   └── run.py
├── Frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
├── notebook/
│   ├── train_yolo11n.ipynb              # Training notebook (Kaggle)
│   ├── evaluate_model_yolov8n_yolov8s_yolo11n.ipynb  # Evaluation
│   └── evaluate-compare-yolov8s-yolo11n-yolov8n.ipynb
├── runs/
│   ├── yolov8n_waste/
│   │   ├── weights/best.pt      # YOLOv8n model (3.2M params)
│   │   ├── results.csv
│   │   └── args.yaml
│   ├── yolov8s_waste/
│   │   ├── weights/best.pt      # YOLOv8s model (11.2M params) ⭐ SELECTED
│   │   ├── results.csv
│   │   └── args.yaml
│   └── yolo11n_waste_optimized/
│       ├── weights/best.pt      # YOLO11n model (2.6M params)
│       ├── results.csv
│       └── args.yaml
├── data_kaggle.yaml
├── yolov8n.pt                   # Pretrained weights
└── yolo11n.pt                   # Pretrained weights
```

## 🚀 Hướng dẫn Cài đặt

### 1. Cài đặt Backend

```powershell
# Di chuyển vào thư mục Backend
cd Backend

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
.\venv\Scripts\Activate.ps1

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy Backend Server

```powershell
# Chạy server
python run.py
```

Server sẽ chạy tại: `http://localhost:8000`

### 3. Chạy Frontend

Mở file `Frontend/index.html` trong trình duyệt hoặc sử dụng Live Server extension trong VS Code.

## 📌 Tính năng

### A. Frontend (Giao diện người dùng)

#### 1. Module Nhận diện & Xử lý
- ✅ Upload ảnh (JPG, PNG, JPEG)
- ✅ Upload video (MP4, AVI)
- ✅ Camera trực tiếp (Live Stream)
- ✅ Vẽ bounding box với mã màu:
  - � Cam: Organic (32 classes - thức ăn thừa, vỏ trái cây, xương cá...)
  - ⚫ Xám: Inorganic (2 classes - túi nylon, xốp...)
  - 🟢 Xanh lá: Recyclable (5 classes - giấy, nhựa, kim loại...)
- ✅ Hiển thị nhãn và độ tin cậy (confidence threshold: 0.5)

#### 2. Module Điều khiển
- ✅ Thanh trượt Ngưỡng tin cậy (0.0 - 1.0)
- ✅ Bộ lọc hiển thị theo class
- ✅ Nút chụp màn hình (Snapshot)

#### 3. Module Thống kê
- ✅ Bộ đếm thời gian thực
- ✅ Biểu đồ tỷ lệ (Bar/Pie Chart)
- ✅ Nhật ký hoạt động với tìm kiếm

#### 4. Module Ra quyết định
- ✅ Đèn báo tín hiệu (Xanh/Đỏ)
- ✅ Logic phân luồng tự động

### B. Backend (API & Logic)

####Model: **YOLOv8s** (11.2M params, 68 epochs trained)
- ✅ Load model thông minh (YOLOv8s best.pt)
- ✅ Inference với confidence=0.5, IoU=0.45, imgsz=640
- ✅ Non-max Suppression (NMS)
- ✅ Hỗ trợ 39 classes với độ chính xác cao
- ✅ Non-max Suppression (NMS) (39 classes)
- ✅ Quy tắc phân nhóm: Organic (0-31), Inorganic (32-33), Recyclable (34-38)
- ✅ Confidence threshold: 0.5 (50%)
- ✅ IoU threshold: 0.45 cho NMS
#### 2. Business Logic
- ✅ Mapping Class ID → Tên loại rác
- ✅ Quy tắc phân nhóm Vô cơ/Hữu cơ

#### 3. API Endpoints
- `POST /predict/image` - Phân loại ảnh
- `POST /predict/video` - Phân loại video
- `POST /predict/frame` - Phân loại frame (realtime)
- `WS /ws/stream` - WebSocket streaming
- `GET /system/status` - Trạng thái hệ thống
- `GET /logs` - Lấy logs
- `GET /logs/export/csv` - Xuất CSV
- `GET /logs/export/excel` - Xuất Excel

#### 4. Image Processing
- ✅ Resize & Normalize (640x640)
- ✅ BGR ↔ RGB conversion

## 🏷️ Classes (39 loại rác)

### Phân loại theo nhóm:

| Nhóm | Số lượng | Class IDs |
|------|----------|-----------|
| 🟠 **Organic (Hữu cơ)** | 32 classes | 0-31 |
| ⚫ **Inorganic (Vô cơ)** | 2 classes | 32-33 |
| 🟢 **Recyclable (Tái chế)** | 5 classes | 34-38 |

### Chi tiết Classes:

**Organic Waste (0-31):**
- Apple, Apple-core, Apple-peel, Bone, Bone-fish
- Bread, Bun, Egg-hard, Egg-scramble, Egg-shell
- Egg-steam, Egg-yolk, Fish, Meat, Mussel
- Mussel-shell, Noodle, Orange, Orange-peel, Other-waste
- Pancake, Pasta, Pear, Pear-core, Pear-peel
- Potato, Rice, Shrimp, Shrimp-shell, Tofu, Tomato, Vegetable

**Inorganic (32-33):**
- plastic_bag, styrofoam
🧠 Model Training & Evaluation

### Models Trained (Kaggle P100 GPU):

| Model | Params | Epochs | Training Time | mAP@0.5 | Status |
|-------|--------|--------|---------------|---------|--------|
| **YOLOv8s** | 11.2M | 68 | ~5.2h | **Highest** | ⭐ **Selected** |
| YOLOv8n | 3.2M | 100 | ~4.7h | Good | Tested |
| YOLO11n | 2.6M | 100 | ~3.8h | Good | Tested |

### Why YOLOv8s?
1. ✅ **Highest accuracy** - Best mAP@0.5 among 3 models
2. ✅ **Balanced performance** - Optimal speed/accuracy trade-off
3. ✅ **Production-ready** - Suitable for cloud/server deployment
4. ✅ **Best generalization** - Stable performance on test set
5. ✅ **Recommended** for Production API Backend

### Model Specifications:
- **Architecture**: YOLOv8s (Small)
- **Input size**: 640×640
- **Batch size**: 16
- **Optimizer**: SGD (default YOLO)
- **Classes**: 39 waste types
- **Hardware**: Kaggle P100 GPU (16GB VRAM)

### Evaluation Notebooks:
- `notebook/train_yolo11n.ipynb` - Training on Kaggle
- `notebook/evaluate_model_yolov8n_yolov8s_yolo11n.ipynb` - Comprehensive evaluation
- `notebook/evaluate-compare-yolov8s-yolo11n-yolov8n.ipynb` - Model comparison

## 🔧 API Documentation

Sau khi chạy server, truy cập:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🎯 Model Performance

**YOLOv8s Production Metrics:**
- Confidence Threshold: 0.5
- IoU Threshold: 0.45
- Image Size: 640×640
- Expected Latency: ~[X]ms per image
- Expected Throughput: ~[Y] images/second

## 📊 Dataset Information

- **Total Classes**: 39
- **Dataset Split**: Train/Valid/Test
- **Image Format**: JPG, PNG
- **Annotation Format**: YOLO format
- **Configuration**: `data_kaggle.yaml`

## 🚀 Deployment

### Production Recommendations:
- **Cloud/Server API**: YOLOv8s (best accuracy)
- **Edge Devices**: YOLOv8n or YOLO11n (smaller, faster)
- **Mobile Apps**: YOLO11n (most compact)

### Current Setup:
- Backend uses **YOLOv8s** for optimal accuracy
- Model path: `runs/yolov8s_waste/weights/best.pt`
- FastAPI server on port 8000

Sau khi chạy server, truy cập:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📝 License

MIT License
