# 🌿 EcoSort AI - Waste Classification System

Hệ thống phân loại rác thông minh sử dụng YOLOv8 và AI.

## 📁 Cấu trúc Project

```
projectFinal/
├── Backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── ai_core.py       # YOLO model & inference
│   │   ├── config.py        # Configuration settings
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
├── runs/
│   └── exp3_final_p100/
│       └── weights/
│           └── best.pt      # Trained model
└── data_kaggle.yaml
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
  - 🟢 Xanh lá: Vô cơ (Chai, Lon, Túi)
  - 🟠 Cam: Hữu cơ (Vỏ chuối, Lá, Vỏ trứng)
- ✅ Hiển thị nhãn và độ tin cậy

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

#### 1. AI Core
- ✅ Load model thông minh (best.pt hoặc fallback)
- ✅ YOLOv8 inference engine
- ✅ Non-max Suppression (NMS)

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

## 🏷️ Classes (6 loại rác)

| ID | Tên | Tiếng Việt | Nhóm |
|----|-----|------------|------|
| 0 | bag | Túi | Vô cơ |
| 1 | banana_peel | Vỏ chuối | Hữu cơ |
| 2 | bottle | Chai | Vô cơ |
| 3 | can | Lon | Vô cơ |
| 4 | eggshell | Vỏ trứng | Hữu cơ |
| 5 | leaves | Lá cây | Hữu cơ |

## 🔧 API Documentation

Sau khi chạy server, truy cập:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📝 License

MIT License
