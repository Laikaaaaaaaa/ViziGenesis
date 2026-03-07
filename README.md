# ViziGenesis ⚡ Nền tảng dự đoán cổ phiếu bằng AI

Ứng dụng full-stack kết hợp **FastAPI**, **PyTorch LSTM** và frontend tương tác để:
- Lấy dữ liệu trực tiếp từ **Yahoo Finance**
- Train model trên **GPU (nếu có)**
- Dự đoán xu hướng ngắn hạn và dài hạn
- Đánh giá chất lượng mô hình bằng loss/accuracy/trend quality

---

## ✨ Tính năng chính

| Tính năng | Mô tả |
|---|---|
| Giá thời gian thực | Lấy giá stock real-time từ Yahoo Finance, có cache TTL |
| Biểu đồ nến | Candlestick + volume (TradingView embed + fallback) |
| Dự đoán AI | 2 chế độ `simple` và `pro` (next-day + dự báo 30 ngày) |
| Chỉ báo kỹ thuật | MA20, MA50, EMA20, RSI, MACD, Bollinger Band, OBV |
| Huấn luyện GPU | Tự động dùng CUDA nếu khả dụng, fallback CPU |
| Train realtime | WebSocket stream loss theo epoch |
| Đánh giá mô hình | MSE, RMSE, MAE, MAPE + trend accuracy + macro F1 |
| Artifact | Lưu model `.pt`, scaler `.pkl`, metadata/metrics `.json` và đóng gói `.zip` |

---

## 📁 Cấu trúc thư mục

```text
ViziGenesis/
├── backend/
│   ├── app.py
│   ├── cli.py          # CLI train/evaluate/validate/test/check-accuracy/package
│   ├── config.py
│   ├── data_utils.py   # Yahoo Finance + cache + preprocess
│   ├── model.py        # LSTM train/predict/save/load
│   └── pipeline.py     # split train/val/test + metrics đánh giá
├── frontend/
│   ├── home.html
│   ├── predict.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── models/             # Artifact sinh ra sau khi train
├── data/
├── requirements.txt
└── run.py
```

---

## 🚀 Chạy nhanh

### 1) Tạo môi trường ảo và cài thư viện

Cần cài Python 3.10.11 trên máy trước khi chạy các lệnh dưới đây.

**Windows (PowerShell):**

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Chạy web app

```bash
py -3.10 run.py
```

Hoặc explicit mode:

```bash
py -3.10 run.py --mode serve --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt: **http://localhost:8000**

---

## 🧠 Train/Evaluate/Validate/Test bằng Yahoo Finance

Toàn bộ lệnh dưới đây đều kéo dữ liệu từ Yahoo Finance theo `--symbol` + `--ai-mode`.

- `simple`: nhanh hơn, mặc định period `2y`, ít feature hơn.
- `pro`: train lâu hơn, mặc định period `10y`, có thêm chỉ báo kỹ thuật.

### Dùng 1 lệnh thống nhất `run.py`

```bash
# Train simple
python run.py --mode train --symbol AAPL --ai-mode simple --epoch 80

# Train professional
python run.py --mode train --symbol AAPL --ai-mode pro --epoch 220

# Evaluate
python run.py --mode evaluate --symbol AAPL --ai-mode pro --split test

# Validate
python run.py --mode validate --symbol AAPL --ai-mode pro

# Test
python run.py --mode test --symbol AAPL --ai-mode pro

# Check full quality metrics
python run.py --mode check-accuracy --symbol AAPL --ai-mode pro

# Package artifact
python run.py --mode package --symbol AAPL --ai-mode pro
```

### Train model

```bash
python -m backend.cli train --symbol AAPL --ai-mode pro --period 10y --epochs 220
```

Kết quả tạo artifact (mode-aware):
- `models/AAPL/model.pt`, `scaler.pkl`, `meta.json`, `metrics.json` (simple)
- `models/AAPL/model_pro.pt`, `scaler_pro.pkl`, `meta_pro.json`, `metrics_pro.json` (pro)

### Evaluate theo split

```bash
python -m backend.cli evaluate --symbol AAPL --ai-mode pro --period 10y --split test
```

### Validate

```bash
python -m backend.cli validate --symbol AAPL --ai-mode pro --period 10y
```

### Test

```bash
python -m backend.cli test --symbol AAPL --ai-mode pro --period 10y
```

### Check accuracy/loss/trend quality (train + val + test)

```bash
python -m backend.cli check-accuracy --symbol AAPL --ai-mode pro --period 10y
```

Các metric gồm:
- **Loss/Regression**: MSE, RMSE, MAE, MAPE
- **Trend quality**: Accuracy (%), Macro Precision, Macro Recall, Macro F1

### Package artifact

```bash
python -m backend.cli package --symbol AAPL --ai-mode pro
```

Sinh file:
- `models/AAPL/AAPL_artifact.zip` (simple)
- `models/AAPL/AAPL_artifact_pro.zip` (pro)

> Ghi chú: Dự án hiện dùng PyTorch nên model chính là `.pt` (không xuất `.h5` mặc định).

---

## 🌐 API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/price/{symbol}` | Giá real-time |
| GET | `/api/history/{symbol}?period=1y` | Dữ liệu OHLCV |
| POST | `/api/train/{symbol}?mode=simple|pro&period=2y|10y&epochs=...` | Train/retrain model theo profile |
| GET | `/api/predict/{symbol}?mode=simple|pro&auto_train=true` | Dự đoán next-day + long-term |
| GET | `/api/model-status/{symbol}?mode=simple|pro` | Kiểm tra model theo profile |
| GET | `/api/download-csv/{symbol}?mode=simple|pro` | Tải CSV dự đoán theo profile |
| WS | `/ws/price/{symbol}` | Stream giá live mỗi 60 giây |
| WS | `/ws/train/{symbol}` | Stream tiến trình training |

---

## 🏗️ Blueprint triển khai mới: mỗi cổ phiếu = 1 model riêng

Phần này là hướng triển khai backend + AI + frontend + training/deploy theo đúng yêu cầu “mỗi mã cổ phiếu một model riêng”, có versioning và update an toàn dữ liệu cũ.

### 1) Thiết kế lưu trữ model theo symbol + version

Mục tiêu:
- Mỗi symbol có model độc lập (`AAPL`, `TSLA`, `MSFT`, ...)
- Mỗi lần train sinh 1 version mới
- Có alias `latest` để backend load nhanh

Gợi ý cấu trúc thư mục:

```text
models/
└── AAPL/
    ├── model.pt             # active model
    ├── scaler.pkl           # active scaler
    ├── metrics.json         # active metrics
    ├── meta.json            # active metadata
    ├── latest.json          # trỏ version active mới nhất
    └── versions/
        ├── v20260305_150000/
        │   ├── model.pt
        │   ├── scaler.pkl
        │   ├── metrics.json
        │   └── meta.json
        └── v20260310_090000/
            └── ...
```

`meta.json` nên có:
- `symbol`, `version`, `created_at`
- `data_range` (from/to), `num_rows`
- `features`, `hyperparams`, `device`
- `metrics` (val/test)

### 2) Backend load model theo symbol người dùng nhập

Flow runtime:
1. User gọi `/api/predict/AAPL`
2. Backend đọc `models/AAPL/latest.json`
3. Load đúng `model.pt` + `scaler.pkl`
4. Nếu cache model còn nóng thì dùng cache RAM (LRU)

Ví dụ FastAPI loader (minh hoạ):

```python
from functools import lru_cache
import json, os
from backend.model import load_trained_model, load_scaler

BASE_MODEL_DIR = "models"

def get_latest_version(symbol: str) -> str:
    latest_file = os.path.join(BASE_MODEL_DIR, symbol.upper(), "latest.json")
    with open(latest_file, "r", encoding="utf-8") as f:
	  return json.load(f)["version"]

@lru_cache(maxsize=64)
def load_symbol_bundle(symbol: str):
    sym = symbol.upper()
    version = get_latest_version(sym)
    model_path_symbol = f"{sym}_{version}"  # hoặc custom path riêng

    model = load_trained_model(model_path_symbol)
    scaler = load_scaler(model_path_symbol)
    if model is None or scaler is None:
	  raise FileNotFoundError(f"Model not found for {sym} v{version}")
    return model, scaler, version
```

### 3) Cơ chế update model khi có dữ liệu mới (không ảnh hưởng dữ liệu cũ)

Chiến lược an toàn:
1. Không overwrite version cũ
2. Train version mới trong thư mục tạm (`staging`)
3. Đánh giá đạt ngưỡng mới promote thành `latest`
4. Fail rollback: giữ `latest` cũ

Quy tắc promote ví dụ:
- `new_val_rmse <= old_val_rmse * 1.02`
- `new_trend_f1 >= old_trend_f1 - 0.01`

Pseudo-workflow:

```text
fetch new data -> train v_new -> evaluate -> compare threshold
  -> pass: update latest.json -> done
  -> fail: keep old latest -> archive v_new
```

### 4) Tối ưu GPU khi train nhiều model

Nguyên tắc:
- Ưu tiên queue theo batch (không train mọi symbol cùng lúc trên 1 GPU)
- Dùng scheduler: 1-2 job GPU song song tùy VRAM
- Mixed precision (`torch.cuda.amp`) để giảm VRAM
- Early stopping + checkpoint tốt nhất
- Warm-start từ version gần nhất của cùng symbol

Khuyến nghị vận hành:
- Nhóm symbol theo độ ưu tiên thanh khoản
- Train incremental mỗi ngày cho top symbols, weekly cho tail symbols
- Cache dữ liệu OHLCV và chỉ append phần mới

### 5) Frontend realtime vẫn giữ đầy đủ

Frontend không đổi UX chính:
- Realtime price qua WebSocket `/ws/price/{symbol}`
- Candlestick từ `/api/history/{symbol}`
- Trend prediction từ `/api/predict/{symbol}`

Điểm cần thêm:
- Hiển thị `model_version` và `model_updated_at`
- Badge trạng thái model: `latest` / `stale` / `training`
- Biểu đồ nến dùng Lightweight Charts (style sàn chứng khoán, crosshair, volume, MA20/MA50)

---

## 🔁 Visual workflow end-to-end

```text
[User nhập symbol] 
	|
	v
[FastAPI /api/predict/{symbol}] 
	|
	v
[Model Registry đọc latest version theo symbol]
	|
	v
[Load model+scaler đúng symbol/version]
	|
	v
[Inference + trend quality score]
	|
	v
[Frontend hiển thị realtime + chart + trend + model version]
```

Training/Deploy workflow:

```text
[Scheduler/Cron] -> [Fetch Yahoo new candles] -> [Train v_new per symbol]
	-> [Evaluate val/test + trend quality]
	-> [Promote/Reject]
	-> [Update latest.json]
	-> [API dùng bản mới ngay]
```

---

## 🖥️ GPU (tuỳ chọn)

Nếu máy có CUDA, app tự động dùng GPU.

Ví dụ cài PyTorch CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra nhanh:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## 📝 Lưu ý

- Python yêu cầu: **3.9+**
- Dữ liệu giá cổ phiếu đến từ Yahoo Finance (free)
- Nếu Yahoo Finance bị chặn mạng/DNS, CLI sẽ tự fallback sang file sample trong `data/` để vẫn train/evaluate được.
- Kết quả model mang tính tham khảo, không phải tư vấn đầu tư
