# ViziGenesis — RunPod GPU Training Setup

## 📋 Bước 1: Setup môi trường trên RunPod

### A. SSH vào RunPod pod sau khi khởi động

```bash
# Sẽ có URL kết nối SSH từ RunPod dashboard
ssh root@<your-pod-ip>
```

### B. Clone repo và cài dependencies

```bash
# Clone project
git clone <repo-url> vizigenesis
cd vizigenesis

# Cài Python 3.10 nếu chưa có (thường sẵn)
# Check Python version
python3 --version

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Cài dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Bước 2: Setup data (tùy chọn)

### Tải dữ liệu lịch sử từ Yahoo Finance (tự động)

Code sẽ tự động kéo dữ liệu từ Yahoo Finance khi training, không cần setup riêng.

### Hoặc tải dữ liệu Local nếu có

```bash
# Copy dữ liệu local vào data/
# Ví dụ: data/AAPL/local.csv, data/MSFT/local.csv
scp -r local_data/* root@<pod-ip>:/root/vizigenesis/data/
```

---

## 🚀 Bước 3: Chạy Training

### Mode SIMPLE (nhanh, ~5-15 phút)

```bash
python run.py --mode train --symbol AAPL --ai-mode simple --epoch 80
```

**Tham số:**
- `--symbol`: Stock symbol (AAPL, MSFT, NVDA, TSLA, VIC.VN, VNM.VN, HPG.VN...)
- `--epoch`: Số epochs (80 cho simple)
- `--period`: Dữ liệu lịch sử (mặc định 2y, có thể dùng 1y/5y/10y)

**Kết quả:**
```
models/AAPL/
  ├── model.pt
  ├── scaler.pkl
  ├── meta.json
  └── metrics.json
```

### Mode PRO (lâu hơn, ~30-60 phút trên GPU)

```bash
python run.py --mode train --symbol AAPL --ai-mode pro --epoch 220 --period 10y
```

**Tham số bổ sung:**
- `--period 10y`: Dùng 10 năm dữ liệu (chi tiết hơn)
- `--epoch 220`: Nhiều epochs hơn để train kỹ

**Kết quả:**
```
models/AAPL/
  ├── model_pro.pt
  ├── scaler_pro.pkl
  ├── meta_pro.json
  └── metrics_pro.json
```

---

## 📊 Bước 4: Đánh giá Model (Optional)

### Check accuracy trên train/val/test

```bash
python run.py --mode check-accuracy --symbol AAPL --ai-mode pro --period 10y
```

### Evaluate một split cụ thể

```bash
python run.py --mode evaluate --symbol AAPL --ai-mode pro --split test
```

### Validate model

```bash
python run.py --mode validate --symbol AAPL --ai-mode pro
```

---

## 🎯 Các ví dụ lệnh Training phổ biến

### Train nhiều symbols cùng lúc (dùng looping)

```bash
#!/bin/bash
for symbol in AAPL MSFT NVDA TSLA; do
  echo "Training $symbol..."
  python run.py --mode train --symbol $symbol --ai-mode pro --epoch 220 --period 10y
done
```

Lưu vào `train_batch.sh`, rồi chạy:

```bash
chmod +x train_batch.sh
./train_batch.sh
```

### Train với custom period

```bash
# 1 năm (nhanh)
python run.py --mode train --symbol AAPL --ai-mode simple --epoch 50 --period 1y

# 5 năm (vừa)
python run.py --mode train --symbol AAPL --ai-mode pro --epoch 150 --period 5y

# 10 năm (chi tiết)
python run.py --mode train --symbol AAPL --ai-mode pro --epoch 220 --period 10y
```

---

## 💾 Bước 5: Download Models về máy local

```bash
# Từ máy local (không ssh):
scp -r root@<pod-ip>:/root/vizigenesis/models ~/Downloads/vizigenesis_models/
```

---

## 🔧 Tips cho RunPod

### 1. Kiểm tra GPU đang dùng

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name()}')"
```

### 2. Monitor GPU memory khi training

```bash
# Terminal 1: Training
python run.py --mode train --symbol AAPL --ai-mode pro --epoch 220

# Terminal 2: Monitor
watch -n 1 nvidia-smi
```

### 3. Save network bandwidth

Nếu dữ liệu Yahoo Finance chậm, tạo `.env`:

```bash
cat > .env << EOF
CACHE_TTL_HISTORY=604800
EOF
```

---

## ⚠️ Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'tvDatafeed'"

Kiểm tra `requirements.txt` - package tên là `tradingview_ta` hoặc `tradingview-datafeed`:

```bash
pip install tradingview-datafeed==2.1.1
```

### Lỗi: "CUDA out of memory"

Giảm batch size hoặc epochs:

```bash
python run.py --mode train --symbol AAPL --ai-mode simple --epoch 50
```

### Lỗi: "Yahoo Finance timeout"

Dữ liệu Yahoo đôi khi slow, có fallback:

```bash
python run.py --mode train --symbol AAPL --ai-mode simple --epoch 50 --period 2y
```

---

## 📝 Environment Variables (Optional)

Tạo `.env` nếu cần:

```bash
cat > .env << EOF
# Macro data source: World Bank Data360 (no API key required)

# Cache settings
CACHE_TTL_HISTORY=604800
CACHE_TTL_REALTIME=300

# Training settings
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
EOF
```

---

## ✅ Checklist

- [ ] SSH vào RunPod
- [ ] Clone repo + pip install
- [ ] Chạy test script: `python run.py --mode train --symbol AAPL --ai-mode simple --epoch 10`
- [ ] Nếu ổn, chạy lệnh training thực
- [ ] Download models về local
- [ ] Stop pod (tiết kiệm $)

---

Chúc mừng! 🎉
