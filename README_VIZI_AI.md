# README_VIZI_AI

Tai lieu nay giai thich cac chinh sua vua duoc them cho he thong `vizi_ai` (multi-modal, multi-stock) va huong dan cach su dung thuc te.

## 1) Da chinh sua gi?

### 1.1 Them module moi `backend/vizi_ai/`

- `backend/vizi_ai/data_pipeline.py`
- `backend/vizi_ai/model.py`
- `backend/vizi_ai/trainer.py`
- `backend/vizi_ai/orchestrator.py`
- `backend/vizi_ai/__init__.py`

### 1.2 Them test smoke test

- `tests/test_vizi_ai.py`

### 1.3 Tich hop vao launcher chinh

- `run.py` duoc them cac mode moi:
  - `vizi-train`
  - `vizi-test`
  - `vizi-profile`
  - `vizi-evaluate`

## 2) Muc tieu cua he thong moi

`vizi_ai` duoc thiet ke de du doan thi truong theo kieu multi-modal:

- Gia + chi bao ky thuat (price sequence)
- Macro (FRED)
- Cross-market (indices, bonds, volatility, commodities, fx, crypto)
- Fundamentals
- News text (hash-token IDs)

Model hoc dong thoi nhieu bai toan (multi-task):

- `direction` (up/down)
- `ret_1d`, `ret_5d`, `ret_21d`
- `regime` (bear/sideways/bull)
- `confidence`

## 3) Chi tiet theo tung file

### 3.1 `backend/vizi_ai/data_pipeline.py`

Chuc nang:

- Streaming data theo `IterableDataset` de giam RAM
- Interleave nhieu ma co phieu trong cung mot stream
- Canh thoi gian (time alignment) giua cac modality
- Chia tap theo ngay (`train_end`, `val_end`) de tranh look-ahead

Dinh dang input da fix cho dung voi du lieu thuc te:

- Stocks: `data/stocks/{SYMBOL}/features.csv` hoac `ohlcv.csv`
- Macro: `data/macro/fred/*.csv` (cot `date,value`)
- Markets: dung ten file thuc te nhu `SP500`, `VIX`, `BTC`, `DXY`, ...
- Fundamentals: doc dung cac file `income_annual.csv`, `balance_sheet_annual.csv`, ...
- News: `data/processed/news_corpus.jsonl`

Shapes mau (da test pass):

- `price_seq`: `(B, 120, 57)`
- `macro_seq`: `(B, 60, 15)`
- `market_seq`: `(B, 60, 10)`
- `fundamental`: `(B, 20)`
- `news_ids`: `(B, 8, 64)`

### 3.2 `backend/vizi_ai/model.py`

Model: `ViziMarketTransformer`

Cau truc:

1. Encoder rieng cho tung modality
2. Fusion transformer (CLS token + modality-type embeddings)
3. Nhieu output heads cho multi-task

Thanh phan chinh:

- `PriceEncoder` (causal attention + RoPE)
- `MacroEncoder`
- `MarketEncoder`
- `FundamentalEncoder`
- `NewsEncoder`
- `CrossModalFusion`
- `PredictionHeads`

### 3.3 `backend/vizi_ai/trainer.py`

Chuc nang:

- Huan luyen voi AMP
- Gradient accumulation
- Uncertainty-weighted multi-task loss
- Early stopping
- Checkpointing + log lich su train/val
- Danh gia aggregate + per-stock (accuracy, MSE, MAE, Sharpe, IC)

Thu muc output model:

- `models/{run_name}_{timestamp}/`
- Co `config.json`, `train_history.json`, `val_history.json`, `eval_test.json`
- Co `best/`, `final/`, va cac `step_XXXXXX/`

### 3.4 `backend/vizi_ai/orchestrator.py`

CLI tong hop cho vizi-ai:

- `run`
- `train`
- `evaluate`
- `test`
- `profile`
- `collect`

## 4) Cach su dung nhanh (khuyen dung)

Tat ca lenh ben duoi chay tai thu muc goc project.

### 4.1 Smoke test nhanh

```powershell
.venv\Scripts\python run.py --mode vizi-test
```

Lenh nay se:

- Lay 5 symbols dau tien
- Tao 1 batch du lieu
- Chay forward pass model
- In shapes va sample outputs

### 4.2 Profile kich thuoc model + VRAM estimate

```powershell
.venv\Scripts\python run.py --mode vizi-profile
```

### 4.3 Train multi-modal model

```powershell
.venv\Scripts\python run.py --mode vizi-train --run-name vizi-o1 --vizi-epochs 30 --batch-size 64
```

Neu muon warm-up specialist truoc:

```powershell
.venv\Scripts\python run.py --mode vizi-train --run-name vizi-o1 --vizi-epochs 30 --specialist-warmup
```

### 4.4 Evaluate mot run da train

```powershell
.venv\Scripts\python run.py --mode vizi-evaluate --run-dir models\vizi-o1_YYYYMMDD_HHMMSS --split test
```

## 5) Cach su dung truc tiep orchestrator

Neu khong qua `run.py`, co the goi truc tiep:

```powershell
.venv\Scripts\python -m backend.vizi_ai.orchestrator test
.venv\Scripts\python -m backend.vizi_ai.orchestrator profile --batch-size 64
.venv\Scripts\python -m backend.vizi_ai.orchestrator train --run-name vizi-o1 --epochs 30 --batch-size 64
.venv\Scripts\python -m backend.vizi_ai.orchestrator evaluate --run-dir models\vizi-o1_YYYYMMDD_HHMMSS --split test
```

## 6) Kiem tra nhanh tinh dung sau khi setup

```powershell
.venv\Scripts\python tests\test_vizi_ai.py
```

Ky vong:

- In duoc shapes modalities
- In duoc prediction heads
- Tinh duoc task losses + total weighted loss
- Ket thuc voi dong `=== ALL TESTS PASSED ===`

## 7) Luu y quan trong

- `run.py` co 2 nhom epoch args:
  - `--epoch/--epochs` cho pipeline cu (`train` cua `backend.cli`)
  - `--vizi-epochs` cho `vizi-train`
- Du lieu phai co san trong `data/` theo cau truc hien tai
- Windows nen de `num_workers=0` trong dataloader streaming (da default)

## 8) Luong van hanh de xuat

1. Chay `vizi-test`
2. Chay `vizi-profile`
3. Chay `vizi-train`
4. Chay `vizi-evaluate`
5. So sanh `eval_test.json` giua cac run names (`vizi-o1`, `vizi-1`, `vizi-b1`)

---

Neu ban muon, buoc tiep theo minh co the bo sung them mot phan "Training recipes" trong file nay (preset cho RTX 4090, preset cho B200, va preset nhanh de debug).