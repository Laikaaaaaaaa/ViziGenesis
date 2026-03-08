"""
ViziGenesis — FastAPI backend
Endpoints: real-time quotes, historical data, training, predictions, CSV export.
WebSocket support for live chart updates & training progress.
"""
import asyncio, json, logging, os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import DEVICE, SEQUENCE_LENGTH, TARGET_COL, FRONTEND_DIR, QUANT_FEATURE_COLS, QUANT_SEQUENCE_LENGTH
from backend.data_utils import (
    get_realtime_price, get_historical_data_with_fallback, prepare_sequences,
    predictions_to_csv, save_sample_data, save_downloaded_history_csv,
    normalize_ai_mode, get_mode_config, get_feature_columns, prepare_feature_dataframe,
    get_symbol_news, search_symbol_catalog, get_realtime_quotes,
    get_training_sample_dates, compute_time_weights,
    prepare_quant_targets_from_raw, migrate_legacy_data_files,
)
from backend.model import (
    train_model, predict, classify_trend, compute_confidence,
    long_term_trend, save_scaler, load_scaler, save_meta, load_meta,
    load_trained_model, model_exists, get_latest_version_info,
    migrate_legacy_artifacts, get_ensemble_info,
)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")
logger = logging.getLogger("vizigenesis")


def _load_env_file() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ[key] = value
    except Exception:
        logger.warning("Failed to load .env file", exc_info=True)


_load_env_file()

# ── App init ──────────────────────────────────────────────────────────
app = FastAPI(title="ViziGenesis", version="1.0.0",
              description="AI-powered stock prediction platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")


# ═══════════════════════════════════════════════════════════════════════
# HTML page routes
# ═══════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the landing page."""
    return FileResponse(
        os.path.join(FRONTEND_DIR, "home.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/predict", response_class=HTMLResponse)
async def serve_predict():
    """Serve the prediction page."""
    return FileResponse(
        os.path.join(FRONTEND_DIR, "predict.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# REST endpoints
# ═══════════════════════════════════════════════════════════════════════

# ── Real-time price ──────────────────────────────────────────────────
@app.get("/api/price/{symbol}")
async def api_price(symbol: str):
    """Get cached real-time stock price."""
    try:
        return get_realtime_price(symbol)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Realtime quote unavailable for {symbol.upper()}: {e}")


@app.get("/api/quotes")
async def api_quotes(symbols: str = Query(default="")):
    """Get realtime quotes for multiple symbols in one request."""
    try:
        parsed = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        parsed = parsed[:80]
        if not parsed:
            return {"items": []}
        return {
            "items": get_realtime_quotes(parsed),
            "count": len(parsed),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Historical OHLCV (for candlestick chart) ────────────────────────
@app.get("/api/history/{symbol}")
async def api_history(symbol: str, period: str = "1y"):
    """Return OHLCV rows as JSON for charting."""
    try:
        df, source = get_historical_data_with_fallback(symbol, period=period)
        save_downloaded_history_csv(symbol, df, source)
        records = []
        for idx, row in df.iterrows():
            records.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        return {"symbol": symbol.upper(), "data": records, "source": source}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/news/{symbol}")
async def api_news(symbol: str, limit: int = Query(default=8, ge=1, le=20)):
    """Return recent related news for the selected symbol."""
    try:
        return {
            "symbol": symbol.upper(),
            "items": get_symbol_news(symbol, limit=limit),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/news/{symbol}")
async def news_alias(symbol: str, limit: int = Query(default=8, ge=1, le=20)):
    """Compatibility alias: return raw news list for a symbol."""
    try:
        return get_symbol_news(symbol, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/symbols/search")
async def api_symbol_search(q: str = Query(default="", min_length=1), limit: int = Query(default=12, ge=1, le=30)):
    """Autocomplete symbols from financedatabase catalog."""
    try:
        return {
            "query": q,
            "items": search_symbol_catalog(q, limit=limit),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Train model ──────────────────────────────────────────────────────
@app.post("/api/train/{symbol}")
async def api_train(
    symbol: str,
    mode: str = Query(default="simple"),
    period: Optional[str] = Query(default=None),
    epochs: Optional[int] = Query(default=None, ge=1, le=600),
):
    """
    Train (or retrain) LSTM model on 2-year daily data.
    Returns training history and device used.
    """
    try:
        ai_mode = normalize_ai_mode(mode)
        mode_cfg = get_mode_config(ai_mode)
        selected_period = period or mode_cfg["period"]
        selected_epochs = int(epochs or mode_cfg["default_epochs"])
        feature_cols = get_feature_columns(ai_mode)

        df_raw, source = get_historical_data_with_fallback(symbol, period=selected_period)
        if source in {"local_sample_csv", "generated_sample_csv"}:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Không thể huấn luyện {symbol.upper()} bằng dữ liệu fallback ({source}). "
                    "Nguồn market live không khả dụng, vui lòng thử lại khi có dữ liệu Yahoo/TradingView."
                ),
            )
        df = prepare_feature_dataframe(df_raw, ai_mode)

        csv_path = save_downloaded_history_csv(symbol, df, source)
        X_tr, y_tr, X_val, y_val, scaler = prepare_sequences(df, feature_cols=feature_cols)
        if len(X_tr) < 10:
            raise HTTPException(status_code=400,
                                detail="Không đủ dữ liệu để huấn luyện (cần > 60 phiên).")

        # Compute time-based sample weights (recent data → higher weight)
        train_dates = get_training_sample_dates(df, len(X_tr))
        time_weights = compute_time_weights(len(X_tr), train_dates)

        model, history = train_model(X_tr, y_tr, X_val, y_val,
                                     symbol=symbol.upper(), profile=ai_mode, epochs=selected_epochs,
                                     sample_weights=time_weights, enable_augmentation=True)
        save_scaler(scaler, symbol.upper(), profile=ai_mode)
        save_meta(symbol.upper(), {
            "mode": ai_mode,
            "training_period": selected_period,
            "epochs_run": len(history),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "device": str(DEVICE),
            "seq_len": SEQUENCE_LENGTH,
            "input_size": len(feature_cols),
            "features": feature_cols,
            "time_weighted": True,
            "augmentation": True,
            "macro_features": [c for c in feature_cols if c in
                               ("SP500", "NASDAQ", "VIX", "BOND_10Y", "INFLATION_PROXY")],
        }, profile=ai_mode)
        return {
            "status": "ok",
            "symbol": symbol.upper(),
            "mode": ai_mode,
            "device": str(DEVICE),
            "data_source": source,
            "period": selected_period,
            "data_csv": csv_path,
            "epochs_run": len(history),
            "features_count": len(feature_cols),
            "time_weighted": True,
            "augmentation": True,
            "history": history,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Predict next day + long-term ─────────────────────────────────────
@app.get("/api/predict/{symbol}")
async def api_predict(
    symbol: str,
    auto_train: bool = Query(default=False),
    mode: str = Query(default="simple"),
    period: Optional[str] = Query(default=None),
):
    """
    Predict next-day trend and long-term outlook using trained model.
    Auto-trains if no model exists yet.
    """
    sym = symbol.upper()
    ai_mode = normalize_ai_mode(mode)
    mode_cfg = get_mode_config(ai_mode)
    selected_period = period or mode_cfg["period"]
    feature_cols = get_feature_columns(ai_mode)
    try:
        df_raw, source = get_historical_data_with_fallback(sym, period=selected_period)
        if source in {"local_sample_csv", "generated_sample_csv"}:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Không thể dự đoán {sym} bằng dữ liệu fallback ({source}). "
                    "Vui lòng thử lại khi có dữ liệu Yahoo/TradingView để đảm bảo giá đúng thị trường."
                ),
            )
        df = prepare_feature_dataframe(df_raw, ai_mode)
        csv_path = save_downloaded_history_csv(sym, df_raw, source)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    scaler = load_scaler(sym, profile=ai_mode)
    model = load_trained_model(sym, profile=ai_mode)

    def _auto_train_now(reason: str):
        X_tr, y_tr, X_val, y_val, fitted_scaler = prepare_sequences(df, feature_cols=feature_cols)
        if len(X_tr) < 10:
            raise HTTPException(status_code=400, detail="Không đủ dữ liệu để dự đoán.")
        # Time-weighted + augmented training
        train_dates = get_training_sample_dates(df, len(X_tr))
        time_weights = compute_time_weights(len(X_tr), train_dates)
        trained_model, history = train_model(
            X_tr, y_tr, X_val, y_val,
            symbol=sym,
            profile=ai_mode,
            epochs=int(mode_cfg["default_epochs"]),
            sample_weights=time_weights,
            enable_augmentation=True,
        )
        save_scaler(fitted_scaler, sym, profile=ai_mode)
        save_meta(sym, {
            "trained_via": "predict_auto_train",
            "auto_train_reason": reason,
            "mode": ai_mode,
            "training_period": selected_period,
            "epochs_run": len(history),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "device": str(DEVICE),
            "seq_len": SEQUENCE_LENGTH,
            "input_size": len(feature_cols),
            "features": feature_cols,
        }, profile=ai_mode)
        return trained_model, fitted_scaler

    # Auto-train only when explicitly requested
    if model is None or scaler is None:
        if not auto_train:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Mã {sym} chưa có model đã huấn luyện ({ai_mode}). "
                    "Vui lòng vào trang 'Dự đoán & Huấn luyện' để train trước "
                    "hoặc gọi API với auto_train=true."
                ),
            )
        model, scaler = _auto_train_now("missing_model_or_scaler")

    # Feature-set mismatch detection (e.g. old 12-feature model vs new 20-feature pro)
    if auto_train and model is not None and scaler is not None:
        try:
            expected_features = len(feature_cols)
            meta_info = load_meta(sym, profile=ai_mode) or {}
            model_input_size = int(meta_info.get("input_size", 0) or 0)
            if model_input_size > 0 and model_input_size != expected_features:
                logger.warning(
                    "Feature-set mismatch for %s (%s): model has %d features, current config expects %d",
                    sym, ai_mode, model_input_size, expected_features,
                )
                model, scaler = _auto_train_now("feature_set_mismatch")
        except Exception:
            pass

    current_price = float(df[TARGET_COL].iloc[-1])

    # If scaler range is from stale/old-price regime, retrain automatically when allowed.
    if auto_train and scaler is not None:
        try:
            close_idx = feature_cols.index(TARGET_COL)
            s_min = float(scaler.data_min_[close_idx]) if hasattr(scaler, "data_min_") else None
            s_max = float(scaler.data_max_[close_idx]) if hasattr(scaler, "data_max_") else None
            if s_min is not None and s_max is not None and np.isfinite(s_min) and np.isfinite(s_max):
                span = max(abs(s_max - s_min), 1.0)
                far_below = current_price < (s_min - 0.6 * span)
                far_above = current_price > (s_max + 0.6 * span)
                ratio = max(current_price / max(abs(s_max), 1e-6), abs(s_max) / max(current_price, 1e-6))
                if far_below or far_above or ratio > 3.5:
                    logger.warning(
                        "Auto-retraining %s (%s): current_price=%s outside scaler range [%s, %s]",
                        sym, ai_mode, current_price, s_min, s_max,
                    )
                    model, scaler = _auto_train_now("stale_scaler_price_regime")
        except Exception:
            pass

    # Prepare last SEQUENCE_LENGTH rows for next-day prediction
    data = df[feature_cols].values.astype(np.float32)
    scaled = scaler.transform(data)
    close_idx = feature_cols.index(TARGET_COL)

    last_seq = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(feature_cols))
    pred_scaled = float(predict(model, last_seq)[0])

    ensemble_info = get_ensemble_info(sym, profile=ai_mode) or {}
    bias_correction = float(ensemble_info.get("bias_correction_scaled", 0.0) or 0.0)
    pred_scaled += bias_correction

    # Inverse-transform to get real price
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0, close_idx] = pred_scaled
    pred_price = scaler.inverse_transform(dummy)[0, close_idx]

    # Final guard for extreme mismatch between current price and prediction due stale artifacts.
    if auto_train and current_price > 0 and pred_price > 0:
        ratio = max(pred_price / current_price, current_price / pred_price)
        if ratio > 4.0:
            model, scaler = _auto_train_now("prediction_price_mismatch")
            scaled = scaler.transform(data)
            close_idx = feature_cols.index(TARGET_COL)
            last_seq = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(feature_cols))
            pred_scaled = float(predict(model, last_seq)[0]) + bias_correction
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, close_idx] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, close_idx]

    trend = classify_trend(current_price, pred_price)
    confidence = compute_confidence(current_price, pred_price)

    # Long-term: roll forward 30 steps
    lt_preds = []
    seq = scaled[-SEQUENCE_LENGTH:].copy()

    recent = df_raw.tail(min(30, len(df_raw)))
    close_s = recent["Close"].astype(float).replace(0, np.nan)
    open_ratio = float((recent["Open"].astype(float) / close_s).replace([np.inf, -np.inf], np.nan).dropna().mean() or 1.0)
    high_ratio = float((recent["High"].astype(float) / close_s).replace([np.inf, -np.inf], np.nan).dropna().mean() or 1.01)
    low_ratio = float((recent["Low"].astype(float) / close_s).replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.99)
    vol_mean = float(recent["Volume"].astype(float).tail(15).mean() or recent["Volume"].astype(float).mean())

    open_ratio = max(0.94, min(1.06, open_ratio))
    high_ratio = max(1.0, min(1.12, high_ratio))
    low_ratio = max(0.88, min(1.0, low_ratio))

    future_ohlcv = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy().reset_index(drop=True)

    for _ in range(30):
        inp = seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        p = float(predict(model, inp)[0]) + bias_correction
        lt_preds.append(p)

        d = np.zeros((1, len(feature_cols)))
        d[0, close_idx] = p
        close_price = float(scaler.inverse_transform(d)[0, close_idx])

        est_open = close_price * open_ratio
        est_high = max(close_price * high_ratio, close_price, est_open)
        est_low = min(close_price * low_ratio, close_price, est_open)

        next_raw = {
            "Open": est_open,
            "High": est_high,
            "Low": est_low,
            "Close": close_price,
            "Volume": vol_mean,
        }
        future_ohlcv = pd.concat([future_ohlcv, pd.DataFrame([next_raw])], ignore_index=True)
        future_features = prepare_feature_dataframe(future_ohlcv, ai_mode)
        feature_row = future_features[feature_cols].iloc[-1].values.astype(np.float32).reshape(1, -1)
        scaled_row = scaler.transform(feature_row)[0]

        new_row = seq[-1].copy()
        new_row[:] = scaled_row
        seq = np.vstack([seq[1:], new_row])

    # Inverse-transform long-term predictions
    lt_prices = []
    for p in lt_preds:
        d = np.zeros((1, len(feature_cols)))
        d[0, close_idx] = p
        lt_prices.append(float(scaler.inverse_transform(d)[0, close_idx]))

    lt_trend = long_term_trend(np.array(lt_prices))

    latest_feat_row = df.iloc[-1].to_dict()
    tech_payload = {
        "MA20": round(float(latest_feat_row.get("MA20", np.nan)), 4) if "MA20" in latest_feat_row else None,
        "MA50": round(float(latest_feat_row.get("MA50", np.nan)), 4) if "MA50" in latest_feat_row else None,
        "EMA20": round(float(latest_feat_row.get("EMA20", np.nan)), 4) if "EMA20" in latest_feat_row else None,
        "RSI": round(float(latest_feat_row.get("RSI", np.nan)), 4) if "RSI" in latest_feat_row else None,
        "MACD": round(float(latest_feat_row.get("MACD", np.nan)), 6) if "MACD" in latest_feat_row else None,
        "Bollinger_Band": round(float(latest_feat_row.get("Bollinger_Band", np.nan)), 6) if "Bollinger_Band" in latest_feat_row else None,
        "OBV": round(float(latest_feat_row.get("OBV", np.nan)), 2) if "OBV" in latest_feat_row else None,
    }

    for k, v in list(tech_payload.items()):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            tech_payload[k] = None

    return {
        "symbol": sym,
        "mode": ai_mode,
        "period": selected_period,
        "feature_set": feature_cols,
        "current_price": round(current_price, 2),
        "predicted_price": round(float(pred_price), 2),
        "next_day_trend": trend,
        "confidence": confidence,
        "long_term_trend": lt_trend,
        "long_term_predictions": [round(p, 2) for p in lt_prices],
        "technical_indicators": tech_payload,
        "model_device": str(DEVICE),
        "data_source": source,
        "data_csv": csv_path,
        "ensemble": ensemble_info,
    }


# ── Model status ─────────────────────────────────────────────────────
@app.get("/api/model-status/{symbol}")
async def api_model_status(symbol: str, mode: str = Query(default="simple")):
    sym = symbol.upper()
    ai_mode = normalize_ai_mode(mode)
    meta = load_meta(sym, profile=ai_mode)
    latest_version = get_latest_version_info(sym, profile=ai_mode)
    return {
        "symbol": sym,
        "mode": ai_mode,
        "trained": model_exists(sym, profile=ai_mode),
        "meta": meta,
        "latest_version": latest_version,
        "device": str(DEVICE),
    }


# ── CSV download ─────────────────────────────────────────────────────
@app.get("/api/download-csv/{symbol}")
async def api_download_csv(symbol: str, mode: str = Query(default="simple"), period: str = Query(default="1y")):
    """Generate and download a CSV of recent actual vs predicted prices."""
    sym = symbol.upper()
    ai_mode = normalize_ai_mode(mode)
    feature_cols = get_feature_columns(ai_mode)
    model = load_trained_model(sym, profile=ai_mode)
    scaler = load_scaler(sym, profile=ai_mode)
    if model is None or scaler is None:
        raise HTTPException(status_code=404,
                            detail="Chưa có model đã huấn luyện cho chế độ đã chọn. Hãy train trước qua /api/train/{symbol}.")
    try:
        df_raw, _ = get_historical_data_with_fallback(sym, period=period)
        df = prepare_feature_dataframe(df_raw, ai_mode)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    data = df[feature_cols].values.astype(np.float32)
    scaled = scaler.transform(data)
    close_idx = feature_cols.index(TARGET_COL)

    dates, actuals, preds = [], [], []
    for i in range(SEQUENCE_LENGTH, len(scaled)):
        seq = scaled[i - SEQUENCE_LENGTH : i].reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        p = predict(model, seq)[0]
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, close_idx] = p
        pred_price = scaler.inverse_transform(dummy)[0, close_idx]
        dates.append(df.index[i].strftime("%Y-%m-%d"))
        actuals.append(float(df[TARGET_COL].iloc[i]))
        preds.append(float(pred_price))

    csv_str = predictions_to_csv(dates, actuals, preds)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={sym}_predictions.csv"},
    )


# ═══════════════════════════════════════════════════════════════════════
# WebSockets
# ═══════════════════════════════════════════════════════════════════════

# ── Live price feed ──────────────────────────────────────────────────
@app.websocket("/ws/price/{symbol}")
async def ws_price(websocket: WebSocket, symbol: str):
    """Push real-time price every 60 s via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = get_realtime_price(symbol)
            await websocket.send_json(data)
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        logger.info(f"WS price client disconnected ({symbol})")


# ── Training progress feed ───────────────────────────────────────────
@app.websocket("/ws/train/{symbol}")
async def ws_train(websocket: WebSocket, symbol: str):
    """
    Start training on connect and stream epoch metrics.
    Send JSON {"epochs": N} to start.
    """
    await websocket.accept()
    try:
        msg = await websocket.receive_json()
        ai_mode = normalize_ai_mode(msg.get("mode", "simple"))
        mode_cfg = get_mode_config(ai_mode)
        train_period = str(msg.get("period") or mode_cfg["period"])
        feature_cols = get_feature_columns(ai_mode)
        epochs = int(msg.get("epochs", mode_cfg["default_epochs"]))

        df_raw, _ = get_historical_data_with_fallback(symbol, period=train_period)
        df = prepare_feature_dataframe(df_raw, ai_mode)
        X_tr, y_tr, X_val, y_val, scaler = prepare_sequences(df, feature_cols=feature_cols)

        # Compute time-based sample weights
        train_dates = get_training_sample_dates(df, len(X_tr))
        time_weights = compute_time_weights(len(X_tr), train_dates)

        loop = asyncio.get_event_loop()

        def progress_cb(epoch, train_loss, val_loss):
            """Synchronous callback → schedule coroutine on the loop."""
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                }),
                loop,
            )

        # Run training in thread so we don't block the event loop
        model, history = await loop.run_in_executor(
            None,
            lambda: train_model(X_tr, y_tr, X_val, y_val,
                                symbol=symbol.upper(), profile=ai_mode, epochs=epochs,
                                callback=progress_cb,
                                sample_weights=time_weights,
                                enable_augmentation=True),
        )
        save_scaler(scaler, symbol.upper(), profile=ai_mode)
        save_meta(symbol.upper(), {
            "mode": ai_mode,
            "training_period": train_period,
            "epochs_run": len(history),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "device": str(DEVICE),
            "seq_len": SEQUENCE_LENGTH,
            "input_size": len(feature_cols),
            "features": feature_cols,
        }, profile=ai_mode)
        await websocket.send_json({"status": "done", "epochs_run": len(history), "mode": ai_mode, "period": train_period})
    except WebSocketDisconnect:
        logger.info(f"WS train client disconnected ({symbol})")
    except Exception as e:
        logger.exception("WS training error")
        await websocket.send_json({"status": "error", "detail": str(e)})


# ── Startup event ────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info(f"ViziGenesis starting — device: {DEVICE}")
    logger.info(f"GPU available: {DEVICE.type == 'cuda'}")
    model_migration = migrate_legacy_artifacts()
    data_migration = migrate_legacy_data_files()
    logger.info(
        "Legacy migration complete | models moved=%s, data moved=%s",
        model_migration.get("moved_count", 0),
        data_migration.get("moved_count", 0),
    )


# ═══════════════════════════════════════════════════════════════════════
# Quant Mode — Institutional-grade endpoints
# ═══════════════════════════════════════════════════════════════════════

# Lazy imports to avoid circular deps — loaded on first quant call
_quant_loaded = False
_quant_modules = {}


def _ensure_quant_imports():
    global _quant_loaded, _quant_modules
    if _quant_loaded:
        return _quant_modules
    try:
        from backend.quant_model import (
            train_quant_model, predict_quant,
            save_quant_model, load_quant_model, load_quant_meta,
            quant_model_exists, save_quant_scaler, load_quant_scaler,
            save_quant_metrics, load_quant_metrics, QuantHybridModel,
        )
        from backend.quant_features import (
            add_quant_technical_indicators, generate_quant_targets,
            prepare_quant_sequences,
        )
        from backend.walk_forward import (
            run_walk_forward_validation, save_calibrator, load_calibrator,
        )
        from backend.backtest import run_backtest

        _quant_modules = {
            "train_quant_model": train_quant_model,
            "predict_quant": predict_quant,
            "save_quant_model": save_quant_model,
            "load_quant_model": load_quant_model,
            "load_quant_meta": load_quant_meta,
            "quant_model_exists": quant_model_exists,
            "save_quant_scaler": save_quant_scaler,
            "load_quant_scaler": load_quant_scaler,
            "save_quant_metrics": save_quant_metrics,
            "load_quant_metrics": load_quant_metrics,
            "prepare_quant_sequences": prepare_quant_sequences,
            "run_walk_forward_validation": run_walk_forward_validation,
            "save_calibrator": save_calibrator,
            "load_calibrator": load_calibrator,
            "run_backtest": run_backtest,
        }
        _quant_loaded = True
    except ImportError as e:
        logger.error("Quant modules not available: %s", e)
        raise HTTPException(status_code=501, detail=f"Quant modules not installed: {e}")
    return _quant_modules


# ── Quant Train (full pipeline: walk-forward + calibration) ──────────
@app.post("/api/quant/train/{symbol}")
async def api_quant_train(
    symbol: str,
    period: Optional[str] = Query(default="10y"),
    epochs: Optional[int] = Query(default=None, ge=10, le=600),
):
    """
    Train the quant hybrid model via walk-forward validation.
    Includes probability calibration and backtest.
    """
    qm = _ensure_quant_imports()
    sym = symbol.upper()
    feature_cols = list(QUANT_FEATURE_COLS)

    try:
        df_raw, source = get_historical_data_with_fallback(sym, period=period or "10y")
        if source in {"local_sample_csv", "generated_sample_csv"}:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot train quant model for {sym} with fallback data ({source}).",
            )

        df = prepare_feature_dataframe(df_raw, "quant")
        targets_df = prepare_quant_targets_from_raw(df_raw, df)

        if len(df) < 500:
            raise HTTPException(
                status_code=400,
                detail=f"Quant mode requires at least 500 trading days. Got {len(df)}.",
            )

        # Run walk-forward validation
        loop = asyncio.get_event_loop()
        wf_result = await loop.run_in_executor(
            None,
            lambda: qm["run_walk_forward_validation"](
                df, targets_df, feature_cols,
                seq_len=QUANT_SEQUENCE_LENGTH,
                epochs_per_fold=min(int(epochs or 80), 80),
            ),
        )

        # Save artifacts
        final_model = wf_result["final_model"]
        final_scaler = wf_result["final_scaler"]
        calibrator = wf_result["calibrator"]

        qm["save_quant_model"](final_model, sym, {
            "mode": "quant",
            "training_period": period or "10y",
            "n_folds": wf_result["aggregate_metrics"].get("n_folds", 0),
            "total_val_samples": wf_result["aggregate_metrics"].get("total_val_samples", 0),
            "data_rows": len(df),
            "n_features": len(feature_cols),
            "features": feature_cols,
            "device": str(DEVICE),
            "data_source": source,
        })
        qm["save_quant_scaler"](final_scaler, sym)

        if calibrator is not None:
            qm["save_calibrator"](calibrator, sym)

        # Save metrics
        metrics = wf_result["aggregate_metrics"]
        qm["save_quant_metrics"](sym, metrics)

        # Run backtest on pooled validation data
        backtest_result = None
        if len(wf_result.get("pooled_val_probs", [])) > 20:
            try:
                # Get actual returns for the validation period
                close_vals = df["Close"].values.astype(float)
                actual_daily_returns = np.diff(close_vals) / close_vals[:-1]

                val_n = len(wf_result["pooled_val_probs"])
                # Use last val_n returns as approximation
                bt_returns = actual_daily_returns[-val_n:] if len(actual_daily_returns) >= val_n else actual_daily_returns
                bt_probs = wf_result["pooled_val_probs"][:len(bt_returns)]

                if calibrator is not None:
                    bt_probs_cal = calibrator.predict(bt_probs)
                else:
                    bt_probs_cal = bt_probs

                bt_dates = pd.to_datetime(df.index[-len(bt_returns):])
                bt_result = qm["run_backtest"](bt_dates, bt_returns, bt_probs_cal)
                backtest_result = bt_result.metrics
            except Exception as e:
                logger.warning("Backtest failed: %s", e)

        return {
            "status": "ok",
            "symbol": sym,
            "mode": "quant",
            "device": str(DEVICE),
            "data_source": source,
            "data_rows": len(df),
            "n_features": len(feature_cols),
            "walk_forward": {
                "n_folds": metrics.get("n_folds", 0),
                "total_val_samples": metrics.get("total_val_samples", 0),
                "raw_metrics": metrics.get("raw", {}),
                "calibrated_metrics": metrics.get("calibrated", {}),
                "reliability_diagram": metrics.get("reliability_diagram", []),
            },
            "backtest": backtest_result,
            "final_training": {
                "epochs_run": len(wf_result.get("final_history", [])),
                "final_val_loss": wf_result["final_history"][-1]["val_loss"] if wf_result.get("final_history") else None,
            },
            "calibrator_available": calibrator is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Quant training failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Quant Predict ─────────────────────────────────────────────────────
@app.get("/api/quant/predict/{symbol}")
async def api_quant_predict(
    symbol: str,
    auto_train: bool = Query(default=False),
    period: Optional[str] = Query(default="10y"),
):
    """
    Get quant-grade predictions with calibrated probabilities,
    multi-horizon returns, and AI explanation.
    """
    qm = _ensure_quant_imports()
    sym = symbol.upper()
    feature_cols = list(QUANT_FEATURE_COLS)

    try:
        df_raw, source = get_historical_data_with_fallback(sym, period=period or "10y")
        if source in {"local_sample_csv", "generated_sample_csv"}:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot predict {sym} with fallback data ({source}).",
            )

        df = prepare_feature_dataframe(df_raw, "quant")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Load model + scaler + calibrator
    model = qm["load_quant_model"](sym, len(feature_cols))
    scaler = qm["load_quant_scaler"](sym)
    calibrator = qm["load_calibrator"](sym)

    if model is None or scaler is None:
        if not auto_train:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"No quant model trained for {sym}. "
                    "Please train via /api/quant/train/{symbol} first, or use auto_train=true."
                ),
            )
        # Auto-train
        logger.info("Auto-training quant model for %s", sym)
        targets_df = prepare_quant_targets_from_raw(df_raw, df)
        loop = asyncio.get_event_loop()
        wf_result = await loop.run_in_executor(
            None,
            lambda: qm["run_walk_forward_validation"](
                df, targets_df, feature_cols,
                seq_len=QUANT_SEQUENCE_LENGTH,
            ),
        )
        model = wf_result["final_model"]
        scaler = wf_result["final_scaler"]
        calibrator = wf_result["calibrator"]
        qm["save_quant_model"](model, sym, {"mode": "quant", "auto_trained": True})
        qm["save_quant_scaler"](scaler, sym)
        if calibrator:
            qm["save_calibrator"](calibrator, sym)

    # Run prediction on latest sequence
    data = df[feature_cols].values.astype(np.float32)
    scaled = scaler.transform(data)
    seq_len = QUANT_SEQUENCE_LENGTH
    last_seq = scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))

    raw_preds = qm["predict_quant"](model, last_seq)

    # Extract predictions
    raw_p_up = float(raw_preds["direction"][0, 0])
    ret_1d = float(raw_preds["return_1d"][0, 0])
    ret_5d = float(raw_preds["return_5d"][0, 0])
    ret_30d = float(raw_preds["return_30d"][0, 0])
    excess = float(raw_preds["excess"][0, 0])

    # Calibrate probability
    if calibrator is not None:
        cal_p_up = float(calibrator.predict([raw_p_up])[0])
    else:
        cal_p_up = raw_p_up

    cal_p_down = 1.0 - cal_p_up

    # Branch weights (how the ensemble weighted each branch)
    branch_weights = raw_preds["branch_weights"][0].tolist()

    # Feature importance from TFT VSN
    feat_weights = raw_preds["feat_weights"][0].tolist()
    feat_importance = sorted(
        [{"feature": feature_cols[i], "weight": round(float(feat_weights[i]), 4)}
         for i in range(len(feature_cols))],
        key=lambda x: x["weight"],
        reverse=True,
    )[:15]  # top 15

    current_price = float(df["Close"].iloc[-1])

    # Predicted price from 1d return
    predicted_price_1d = current_price * (1 + ret_1d / 100)
    predicted_price_5d = current_price * (1 + ret_5d / 100)
    predicted_price_30d = current_price * (1 + ret_30d / 100)

    # Trend classification
    if cal_p_up > 0.60:
        direction = "UP"
        signal_strength = "STRONG" if cal_p_up > 0.70 else "MODERATE"
    elif cal_p_down > 0.60:
        direction = "DOWN"
        signal_strength = "STRONG" if cal_p_down > 0.70 else "MODERATE"
    else:
        direction = "NEUTRAL"
        signal_strength = "WEAK"

    # AI explanation summary
    top_3_features = feat_importance[:3]
    explanation_parts = [
        f"Model direction: {direction} ({signal_strength}) — calibrated P(up)={cal_p_up:.1%}, P(down)={cal_p_down:.1%}.",
        f"1-day predicted return: {ret_1d:+.2f}% → ${predicted_price_1d:.2f}.",
        f"5-day return: {ret_5d:+.2f}%, 30-day return: {ret_30d:+.2f}%.",
        f"Excess vs NASDAQ: {excess:+.2f}%.",
        f"Top drivers: {', '.join(f['feature'] for f in top_3_features)}.",
        f"Ensemble weights: TFT={branch_weights[0]:.1%}, BiLSTM={branch_weights[1]:.1%}, GRU={branch_weights[2]:.1%}.",
    ]

    # Load metrics if available
    metrics = qm["load_quant_metrics"](sym)

    return {
        "symbol": sym,
        "mode": "quant",
        "current_price": round(current_price, 2),
        "direction": {
            "signal": direction,
            "strength": signal_strength,
            "calibrated_p_up": round(cal_p_up, 4),
            "calibrated_p_down": round(cal_p_down, 4),
            "raw_p_up": round(raw_p_up, 4),
        },
        "multi_horizon": {
            "return_1d_pct": round(ret_1d, 4),
            "return_5d_pct": round(ret_5d, 4),
            "return_30d_pct": round(ret_30d, 4),
            "price_1d": round(predicted_price_1d, 2),
            "price_5d": round(predicted_price_5d, 2),
            "price_30d": round(predicted_price_30d, 2),
        },
        "excess_return_vs_nasdaq_pct": round(excess, 4),
        "ensemble": {
            "branch_weights": {
                "tft": round(branch_weights[0], 4),
                "bilstm": round(branch_weights[1], 4),
                "gru": round(branch_weights[2], 4),
            },
        },
        "feature_importance": feat_importance,
        "ai_explanation": " ".join(explanation_parts),
        "walk_forward_metrics": metrics,
        "calibrator_available": calibrator is not None,
        "data_source": source,
        "model_device": str(DEVICE),
    }


# ── Quant Model Status ───────────────────────────────────────────────
@app.get("/api/quant/status/{symbol}")
async def api_quant_status(symbol: str):
    """Check quant model training status and metrics."""
    qm = _ensure_quant_imports()
    sym = symbol.upper()
    meta = qm["load_quant_meta"](sym)
    metrics = qm["load_quant_metrics"](sym)
    has_calibrator = qm["load_calibrator"](sym) is not None

    return {
        "symbol": sym,
        "mode": "quant",
        "trained": qm["quant_model_exists"](sym),
        "meta": meta,
        "metrics": metrics,
        "calibrator_available": has_calibrator,
        "device": str(DEVICE),
    }


# ── WebSocket for quant training progress ─────────────────────────────
@app.websocket("/ws/quant/train/{symbol}")
async def ws_quant_train(websocket: WebSocket, symbol: str):
    """Stream quant training progress (walk-forward folds + final training)."""
    await websocket.accept()
    qm = _ensure_quant_imports()
    try:
        msg = await websocket.receive_json()
        train_period = str(msg.get("period", "10y"))
        epochs = int(msg.get("epochs", 80))
        feature_cols = list(QUANT_FEATURE_COLS)

        df_raw, source = get_historical_data_with_fallback(symbol, period=train_period)
        df = prepare_feature_dataframe(df_raw, "quant")
        targets_df = prepare_quant_targets_from_raw(df_raw, df)

        loop = asyncio.get_event_loop()

        def progress_cb(fold_or_phase, epoch, train_loss, val_loss):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "phase": str(fold_or_phase),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }),
                loop,
            )

        wf_result = await loop.run_in_executor(
            None,
            lambda: qm["run_walk_forward_validation"](
                df, targets_df, feature_cols,
                seq_len=QUANT_SEQUENCE_LENGTH,
                epochs_per_fold=min(epochs, 80),
                callback=progress_cb,
            ),
        )

        # Save everything
        qm["save_quant_model"](wf_result["final_model"], symbol.upper(), {
            "mode": "quant", "training_period": train_period, "device": str(DEVICE),
        })
        qm["save_quant_scaler"](wf_result["final_scaler"], symbol.upper())
        if wf_result["calibrator"]:
            qm["save_calibrator"](wf_result["calibrator"], symbol.upper())
        qm["save_quant_metrics"](symbol.upper(), wf_result["aggregate_metrics"])

        await websocket.send_json({
            "status": "done",
            "mode": "quant",
            "metrics": wf_result["aggregate_metrics"],
        })

    except WebSocketDisconnect:
        logger.info("WS quant train client disconnected (%s)", symbol)
    except Exception as e:
        logger.exception("WS quant training error")
        await websocket.send_json({"status": "error", "detail": str(e)})


# ═══════════════════════════════════════════════════════════════════════
# V2 Mode — Institutional-grade panel-trained system
# ═══════════════════════════════════════════════════════════════════════

_v2_loaded = False
_v2_modules = {}


def _ensure_v2_imports():
    """Lazy-load V2 modules (heavy deps: lightgbm, xgboost, etc.)."""
    global _v2_loaded, _v2_modules
    if _v2_loaded:
        return _v2_modules
    try:
        from backend.v2.config import (
            PILOT_TICKERS, V2_FEATURE_COLS, D_MODEL, SEQ_LEN as V2_SEQ_LEN,
            BT_SIGNAL_LONG, BT_SIGNAL_SHORT, BT_INITIAL_CAPITAL,
        )
        from backend.v2.panel_data import build_panel_dataset, split_panel_data, fetch_stock_data
        from backend.v2.features import (
            build_full_features, generate_targets, generate_regime_labels,
            prepare_panel_sequences,
        )
        from backend.v2.model import (
            HybridForecaster, train_hybrid_model, predict_hybrid,
            save_v2_model, load_v2_model, save_v2_scaler, load_v2_scaler,
            save_v2_metrics, load_v2_metrics,
        )
        from backend.v2.tree_models import (
            train_lightgbm, train_xgboost, predict_tree_models,
            save_tree_models, load_tree_models,
        )
        from backend.v2.meta_router import (
            combine_predictions, compute_uncertainty, rule_based_weights,
        )
        from backend.v2.calibration import (
            CombinedCalibrator, compute_classification_metrics,
            detect_concept_drift,
        )
        from backend.v2.walk_forward import run_walk_forward_validation as run_v2_wf
        from backend.v2.backtest import run_backtest as run_v2_backtest, run_stress_tests
        from backend.v2.regime import detect_regime, get_current_regime
        from backend.v2.sentiment import build_sentiment_features
        from backend.v2.market_data import fetch_fred_macro

        _v2_modules = {
            "PILOT_TICKERS": PILOT_TICKERS,
            "V2_FEATURE_COLS": V2_FEATURE_COLS,
            "D_MODEL": D_MODEL,
            "V2_SEQ_LEN": V2_SEQ_LEN,
            "build_panel_dataset": build_panel_dataset,
            "split_panel_data": split_panel_data,
            "fetch_stock_data": fetch_stock_data,
            "build_full_features": build_full_features,
            "generate_targets": generate_targets,
            "generate_regime_labels": generate_regime_labels,
            "prepare_panel_sequences": prepare_panel_sequences,
            "HybridForecaster": HybridForecaster,
            "train_hybrid_model": train_hybrid_model,
            "predict_hybrid": predict_hybrid,
            "save_v2_model": save_v2_model,
            "load_v2_model": load_v2_model,
            "save_v2_scaler": save_v2_scaler,
            "load_v2_scaler": load_v2_scaler,
            "save_v2_metrics": save_v2_metrics,
            "load_v2_metrics": load_v2_metrics,
            "train_lightgbm": train_lightgbm,
            "train_xgboost": train_xgboost,
            "predict_tree_models": predict_tree_models,
            "save_tree_models": save_tree_models,
            "load_tree_models": load_tree_models,
            "combine_predictions": combine_predictions,
            "compute_uncertainty": compute_uncertainty,
            "rule_based_weights": rule_based_weights,
            "CombinedCalibrator": CombinedCalibrator,
            "compute_classification_metrics": compute_classification_metrics,
            "detect_concept_drift": detect_concept_drift,
            "run_v2_wf": run_v2_wf,
            "run_v2_backtest": run_v2_backtest,
            "run_stress_tests": run_stress_tests,
            "detect_regime": detect_regime,
            "get_current_regime": get_current_regime,
            "build_sentiment_features": build_sentiment_features,
            "fetch_fred_macro": fetch_fred_macro,
        }
        _v2_loaded = True
        logger.info("V2 modules loaded successfully")
    except ImportError as e:
        logger.error("V2 modules not available: %s", e)
        raise HTTPException(status_code=501, detail=f"V2 modules not available: {e}")
    return _v2_modules


# ── V2 Panel Train (walk-forward + calibration + backtest) ────────────
@app.post("/api/v2/train")
async def api_v2_train(
    tickers: Optional[str] = Query(default=None, description="Comma-sep tickers, default=pilot"),
    period: Optional[str] = Query(default="10y"),
    epochs: Optional[int] = Query(default=None, ge=10, le=600),
):
    """
    Train V2 panel model across multiple stocks.
    Walk-forward cross-validated with calibration and backtest.
    """
    v2 = _ensure_v2_imports()
    try:
        ticker_list = (
            [t.strip().upper() for t in tickers.split(",") if t.strip()]
            if tickers else list(v2["PILOT_TICKERS"])
        )

        # Build panel dataset
        loop = asyncio.get_event_loop()
        panel = await loop.run_in_executor(
            None,
            lambda: v2["build_panel_dataset"](ticker_list, period=period or "10y"),
        )

        if panel is None or len(panel.get("X", [])) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data to build panel dataset.")

        # Split summary (split_panel_data returns a dict, not tuple)
        split_data = v2["split_panel_data"](panel)

        # Walk-forward validation
        wf_result = await loop.run_in_executor(
            None,
            lambda: v2["run_v2_wf"](
                panel["X"],
                panel["y_direction"],
                panel["y_returns"],
                panel["y_regime"],
                panel["stock_ids"],
                panel["dates"],
                n_stocks=int(panel.get("n_stocks", len(ticker_list))),
                n_features=int(panel["X"].shape[-1]) if hasattr(panel.get("X"), "shape") else 0,
                epochs_per_fold=min(int(epochs or 80), 80),
            ),
        )

        # Save all artifacts
        model_tag = "_".join(ticker_list[:3])
        v2["save_v2_model"](wf_result["final_model"], model_tag)
        v2["save_v2_scaler"](panel["scaler"], model_tag)
        v2["save_v2_metrics"](model_tag, wf_result["aggregate_metrics"])

        # Save tree models if trained
        if "tree_models" in wf_result:
            v2["save_tree_models"](wf_result["tree_models"], model_tag)

        return {
            "status": "ok",
            "tickers": ticker_list,
            "mode": "v2_panel",
            "device": str(DEVICE),
            "n_samples": len(panel["X"]),
            "n_features": panel["X"].shape[-1] if hasattr(panel["X"], "shape") else 0,
            "split": {
                "n_train": int(split_data.get("n_train", 0)),
                "n_val": int(split_data.get("n_val", 0)),
                "n_test": int(split_data.get("n_test", 0)),
            },
            "walk_forward": wf_result["aggregate_metrics"],
            "calibrator_available": wf_result.get("calibrator") is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("V2 training failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── V2 Predict (single stock using panel model) ──────────────────────
@app.get("/api/v2/predict/{symbol}")
async def api_v2_predict(
    symbol: str,
    period: Optional[str] = Query(default="2y"),
    model_tag: Optional[str] = Query(default=None, description="Model tag (default: pilot tag)"),
):
    """
    V2 prediction: calibrated direction, multi-horizon returns,
    regime detection, feature importance, AI explanation.
    """
    v2 = _ensure_v2_imports()
    sym = symbol.upper()
    tag = model_tag or "_".join(list(v2["PILOT_TICKERS"])[:3])

    try:
        # Load model + scaler
        feature_cols = list(v2["V2_FEATURE_COLS"])
        model = v2["load_v2_model"](tag, len(feature_cols))
        scaler = v2["load_v2_scaler"](tag)

        if model is None or scaler is None:
            raise HTTPException(
                status_code=409,
                detail=f"No V2 model found for tag '{tag}'. Train first via POST /api/v2/train.",
            )

        # Fetch & prepare features for this symbol
        loop = asyncio.get_event_loop()
        df_raw = await loop.run_in_executor(None, lambda: v2["fetch_stock_data"](sym, period=period or "2y"))
        df_feat = await loop.run_in_executor(None, lambda: v2["build_full_features"](df_raw, sym))

        if df_feat is None or len(df_feat) < v2["V2_SEQ_LEN"]:
            raise HTTPException(status_code=400, detail=f"Insufficient feature data for {sym}.")

        # Regime detection
        regime_info = v2["get_current_regime"](df_feat)

        # Prepare sequence
        data = df_feat[feature_cols].values.astype(np.float32)
        scaled = scaler.transform(data)
        seq_len = v2["V2_SEQ_LEN"]
        last_seq = scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))

        # Neural prediction
        import torch
        x_tensor = torch.FloatTensor(last_seq).to(DEVICE)
        preds = v2["predict_hybrid"](model, x_tensor)

        raw_p_up = float(preds["direction"][0])
        ret_1d = float(preds["return_1d"][0])
        ret_5d = float(preds["return_5d"][0])
        ret_30d = float(preds["return_30d"][0])
        excess = float(preds["excess"][0])

        # Tree model predictions (if available)
        tree_models = v2["load_tree_models"](tag)
        tree_preds = None
        if tree_models is not None:
            from backend.v2.tree_models import flatten_sequences
            flat = flatten_sequences(last_seq)
            tree_preds = v2["predict_tree_models"](tree_models, flat)

        # Uncertainty
        uncertainty = v2["compute_uncertainty"](raw_p_up)

        # Meta-router blending
        current_vix = float(df_feat["VIX"].iloc[-1]) if "VIX" in df_feat.columns else 20.0
        regime_label = regime_info.get("regime", "sideways")
        weights = v2["rule_based_weights"](regime_label, uncertainty, current_vix)

        if tree_preds is not None:
            combined = v2["combine_predictions"](
                neural_preds={"direction": np.array([raw_p_up]), "return_1d": np.array([ret_1d])},
                lgbm_preds=tree_preds.get("lgbm"),
                xgb_preds=tree_preds.get("xgb"),
                weights=weights,
            )
            final_p_up = float(combined["direction"][0])
            final_ret_1d = float(combined["return_1d"][0])
        else:
            final_p_up = raw_p_up
            final_ret_1d = ret_1d

        # Direction + strength
        p_down = 1.0 - final_p_up
        if final_p_up > 0.60:
            direction = "UP"
            strength = "STRONG" if final_p_up > 0.70 else "MODERATE"
        elif p_down > 0.60:
            direction = "DOWN"
            strength = "STRONG" if p_down > 0.70 else "MODERATE"
        else:
            direction = "NEUTRAL"
            strength = "WEAK"

        current_price = float(df_feat["Close"].iloc[-1])

        # Branch weights
        branch_w = preds.get("branch_weights", [[0.33, 0.33, 0.34]])[0]

        # Feature importance
        feat_w = preds.get("feat_weights", [[]])[0]
        feat_importance = []
        if len(feat_w) == len(feature_cols):
            feat_importance = sorted(
                [{"feature": feature_cols[i], "weight": round(float(feat_w[i]), 4)}
                 for i in range(len(feature_cols))],
                key=lambda x: x["weight"], reverse=True,
            )[:15]

        # AI explanation
        explanation = (
            f"V2 Panel Model | Direction: {direction} ({strength}) — P(up)={final_p_up:.1%}. "
            f"Regime: {regime_label} (confidence {regime_info.get('confidence', 0):.0%}). "
            f"1d return: {final_ret_1d:+.2f}%, 5d: {ret_5d:+.2f}%, 30d: {ret_30d:+.2f}%. "
            f"Excess vs NASDAQ: {excess:+.2f}%. "
            f"Uncertainty: {uncertainty:.1%}. "
            f"Meta-router weights: neural={weights.get('neural', 0):.0%}, "
            f"lgbm={weights.get('lgbm', 0):.0%}, xgb={weights.get('xgb', 0):.0%}."
        )

        return {
            "symbol": sym,
            "mode": "v2_panel",
            "model_tag": tag,
            "current_price": round(current_price, 2),
            "direction": {
                "signal": direction,
                "strength": strength,
                "p_up": round(final_p_up, 4),
                "p_down": round(1.0 - final_p_up, 4),
                "raw_neural_p_up": round(raw_p_up, 4),
            },
            "multi_horizon": {
                "return_1d_pct": round(final_ret_1d, 4),
                "return_5d_pct": round(ret_5d, 4),
                "return_30d_pct": round(ret_30d, 4),
                "price_1d": round(current_price * (1 + final_ret_1d / 100), 2),
                "price_5d": round(current_price * (1 + ret_5d / 100), 2),
                "price_30d": round(current_price * (1 + ret_30d / 100), 2),
            },
            "excess_return_vs_nasdaq_pct": round(excess, 4),
            "regime": regime_info,
            "uncertainty": round(uncertainty, 4),
            "meta_router_weights": weights,
            "ensemble": {
                "tft": round(float(branch_w[0]), 4) if len(branch_w) > 0 else None,
                "bilstm": round(float(branch_w[1]), 4) if len(branch_w) > 1 else None,
                "gru": round(float(branch_w[2]), 4) if len(branch_w) > 2 else None,
            },
            "feature_importance": feat_importance,
            "ai_explanation": explanation,
            "device": str(DEVICE),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("V2 prediction failed for %s", sym)
        raise HTTPException(status_code=500, detail=str(e))


# ── V2 Regime ─────────────────────────────────────────────────────────
@app.get("/api/v2/regime/{symbol}")
async def api_v2_regime(symbol: str, period: str = Query(default="2y")):
    """Get current market regime for a symbol."""
    v2 = _ensure_v2_imports()
    sym = symbol.upper()
    try:
        loop = asyncio.get_event_loop()
        df_raw = await loop.run_in_executor(None, lambda: v2["fetch_stock_data"](sym, period=period))
        df_feat = await loop.run_in_executor(None, lambda: v2["build_full_features"](df_raw, sym))
        regime = v2["get_current_regime"](df_feat)
        return {"symbol": sym, "regime": regime}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── V2 Backtest ───────────────────────────────────────────────────────
@app.post("/api/v2/backtest/{symbol}")
async def api_v2_backtest(
    symbol: str,
    period: Optional[str] = Query(default="5y"),
    model_tag: Optional[str] = Query(default=None),
):
    """
    Run full V2 backtest with stress tests.
    """
    v2 = _ensure_v2_imports()
    sym = symbol.upper()
    tag = model_tag or "_".join(list(v2["PILOT_TICKERS"])[:3])

    try:
        feature_cols = list(v2["V2_FEATURE_COLS"])
        model = v2["load_v2_model"](tag, len(feature_cols))
        scaler = v2["load_v2_scaler"](tag)

        if model is None or scaler is None:
            raise HTTPException(status_code=409, detail=f"No V2 model for tag '{tag}'.")

        loop = asyncio.get_event_loop()
        df_raw = await loop.run_in_executor(None, lambda: v2["fetch_stock_data"](sym, period=period or "5y"))
        df_feat = await loop.run_in_executor(None, lambda: v2["build_full_features"](df_raw, sym))

        if df_feat is None or len(df_feat) < v2["V2_SEQ_LEN"] + 100:
            raise HTTPException(status_code=400, detail="Insufficient data for backtest.")

        # Generate predictions for entire history
        data = df_feat[feature_cols].values.astype(np.float32)
        scaled = scaler.transform(data)
        seq_len = v2["V2_SEQ_LEN"]

        import torch
        direction_probs = []
        expected_rets = []
        for i in range(seq_len, len(scaled)):
            seq = scaled[i - seq_len:i].reshape(1, seq_len, len(feature_cols))
            x_tensor = torch.FloatTensor(seq).to(DEVICE)
            preds = v2["predict_hybrid"](model, x_tensor)
            direction_probs.append(float(preds["direction"][0]))
            expected_rets.append(float(preds["return_1d"][0]))

        direction_probs = np.array(direction_probs)
        expected_rets = np.array(expected_rets)

        # Actual returns
        close_vals = df_feat["Close"].values.astype(float)
        actual_rets = np.diff(close_vals) / close_vals[:-1]
        bt_returns = actual_rets[seq_len:][:len(direction_probs)]
        bt_dates = pd.to_datetime(df_feat.index[seq_len + 1:seq_len + 1 + len(bt_returns)])

        bt_result = v2["run_v2_backtest"](
            bt_dates, bt_returns, direction_probs[:len(bt_returns)],
            expected_returns=expected_rets[:len(bt_returns)],
        )

        # Stress tests
        stress = v2["run_stress_tests"](bt_result, bt_dates)
        bt_result.stress_tests = stress

        return {
            "symbol": sym,
            "mode": "v2_panel",
            "model_tag": tag,
            "metrics": bt_result.metrics,
            "monthly_pnl": bt_result.monthly_pnl,
            "stress_tests": stress,
            "n_trading_days": len(bt_returns),
            "equity_curve_sample": bt_result.equity_curve[::max(1, len(bt_result.equity_curve) // 200)],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("V2 backtest failed for %s", sym)
        raise HTTPException(status_code=500, detail=str(e))


# ── V2 Status ─────────────────────────────────────────────────────────
@app.get("/api/v2/status")
async def api_v2_status(
    model_tag: Optional[str] = Query(default=None),
):
    """Check V2 model status and metrics."""
    v2 = _ensure_v2_imports()
    tag = model_tag or "_".join(list(v2["PILOT_TICKERS"])[:3])
    feature_cols = list(v2["V2_FEATURE_COLS"])

    model = v2["load_v2_model"](tag, len(feature_cols))
    metrics = v2["load_v2_metrics"](tag)

    return {
        "mode": "v2_panel",
        "model_tag": tag,
        "trained": model is not None,
        "metrics": metrics,
        "pilot_tickers": list(v2["PILOT_TICKERS"]),
        "n_features": len(feature_cols),
        "device": str(DEVICE),
    }


# ── WebSocket for V2 training progress ────────────────────────────────
@app.websocket("/ws/v2/train")
async def ws_v2_train(websocket: WebSocket):
    """Stream V2 panel training progress."""
    await websocket.accept()
    v2 = _ensure_v2_imports()
    try:
        msg = await websocket.receive_json()
        tickers_raw = msg.get("tickers", None)
        ticker_list = (
            [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
            if tickers_raw else list(v2["PILOT_TICKERS"])
        )
        train_period = str(msg.get("period", "10y"))
        epochs = int(msg.get("epochs", 80))

        loop = asyncio.get_event_loop()

        # Phase 1: Build panel
        await websocket.send_json({"phase": "building_panel", "tickers": ticker_list})
        panel = await loop.run_in_executor(
            None,
            lambda: v2["build_panel_dataset"](ticker_list, period=train_period),
        )

        if panel is None or len(panel.get("X", [])) < 100:
            await websocket.send_json({"status": "error", "detail": "Insufficient panel data."})
            return

        await websocket.send_json({
            "phase": "panel_ready",
            "n_samples": len(panel["X"]),
            "n_features": panel["X"].shape[-1] if hasattr(panel["X"], "shape") else 0,
        })

        # Phase 2: Walk-forward training with progress callback
        def progress_cb(fold_or_phase, epoch, train_loss, val_loss):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "phase": str(fold_or_phase),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }),
                loop,
            )

        wf_result = await loop.run_in_executor(
            None,
            lambda: v2["run_v2_wf"](panel, epochs_per_fold=min(epochs, 80)),
        )

        # Phase 3: Save artifacts
        model_tag = "_".join(ticker_list[:3])
        v2["save_v2_model"](wf_result["final_model"], model_tag)
        v2["save_v2_scaler"](panel["scaler"], model_tag)
        v2["save_v2_metrics"](model_tag, wf_result["aggregate_metrics"])

        await websocket.send_json({
            "status": "done",
            "mode": "v2_panel",
            "model_tag": model_tag,
            "metrics": wf_result["aggregate_metrics"],
        })

    except WebSocketDisconnect:
        logger.info("WS V2 train client disconnected")
    except Exception as e:
        logger.exception("WS V2 training error")
        try:
            await websocket.send_json({"status": "error", "detail": str(e)})
        except Exception:
            pass
