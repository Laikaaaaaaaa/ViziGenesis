"""
Pipeline utilities for training/evaluation with explicit train/val/test split.
Includes regression metrics, trend quality metrics, and artifact helpers.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MinMaxScaler

from backend.config import FEATURE_COLS, TARGET_COL, SEQUENCE_LENGTH
from backend.model import classify_trend, symbol_model_dir


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    scaler: MinMaxScaler


def build_train_val_test_split(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    feature_cols: List[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> SplitData:
    """
    Build train/val/test sequences.
    Scaler is fitted only on train range to avoid data leakage.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    data = df[feature_cols].values.astype(np.float32)
    n = len(data)
    if n <= seq_len + 5:
        raise ValueError("Not enough rows to create sequences.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = MinMaxScaler()
    scaler.fit(data[:train_end])
    scaled = scaler.transform(data)

    close_idx = feature_cols.index(target_col)

    X_train, y_train, idx_train = [], [], []
    X_val, y_val, idx_val = [], [], []
    X_test, y_test, idx_test = [], [], []

    for i in range(seq_len, n):
        seq = scaled[i - seq_len : i]
        target = scaled[i, close_idx]

        if i < train_end:
            X_train.append(seq)
            y_train.append(target)
            idx_train.append(i)
        elif i < val_end:
            X_val.append(seq)
            y_val.append(target)
            idx_val.append(i)
        else:
            X_test.append(seq)
            y_test.append(target)
            idx_test.append(i)

    return SplitData(
        X_train=np.array(X_train), y_train=np.array(y_train),
        X_val=np.array(X_val), y_val=np.array(y_val),
        X_test=np.array(X_test), y_test=np.array(y_test),
        idx_train=np.array(idx_train), idx_val=np.array(idx_val), idx_test=np.array(idx_test),
        scaler=scaler,
    )


def inverse_close(
    scaler: MinMaxScaler,
    scaled_close: np.ndarray,
    feature_cols: List[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> np.ndarray:
    """Inverse transform close-price values from scaled space to real price."""
    close_idx = feature_cols.index(target_col)
    dummy = np.zeros((len(scaled_close), len(feature_cols)), dtype=np.float32)
    dummy[:, close_idx] = scaled_close
    inv = scaler.inverse_transform(dummy)
    return inv[:, close_idx]


def evaluate_predictions(
    df: pd.DataFrame,
    idx: np.ndarray,
    pred_close: np.ndarray,
    actual_close: np.ndarray,
) -> Dict:
    """Compute loss, accuracy, and trend quality metrics."""
    if len(idx) == 0:
        raise ValueError("Empty evaluation set.")

    mse = float(mean_squared_error(actual_close, pred_close))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(actual_close, pred_close))
    mape = float(np.mean(np.abs((actual_close - pred_close) / np.clip(actual_close, 1e-6, None))) * 100)

    # Trend labels from previous close -> current close / predicted close
    actual_trends = []
    pred_trends = []
    for i, pred, actual in zip(idx, pred_close, actual_close):
        prev_close = float(df[TARGET_COL].iloc[i - 1])
        actual_trends.append(classify_trend(prev_close, float(actual)))
        pred_trends.append(classify_trend(prev_close, float(pred)))

    labels = ["DOWN", "NEUTRAL", "UP"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_trends,
        pred_trends,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    trend_accuracy = float(np.mean(np.array(actual_trends) == np.array(pred_trends)) * 100)

    return {
        "samples": int(len(actual_close)),
        "regression": {
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "mape_percent": round(mape, 4),
        },
        "trend_quality": {
            "accuracy_percent": round(trend_accuracy, 4),
            "macro_precision": round(float(precision), 6),
            "macro_recall": round(float(recall), 6),
            "macro_f1": round(float(f1), 6),
            "labels": labels,
        },
    }


def metrics_path(symbol: str, profile: str = "simple") -> str:
    suffix = "" if (profile or "simple").lower() == "simple" else f"_{(profile or 'simple').lower()}"
    return os.path.join(symbol_model_dir(symbol), f"metrics{suffix}.json")


def save_metrics(symbol: str, payload: Dict, profile: str = "simple"):
    with open(metrics_path(symbol, profile=profile), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
