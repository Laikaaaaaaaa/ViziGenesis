"""
CLI for Yahoo Finance training/evaluation/testing and artifact packaging.

Usage examples:
  python -m backend.cli train --symbol AAPL --epochs 30
  python -m backend.cli evaluate --symbol AAPL --split test
  python -m backend.cli validate --symbol AAPL
  python -m backend.cli test --symbol AAPL
  python -m backend.cli check-accuracy --symbol AAPL
  python -m backend.cli package --symbol AAPL
"""
import argparse
import json
import os
import zipfile

from backend.config import DEVICE, EPOCHS, MODEL_DIR
from backend.data_utils import (
    get_historical_data_with_fallback,
    normalize_ai_mode,
    get_mode_config,
    get_feature_columns,
    prepare_feature_dataframe,
)
from backend.model import (
    train_model,
    predict,
    save_scaler,
    load_scaler,
    save_meta,
    load_trained_model,
    get_symbol_artifact_paths,
    symbol_model_dir,
    migrate_legacy_artifacts,
)
from backend.pipeline import (
    build_train_val_test_split,
    inverse_close,
    evaluate_predictions,
    save_metrics,
    metrics_path,
)


def _get_split_arrays(split_data, split: str):
    split = split.lower()
    if split == "train":
        return split_data.X_train, split_data.y_train, split_data.idx_train
    if split == "val":
        return split_data.X_val, split_data.y_val, split_data.idx_val
    if split == "test":
        return split_data.X_test, split_data.y_test, split_data.idx_test
    raise ValueError("split must be one of train|val|test")


def cmd_train(args):
    symbol = args.symbol.upper()
    ai_mode = normalize_ai_mode(args.ai_mode)
    mode_cfg = get_mode_config(ai_mode)
    period = args.period or mode_cfg["period"]
    feature_cols = get_feature_columns(ai_mode)

    df_raw, data_source = get_historical_data_with_fallback(symbol, period=period)
    df = prepare_feature_dataframe(df_raw, ai_mode)
    split_data = build_train_val_test_split(df, feature_cols=feature_cols)

    model, history = train_model(
        split_data.X_train,
        split_data.y_train,
        split_data.X_val,
        split_data.y_val,
        symbol=symbol,
        profile=ai_mode,
        epochs=args.epochs,
    )
    save_scaler(split_data.scaler, symbol, profile=ai_mode)

    # evaluate on val/test after training
    metrics_all = {}
    for split in ["val", "test"]:
        X, y, idx = _get_split_arrays(split_data, split)
        pred_scaled = predict(model, X)
        pred_close = inverse_close(split_data.scaler, pred_scaled, feature_cols=feature_cols)
        actual_close = inverse_close(split_data.scaler, y, feature_cols=feature_cols)
        metrics_all[split] = evaluate_predictions(df, idx, pred_close, actual_close)

    payload = {
        "symbol": symbol,
        "mode": ai_mode,
        "period": period,
        "device": str(DEVICE),
        "data_source": data_source,
        "epochs_run": len(history),
        "history": history,
        "metrics": metrics_all,
    }
    save_metrics(symbol, payload, profile=ai_mode)
    save_meta(symbol, {
        "mode": ai_mode,
        "training_period": period,
        "device": str(DEVICE),
        "epochs_run": len(history),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "input_size": len(feature_cols),
        "features": feature_cols,
        "artifact": {
            "model": f"models/{symbol}/model{'' if ai_mode == 'simple' else '_' + ai_mode}.pt",
            "scaler": f"models/{symbol}/scaler{'' if ai_mode == 'simple' else '_' + ai_mode}.pkl",
            "metrics": f"models/{symbol}/metrics{'' if ai_mode == 'simple' else '_' + ai_mode}.json",
        },
    }, profile=ai_mode)

    print(json.dumps({
        "status": "trained",
        "symbol": symbol,
        "mode": ai_mode,
        "data_source": data_source,
        "device": str(DEVICE),
        "epochs": len(history),
        "val": metrics_all["val"],
        "test": metrics_all["test"],
        "metrics_file": metrics_path(symbol, profile=ai_mode),
    }, indent=2, ensure_ascii=False))


def cmd_evaluate(args):
    symbol = args.symbol.upper()
    ai_mode = normalize_ai_mode(args.ai_mode)
    mode_cfg = get_mode_config(ai_mode)
    period = args.period or mode_cfg["period"]
    feature_cols = get_feature_columns(ai_mode)

    model = load_trained_model(symbol, profile=ai_mode)
    scaler = load_scaler(symbol, profile=ai_mode)
    if model is None or scaler is None:
        raise RuntimeError(f"Model/scaler mode={ai_mode} chưa có. Hãy chạy lệnh train trước.")

    df_raw, data_source = get_historical_data_with_fallback(symbol, period=period)
    df = prepare_feature_dataframe(df_raw, ai_mode)
    split_data = build_train_val_test_split(df, feature_cols=feature_cols)
    X, y, idx = _get_split_arrays(split_data, args.split)

    pred_scaled = predict(model, X)
    pred_close = inverse_close(scaler, pred_scaled, feature_cols=feature_cols)
    actual_close = inverse_close(scaler, y, feature_cols=feature_cols)
    report = evaluate_predictions(df, idx, pred_close, actual_close)

    print(json.dumps({
        "symbol": symbol,
        "mode": ai_mode,
        "data_source": data_source,
        "period": period,
        "split": args.split,
        "device": str(DEVICE),
        "report": report,
    }, indent=2, ensure_ascii=False))


def cmd_check_accuracy(args):
    symbol = args.symbol.upper()
    ai_mode = normalize_ai_mode(args.ai_mode)
    mode_cfg = get_mode_config(ai_mode)
    period = args.period or mode_cfg["period"]
    feature_cols = get_feature_columns(ai_mode)

    model = load_trained_model(symbol, profile=ai_mode)
    scaler = load_scaler(symbol, profile=ai_mode)
    if model is None or scaler is None:
        raise RuntimeError(f"Model/scaler mode={ai_mode} chưa có. Hãy chạy lệnh train trước.")

    df_raw, data_source = get_historical_data_with_fallback(symbol, period=period)
    df = prepare_feature_dataframe(df_raw, ai_mode)
    split_data = build_train_val_test_split(df, feature_cols=feature_cols)

    result = {}
    for split in ["train", "val", "test"]:
        X, y, idx = _get_split_arrays(split_data, split)
        pred_scaled = predict(model, X)
        pred_close = inverse_close(scaler, pred_scaled, feature_cols=feature_cols)
        actual_close = inverse_close(scaler, y, feature_cols=feature_cols)
        result[split] = evaluate_predictions(df, idx, pred_close, actual_close)

    print(json.dumps({
        "symbol": symbol,
        "mode": ai_mode,
        "data_source": data_source,
        "period": period,
        "device": str(DEVICE),
        "accuracy_loss_trend_quality": result,
    }, indent=2, ensure_ascii=False))


def cmd_package(args):
    symbol = args.symbol.upper()
    ai_mode = normalize_ai_mode(args.ai_mode)
    paths = get_symbol_artifact_paths(symbol, profile=ai_mode)
    files = [
        paths.get("active_model"),
        paths.get("active_scaler"),
        paths.get("active_meta"),
        paths.get("active_metrics"),
    ]
    files = [f for f in files if f and os.path.exists(f)]

    # Legacy fallback for older flat files
    if not files:
        pt_file = os.path.join(MODEL_DIR, f"{symbol}_lstm.pt")
        pkl_file = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
        meta_file = os.path.join(MODEL_DIR, f"{symbol}_meta.json")
        m_file = os.path.join(MODEL_DIR, f"{symbol}_metrics.json")
        files = [f for f in [pt_file, pkl_file, meta_file, m_file] if os.path.exists(f)]

    if not files:
        raise RuntimeError("Không tìm thấy artifact. Hãy train model trước.")

    zip_suffix = "" if ai_mode == "simple" else f"_{ai_mode}"
    zip_path = os.path.join(symbol_model_dir(symbol), f"{symbol}_artifact{zip_suffix}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            zf.write(file, arcname=os.path.basename(file))

    print(json.dumps({
        "status": "packaged",
        "symbol": symbol,
        "mode": ai_mode,
        "artifact_zip": zip_path,
        "included_files": [os.path.basename(f) for f in files],
        "formats": [".pt", ".pkl", ".json"],
        "note": "Dự án đang dùng PyTorch nên artifact model chính là .pt (không tạo .h5).",
    }, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="ViziGenesis CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train model từ Yahoo Finance")
    p_train.add_argument("--symbol", required=True)
    p_train.add_argument("--period", default=None)
    p_train.add_argument("--ai-mode", default="simple", choices=["simple", "pro"])
    p_train.add_argument("--epochs", type=int, default=EPOCHS)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Đánh giá model theo split")
    p_eval.add_argument("--symbol", required=True)
    p_eval.add_argument("--period", default=None)
    p_eval.add_argument("--ai-mode", default="simple", choices=["simple", "pro"])
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.set_defaults(func=cmd_evaluate)

    p_validate = sub.add_parser("validate", help="Đánh giá trên validation set")
    p_validate.add_argument("--symbol", required=True)
    p_validate.add_argument("--period", default=None)
    p_validate.add_argument("--ai-mode", default="simple", choices=["simple", "pro"])
    p_validate.set_defaults(func=lambda a: cmd_evaluate(argparse.Namespace(**{**vars(a), "split": "val"})))

    p_test = sub.add_parser("test", help="Đánh giá trên test set")
    p_test.add_argument("--symbol", required=True)
    p_test.add_argument("--period", default=None)
    p_test.add_argument("--ai-mode", default="simple", choices=["simple", "pro"])
    p_test.set_defaults(func=lambda a: cmd_evaluate(argparse.Namespace(**{**vars(a), "split": "test"})))

    p_chk = sub.add_parser("check-accuracy", help="Check loss/accuracy/trend quality")
    p_chk.add_argument("--symbol", required=True)
    p_chk.add_argument("--period", default=None)
    p_chk.add_argument("--ai-mode", default="simple", choices=["simple", "pro"])
    p_chk.set_defaults(func=cmd_check_accuracy)

    p_pkg = sub.add_parser("package", help="Package artifact model")
    p_pkg.add_argument("--symbol", required=True)
    p_pkg.add_argument("--ai-mode", default="simple", choices=["simple", "pro"])
    p_pkg.set_defaults(func=cmd_package)

    return parser


def main():
    migrate_legacy_artifacts()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
