"""
ViziGenesis unified launcher.

Examples:
  python run.py
  python run.py --mode serve --port 8000
  python run.py --mode train --symbol AAPL --epoch 20
  python run.py --mode evaluate --symbol AAPL --split test
    python run.py --mode vizi-train --run-name vizi-o1 --vizi-epochs 30
  python run.py --mode vizi-test
"""
import argparse
import os
import subprocess
import sys

import uvicorn


def _load_env_file(env_path: str) -> None:
    """Load simple KEY=VALUE pairs from .env into process env."""
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
        pass


def run_backend_cli(mode: str, symbol: str, period: str, epoch: int, split: str, ai_mode: str):
    """Delegate non-serve actions to backend CLI."""
    cmd = [sys.executable, "-m", "backend.cli", mode, "--symbol", symbol]
    cmd.extend(["--ai-mode", ai_mode])

    if mode in {"train", "evaluate", "validate", "test", "check-accuracy"}:
        if period:
            cmd.extend(["--period", period])

    if mode == "train":
        cmd.extend(["--epochs", str(epoch)])
    elif mode == "evaluate":
        cmd.extend(["--split", split])

    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViziGenesis launcher")
    parser.add_argument(
        "--mode",
        default="serve",
        choices=["serve", "train", "evaluate", "validate", "test", "check-accuracy", "package",
                 "vizi-train", "vizi-test", "vizi-profile", "vizi-evaluate"],
        help="Execution mode",
    )
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol, e.g. AAPL")
    parser.add_argument("--period", default=None, help="Yahoo Finance period, e.g. 1y/2y/5y/10y")
    parser.add_argument("--ai-mode", default="simple", choices=["simple", "pro"], help="Model profile: simple or pro")
    parser.add_argument("--epoch", "--epochs", dest="epoch", type=int, default=50,
                        help="Epochs for train mode")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Split for evaluate mode")

    parser.add_argument("--host", default="0.0.0.0", help="Host for serve mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for serve mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for serve mode")
    parser.add_argument(
        "--fred-api-key",
        default=None,
        help="FRED API key (sets FRED_API_KEY environment variable for this run)",
    )

    # vizi-ai arguments
    parser.add_argument("--run-name", default="vizi-o1", help="Run name for vizi-train versioning")
    parser.add_argument("--vizi-epochs", type=int, default=30, help="Max epochs for vizi-train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for vizi-train")
    parser.add_argument("--specialist-warmup", action="store_true",
                        help="Enable per-stock specialist warm-up before vizi-train")
    parser.add_argument("--run-dir", default=None, help="Run directory for vizi-evaluate")

    return parser


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    _load_env_file(os.path.join(project_root, ".env"))

    args = build_parser().parse_args()

    if args.fred_api_key:
        os.environ["FRED_API_KEY"] = args.fred_api_key

    if args.mode == "serve":
        print("⚡ ViziGenesis starting web app …")
        print(f"   Open http://localhost:{args.port} in your browser.\n")
        uvicorn.run(
            "backend.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
        return

    # ── Vizi-AI multi-modal modes ──
    if args.mode.startswith("vizi-"):
        from backend.vizi_ai import orchestrator
        vizi_cmd = args.mode.replace("vizi-", "")
        # Build a namespace matching the orchestrator's expectations
        vizi_args = argparse.Namespace(
            command=vizi_cmd,
            run_name=args.run_name,
            epochs=args.vizi_epochs,
            batch_size=args.batch_size,
            lr=3e-4,
            d_model=None,
            n_layers=None,
            val_every=500,
            patience=5,
            specialist_warmup=args.specialist_warmup,
            skip_collect=True,
            section=None,
            run_dir=args.run_dir or "",
            split=args.split,
        )
        cmd_map = {
            "train": orchestrator.cmd_train,
            "test": orchestrator.cmd_test,
            "profile": orchestrator.cmd_profile,
            "evaluate": orchestrator.cmd_evaluate,
        }
        fn = cmd_map.get(vizi_cmd)
        if fn:
            fn(vizi_args)
        else:
            print(f"Unknown vizi command: {vizi_cmd}")
        return

    print(f"⚡ Running mode: {args.mode} | symbol={args.symbol} | ai-mode={args.ai_mode} | period={args.period or 'default'}")
    run_backend_cli(
        mode=args.mode,
        symbol=args.symbol,
        period=args.period,
        epoch=args.epoch,
        split=args.split,
        ai_mode=args.ai_mode,
    )


if __name__ == "__main__":
    main()
