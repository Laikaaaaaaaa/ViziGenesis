"""
ViziGenesis unified launcher.

Examples:
  python run.py
  python run.py --mode serve --port 8000
  python run.py --mode train --symbol AAPL --epoch 20
  python run.py --mode evaluate --symbol AAPL --split test
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
        choices=["serve", "train", "evaluate", "validate", "test", "check-accuracy", "package"],
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
