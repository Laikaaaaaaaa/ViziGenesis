"""
ViziGenesis vizi-o1 — Orchestrator & CLI
==========================================
Single entry-point for the complete multi-modal training pipeline.

Usage:
    # Full pipeline: collect → preprocess → train → evaluate
    python -m backend.vizi_ai.orchestrator run

    # Just train (data already collected)
    python -m backend.vizi_ai.orchestrator train --run-name vizi-o1

    # Evaluate an existing run
    python -m backend.vizi_ai.orchestrator evaluate --run-dir models/vizi-o1_20260308_120000

    # Per-stock specialists + full training
    python -m backend.vizi_ai.orchestrator train --specialist-warmup

    # Quick test (5 stocks, 2 epochs)
    python -m backend.vizi_ai.orchestrator test

    # Collect additional data
    python -m backend.vizi_ai.orchestrator collect

    # Profile model (count params, VRAM estimate)
    python -m backend.vizi_ai.orchestrator profile

Versioning scheme:
    models/
    ├── vizi-o1_20260308_120000/      ← run name + timestamp
    │   ├── config.json               ← full configuration
    │   ├── train_history.json        ← per-step training metrics
    │   ├── val_history.json          ← validation metrics
    │   ├── eval_test.json            ← test set evaluation report
    │   ├── final/                    ← final checkpoint
    │   │   └── checkpoint.pt
    │   ├── best/                     ← best validation checkpoint
    │   │   └── checkpoint.pt
    │   └── step_001000/              ← periodic checkpoints
    │       └── checkpoint.pt
    ├── vizi-1_20260309_...           ← next iteration
    └── vizi-b1_...                   ← B200 scaling run
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vizi_ai")


def cmd_profile(args):
    """Profile the model: parameter count, VRAM estimate, architecture."""
    from backend.vizi_ai.model import ModelConfig, ViziMarketTransformer

    cfg = ModelConfig()
    if args.d_model:
        cfg.d_model = args.d_model
    if args.n_layers:
        cfg.n_layers = args.n_layers

    model = ViziMarketTransformer(cfg)
    print(model.summary())

    # VRAM estimate
    total_params = model.count_parameters()
    # fp16 params + optimizer states (Adam: 2× model) + gradients
    fp16_mb = total_params * 2 / 1e6
    adam_states_mb = total_params * 4 * 2 / 1e6  # fp32 for m and v
    grad_mb = total_params * 2 / 1e6
    # Activations (rough: batch_size × seq_len × d_model × n_layers × 4 bytes)
    act_mb = args.batch_size * 200 * cfg.d_model * (cfg.n_layers + cfg.fusion_layers) * 4 / 1e6

    total_vram = fp16_mb + adam_states_mb + grad_mb + act_mb
    print(f"\nVRAM Estimate (batch_size={args.batch_size}):")
    print(f"  Model (fp16):    {fp16_mb:>8.1f} MB")
    print(f"  Adam states:     {adam_states_mb:>8.1f} MB")
    print(f"  Gradients:       {grad_mb:>8.1f} MB")
    print(f"  Activations:     {act_mb:>8.1f} MB")
    print(f"  ─────────────────────────────")
    print(f"  Total:           {total_vram:>8.1f} MB")
    print(f"\n  RTX 4090 (24GB): {'✓ FITS' if total_vram < 22000 else '✗ TOO LARGE — reduce batch/model'}")
    print(f"  B200 (80GB):     {'✓ FITS' if total_vram < 75000 else '✗ TOO LARGE'}")


def cmd_collect(args):
    """Collect additional financial data."""
    logger.info("Running data collection...")
    script = ROOT / "data" / "collect_all.py"
    if args.section:
        subprocess.run([sys.executable, str(script), "--section", args.section], check=True)
    else:
        subprocess.run([sys.executable, str(script)], check=True)

    # Preprocess
    logger.info("Running preprocessing...")
    subprocess.run([sys.executable, str(ROOT / "data" / "preprocess.py")], check=True)

    # Generate instructions
    logger.info("Generating instruction data...")
    subprocess.run([sys.executable, str(ROOT / "data" / "generate_instructions.py")], check=True)


def cmd_train(args):
    """Full multi-modal training pipeline."""
    from backend.vizi_ai.model import ModelConfig
    from backend.vizi_ai.data_pipeline import DataConfig, discover_symbols
    from backend.vizi_ai.trainer import TrainConfig, ViziTrainer, train_per_stock_specialists

    # ── Configure ──
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        run_name=args.run_name,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_every_steps=args.val_every,
        patience=args.patience,
    )

    if args.d_model:
        model_cfg.d_model = args.d_model
        model_cfg.d_ff = args.d_model * 4
    if args.n_layers:
        model_cfg.n_layers = args.n_layers

    logger.info("═" * 60)
    logger.info("ViziGenesis Multi-Modal Training — %s", train_cfg.run_name)
    logger.info("═" * 60)

    # ── Phase 1: Per-stock specialist warm-up (optional) ──
    trainer = ViziTrainer(model_cfg, data_cfg, train_cfg)

    if args.specialist_warmup:
        logger.info("\n── Phase 1: Per-stock Specialist Warm-up (parallel) ──")
        train_per_stock_specialists(
            trainer.symbols[:50],  # top 50 by data size
            trainer.model,
            data_cfg,
            train_cfg,
            n_epochs=2,
            num_parallel_stocks=getattr(args, "parallel_stocks", 0),
        )

    # ── Phase 2: Full-universe training ──
    logger.info("\n── Phase 2: Full-Universe Multi-Modal Training ──")
    trainer.train()

    logger.info("\n═══ Training pipeline complete ═══")
    logger.info("Run directory: %s", train_cfg.run_dir)


def cmd_evaluate(args):
    """Evaluate a trained model."""
    from backend.vizi_ai.model import ModelConfig, ViziMarketTransformer
    from backend.vizi_ai.data_pipeline import DataConfig
    from backend.vizi_ai.trainer import TrainConfig, ViziTrainer

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        return

    # Load config
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            saved = json.load(f)
        model_cfg = ModelConfig(**saved.get("model", {}))
    else:
        model_cfg = ModelConfig()

    data_cfg = DataConfig()
    train_cfg = TrainConfig(run_name="eval")

    trainer = ViziTrainer(model_cfg, data_cfg, train_cfg)

    # Load best checkpoint
    best_dir = run_dir / "best"
    if best_dir.exists():
        trainer._load_checkpoint(best_dir)
    elif (run_dir / "final").exists():
        trainer._load_checkpoint(run_dir / "final")
    else:
        logger.error("No checkpoint found in %s", run_dir)
        return

    report = trainer.evaluate(args.split, run_dir)
    print(json.dumps(report["aggregate"], indent=2))


def cmd_test(args):
    """Quick smoke test with minimal data."""
    from backend.vizi_ai.model import ModelConfig, ViziMarketTransformer
    from backend.vizi_ai.data_pipeline import DataConfig, discover_symbols, create_dataloader

    logger.info("── Quick Smoke Test ──")

    data_cfg = DataConfig()
    symbols = discover_symbols(data_cfg.data_root)[:5]  # 5 stocks max
    if not symbols:
        logger.error("No stock data found")
        return

    logger.info("Testing with %d symbols: %s", len(symbols), symbols)

    # Create a tiny dataloader
    loader = create_dataloader(symbols, data_cfg, "train", batch_size=4, shuffle=True)

    # Get one batch to probe dimensions
    batch = None
    for b in loader:
        batch = b
        break

    if batch is None:
        logger.error("No data produced by pipeline")
        return

    logger.info("Batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info("  %-15s: %s", k, tuple(v.shape))
        elif isinstance(v, dict):
            for kk, vv in v.items():
                logger.info("  targets/%-8s: %s", kk, tuple(vv.shape))

    # Create model with probed dimensions
    model_cfg = ModelConfig(
        n_price_features=batch["price_seq"].shape[-1],
        n_macro_features=batch["macro_seq"].shape[-1],
        n_market_features=batch["market_seq"].shape[-1],
        n_fundamental_features=batch["fundamental"].shape[-1],
        n_stocks=len(symbols),
        d_model=128,   # smaller for test
        n_layers=2,
        fusion_layers=2,
        d_ff=512,
    )

    model = ViziMarketTransformer(model_cfg)
    logger.info("\n%s", model.summary())

    # Forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.to(device)
        elif isinstance(v, dict):
            batch_dev[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            batch_dev[k] = v

    model.eval()
    with torch.no_grad():
        preds = model(batch_dev)

    logger.info("\nPrediction shapes:")
    for k, v in preds.items():
        sample_val = v[0, 0].item() if v.dim() == 2 and v.size(1) == 1 else v[0].argmax().item() if v.dim() == 2 else v[0].item()
        logger.info("  %-12s: %s  (sample: %.4f)", k, tuple(v.shape), sample_val)

    logger.info("\n✓ Smoke test passed — pipeline ↔ model ↔ predictions OK")

    # VRAM check
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        logger.info("GPU Memory: %.1f MB allocated, %.1f MB reserved", allocated, reserved)


def cmd_run(args):
    """Full pipeline: collect → preprocess → train → evaluate."""
    if args.skip_collect:
        logger.info("Skipping data collection (--skip-collect)")
    else:
        cmd_collect(args)

    cmd_train(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="ViziGenesis vizi-o1 — Multi-Modal Financial AI Orchestrator"
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── run ──
    p_run = sub.add_parser("run", help="Full pipeline: collect → train → evaluate")
    p_run.add_argument("--skip-collect", action="store_true", help="Skip data collection")
    _add_train_args(p_run)
    _add_collect_args(p_run)

    # ── train ──
    p_train = sub.add_parser("train", help="Train the multi-modal model")
    _add_train_args(p_train)

    # ── evaluate ──
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--run-dir", required=True, help="Path to run directory")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])

    # ── test ──
    p_test = sub.add_parser("test", help="Quick smoke test")

    # ── profile ──
    p_prof = sub.add_parser("profile", help="Profile model size & VRAM")
    p_prof.add_argument("--d-model", type=int, default=None)
    p_prof.add_argument("--n-layers", type=int, default=None)
    p_prof.add_argument("--batch-size", type=int, default=64)

    # ── collect ──
    p_coll = sub.add_parser("collect", help="Collect financial data")
    _add_collect_args(p_coll)

    return parser


def _add_train_args(parser):
    parser.add_argument("--run-name", default="vizi-o1", help="Run name for versioning")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=None, help="Encoder depth")
    parser.add_argument("--val-every", type=int, default=500, help="Validate every N steps")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--specialist-warmup", action="store_true",
                        help="Run per-stock specialist warm-up before full training")
    parser.add_argument("--parallel-stocks", type=int, default=0,
                        help="Stocks per parallel mega-batch in Phase 1 (0=auto-detect from VRAM)")


def _add_collect_args(parser):
    parser.add_argument("--section", default=None,
                        help="Collect specific section: macro|markets|stocks|fundamentals|sec|news|central_bank")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cmds = {
        "run":      cmd_run,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "test":     cmd_test,
        "profile":  cmd_profile,
        "collect":  cmd_collect,
    }

    cmd_fn = cmds.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
