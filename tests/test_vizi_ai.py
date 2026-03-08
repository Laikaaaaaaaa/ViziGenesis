"""Quick smoke test for the vizi_ai multi-modal pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from backend.vizi_ai.data_pipeline import DataConfig, discover_symbols, create_dataloader
from backend.vizi_ai.model import ModelConfig, ViziMarketTransformer
from backend.vizi_ai.trainer import _compute_losses, UncertaintyMultiTaskLoss

def main():
    cfg = DataConfig()
    symbols = discover_symbols(cfg.data_root)[:5]
    print(f"Symbols: {len(symbols)} → {symbols}")

    loader = create_dataloader(symbols, cfg, "train", batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  targets/{kk}: {tuple(vv.shape)} {vv.dtype}")

    model_cfg = ModelConfig(
        n_price_features=batch["price_seq"].shape[-1],
        n_macro_features=batch["macro_seq"].shape[-1],
        n_market_features=batch["market_seq"].shape[-1],
        n_fundamental_features=batch["fundamental"].shape[-1],
        n_stocks=len(symbols),
        d_model=128, n_layers=2, fusion_layers=2, d_ff=512,
    )
    model = ViziMarketTransformer(model_cfg)
    print(f"\n{model.summary()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.to(device).float() if v.is_floating_point() else v.to(device)
        elif isinstance(v, dict):
            batch_dev[k] = {
                kk: vv.to(device).float() if vv.is_floating_point() else vv.to(device)
                for kk, vv in v.items()
            }

    model.eval()
    with torch.no_grad():
        preds = model(batch_dev)

    print("\nPredictions:")
    for k, v in preds.items():
        print(f"  {k}: {tuple(v.shape)}")

    # Test loss
    losses = _compute_losses(preds, batch_dev["targets"])
    print(f"\nTask losses ({len(losses)}):")
    for i, l in enumerate(losses):
        print(f"  task {i}: {l.item():.4f}")

    mt = UncertaintyMultiTaskLoss(5).to(device)
    total, info = mt(losses)
    print(f"\nTotal weighted loss: {total.item():.4f}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.memory_allocated()/1e6:.1f} MB")

    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    main()
