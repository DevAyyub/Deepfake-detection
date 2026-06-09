#!/usr/bin/env python
"""train.py — fit the spatial-only model (E1's spatial-only arm) on FF++ c23.

Merges base.yaml with a model config (configs/model/spatial_only.yaml), builds the
train/val loaders via get_dataloaders, builds the model via build_spatial_classifier,
and runs sfdet.engine.trainer.fit. Produces a real checkpoint (run_dir/best.pt)
selected on validation AUC-ROC.

RUN (on the box with a GPU, the crops, and the FF++ split JSONs):
    python scripts/train.py --model-config configs/model/spatial_only.yaml
    python scripts/train.py --model-config configs/model/spatial_only.yaml --epochs 20 --run-name e1_spatial
    python scripts/train.py ... --max-train-batches 5 --max-val-batches 5   # quick wiring check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay onto base (overlay wins); returns a new dict."""
    out = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config(base_path, model_path, data_path=None) -> dict:
    cfg = yaml.safe_load(Path(base_path).read_text()) or {}
    for overlay_path in (data_path, model_path):       # base <- data <- model
        if overlay_path:
            cfg = deep_merge(cfg, yaml.safe_load(Path(overlay_path).read_text()) or {})
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the spatial-only deepfake detector.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "model" / "spatial_only.yaml"))
    ap.add_argument("--data-config", default=None, help="optional configs/data/<ds>.yaml overlay")
    ap.add_argument("--paths", default=str(REPO_ROOT / "paths.yaml"))
    ap.add_argument("--splits-dir", default=None, help="FF++ splits dir (else paths.yaml/<ff_root>/splits)")
    ap.add_argument("--out-dir", default=None, help="override logging.out_dir")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--balance", default="method", choices=["method", "binary", "none"])
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    # Heavy imports deferred so --help / config-merge don't require torch.
    import torch
    from sfdet.data.dataset import get_dataloaders, load_configs  # noqa: F401
    from sfdet.engine.trainer import fit, set_seed
    from sfdet.models.factory import build_model

    cfg = load_merged_config(args.config, args.model_config, args.data_config)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    paths = yaml.safe_load(Path(args.paths).read_text()) or {}

    set_seed(int(cfg.get("seed", 1337)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    balance = None if args.balance == "none" else args.balance
    train_loader, val_loader, _ = get_dataloaders(
        cfg, paths, splits_dir=args.splits_dir, num_workers=args.num_workers,
        balance=balance, verbose=True)

    model, forward_fn, variant = build_model(cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: variant={variant} backbone={cfg.get('model', {}).get('backbone', '-')} "
          f"| trainable params {n_params/1e6:.1f}M")

    result = fit(model, train_loader, val_loader, cfg, device,
                 out_dir=args.out_dir, run_name=args.run_name, resume=args.resume,
                 max_train_batches=args.max_train_batches, max_val_batches=args.max_val_batches,
                 forward_fn=forward_fn)
    print(f"\nbest checkpoint: {result['best_path']}  (val frame AUC {result['best_auc']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
