#!/usr/bin/env python
"""evaluate.py — run the evaluation harness on a trained checkpoint across all five
datasets (FF++ in-domain, Celeb-DF v2, DFDC, WildDeepfake, DF40 per subset).

Produces the per-dataset frame- and video-level AUC/acc/EER table (AUC anchor),
with the video reduction explicit and leakage/sanity checks attached. Writes
<out_dir>/<stem>.json and .csv and prints the table.

RUN (on the box with the crops, the FF++ split JSONs, and the checkpoint):
    python scripts/evaluate.py --checkpoint experiments/results/<run>/best.pt \
        --model-config configs/model/spatial_only.yaml
    python scripts/evaluate.py --checkpoint <ckpt> --video-reduce median --stem e1_spatial
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in (overlay or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained detector across all five datasets.")
    ap.add_argument("--checkpoint", required=True, help="path to best.pt")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "model" / "spatial_only.yaml"))
    ap.add_argument("--paths", default=str(REPO_ROOT / "paths.yaml"))
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="default: <checkpoint dir>")
    ap.add_argument("--stem", default="eval", help="output filename stem")
    ap.add_argument("--video-reduce", default=None, choices=["mean", "median"],
                    help="override eval.video_reduce")
    ap.add_argument("--threshold", type=float, default=0.5, help="threshold for accuracy")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="fp16 inference (default: fp32 for stable numbers)")
    ap.add_argument("--max-batches", type=int, default=None, help="debug: cap batches per loader")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-indomain", action="store_true", help="skip the FF++ in-domain test row")
    args = ap.parse_args()

    import torch
    from sfdet.data.dataset import get_dataloaders, load_ffpp_split_ids, _default_splits_dir
    from sfdet.engine.evaluator import evaluate_all, format_table, save_results
    from sfdet.metrics.report import emit_results_table
    from sfdet.models.spatial_branch import build_spatial_classifier

    cfg = deep_merge(yaml.safe_load(Path(args.config).read_text()) or {},
                     yaml.safe_load(Path(args.model_config).read_text()) or {})
    paths = yaml.safe_load(Path(args.paths).read_text()) or {}
    video_reduce = args.video_reduce or cfg.get("eval", {}).get("video_reduce", "mean")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build the model from the checkpoint's own config when present (guarantees the
    # architecture matches), with pretrained=False so eval doesn't re-download weights.
    ckpt = torch.load(args.checkpoint, map_location=device)
    build_cfg = copy.deepcopy(ckpt.get("cfg", cfg))
    build_cfg.setdefault("model", {})["pretrained"] = False
    model = build_spatial_classifier(build_cfg).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"val AUC {ckpt.get('best', float('nan')):.4f})")

    _, _, test_loaders = get_dataloaders(
        cfg, paths, splits_dir=args.splits_dir, num_workers=args.num_workers,
        include_indomain_test=not args.no_indomain, verbose=True)
    if not test_loaders:
        print("no test loaders built — check manifests / paths.yaml")
        return 2

    # training-split identities for the FF++ in-domain identity-overlap check
    train_origin_ids = None
    try:
        sdir = args.splits_dir or _default_splits_dir(paths)
        train_origin_ids = load_ffpp_split_ids(sdir, "train")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not load FF++ train split ids for the leakage check ({e})")

    out = evaluate_all(model, test_loaders, device, video_reduce=video_reduce,
                       threshold=args.threshold, train_origin_ids=train_origin_ids,
                       train_dataset=cfg.get("train", {}).get("dataset", "faceforensics_c23"),
                       amp=args.amp, max_batches=args.max_batches, verbose=True)

    print("\n" + format_table(out))
    out_dir = args.out_dir or str(Path(args.checkpoint).resolve().parent)
    saved = save_results(out, out_dir, stem=args.stem)
    md_path = Path(out_dir) / f"{args.stem}.md"
    md_path.write_text(emit_results_table(out))
    print(f"\nsaved: {saved['json']}\n       {saved['csv']}\n       {md_path}  (\u00a72.9 paste-ready)")
    if not out["sanity"]["ok"]:
        print("\n[!] sanity flags raised — inspect identity overlap / single-class / mixed-label above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
