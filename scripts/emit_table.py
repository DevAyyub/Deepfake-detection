#!/usr/bin/env python
"""emit_table.py — render the §2.9 paste-ready results table from an eval.json.

Lets you (re)produce the labeled table from a previous evaluation run without
re-evaluating. Torch-free.

    python scripts/emit_table.py --eval-json experiments/results/<run>/eval.json
    python scripts/emit_table.py --eval-json <path> --format latex --percent --out table.tex
    python scripts/emit_table.py --eval-json <path> --compression dfdc=DFDC-full celebdf_v2=CDF-v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfdet.metrics.report import emit_results_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the §2.9 results table from eval.json.")
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--format", default="md", choices=["md", "latex"])
    ap.add_argument("--percent", action="store_true", help="report metrics x100 instead of [0,1]")
    ap.add_argument("--title", default="Spatial-only (E1) — cross-dataset results")
    ap.add_argument("--train-protocol", default="FaceForensics++ c23, four-manipulation")
    ap.add_argument("--compression", nargs="*", default=[],
                    help="per-dataset label overrides, e.g. dfdc=DFDC-full")
    ap.add_argument("--out", default=None, help="write to this file (default: stdout)")
    args = ap.parse_args()

    eval_out = json.loads(Path(args.eval_json).read_text())
    comp_map = {}
    for kv in args.compression:
        k, _, v = kv.partition("=")
        if k and v:
            comp_map[k] = v

    text = emit_results_table(eval_out, fmt=args.format, compression_map=comp_map or None,
                              percent=args.percent, title=args.title,
                              train_protocol=args.train_protocol)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
