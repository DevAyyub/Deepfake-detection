#!/usr/bin/env python3
"""scripts/preprocess_faces.py — config-driven face extraction for the 3 video datasets.

Thin wrapper over ``sfdet.preprocess.extract_faces.run_extraction`` that resolves
the dataset root from ``paths.yaml`` and the extraction settings from
``configs/base.yaml``'s ``preprocess:`` block, so a run is just:

    python scripts/preprocess_faces.py --dataset faceforensics_c23

CLI flags override config values, e.g.:
    python scripts/preprocess_faces.py --dataset celebdf_v2 --limit 5      # smoke test
    python scripts/preprocess_faces.py --dataset faceforensics_c23 --no-align   # ABL: unaligned
    python scripts/preprocess_faces.py --dataset faceforensics_c23 --border reflect  # ABL: FFT-hygiene

Add ``--build-manifest`` to also write the unified CSV in the same run; otherwise
the next manifest command is printed for you.

Only the three VIDEO datasets are extractable here (FF++ c23, Celeb-DF v2, DFDC).
WildDeepfake and DF40 ship pre-cropped — they skip extraction and go straight to
``python -m sfdet.preprocess.manifest --dataset <name> --data-root <root>``.

Runs in the ISOLATED preprocess env (requirements-preprocess.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the src-layout package importable without requiring `pip install -e .`
# in the (isolated) preprocess env.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Hardcoded here (rather than imported from extract_faces) so the wrapper can be
# imported/inspected without dlib present; the heavy import is deferred to main().
VIDEO_DATASETS = ("faceforensics_c23", "celebdf_v2", "dfdc")

# base.yaml uses generic names ("linear"); extract_faces' INTERP keys are cv2-style.
INTERP_ALIAS = {
    "linear": "bilinear", "bilinear": "bilinear",
    "cubic": "bicubic", "bicubic": "bicubic",
    "area": "area", "nearest": "nearest",
}


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def cfg_kwargs_from_preprocess(pp: dict) -> dict:
    """Map configs/base.yaml `preprocess:` keys onto ExtractConfig kwargs.

    Note the renames: res->crop_size, interpolation->interp (with cv2-name
    aliasing), border_mode->border, sampling_mode->mode. base.yaml exposes a
    single `scale` knob; it feeds both the aligned-template margin (scale) and the
    unaligned-bbox expansion (enlarge). `detector` and `min_face_size` from the
    config are not ExtractConfig fields (extraction is dlib-only; size-filtering
    is not wired) and are intentionally ignored here.
    """
    interp_raw = str(pp.get("interpolation", "bilinear")).lower()
    scale = float(pp.get("scale", 1.3))
    return {
        "predictor_path": pp.get("predictor_path", "dlib_tools/shape_predictor_81_face_landmarks.dat"),
        "crop_size": int(pp.get("res", 256)),
        "align": bool(pp.get("align", True)),
        "scale": scale,
        "enlarge": scale,
        "interp": INTERP_ALIAS.get(interp_raw, "bilinear"),
        "border": str(pp.get("border_mode", "constant")),
        "mode": str(pp.get("sampling_mode", "fixed_num_frames")),
        "num_frames": int(pp.get("num_frames", 32)),
        "stride": int(pp.get("stride", 10)),
        "save_masks": bool(pp.get("save_masks", True)),
        "save_landmarks": bool(pp.get("save_landmarks", True)),
    }


def _resolve(path_str: str, base: Path) -> str:
    """Resolve a possibly-relative config path against `base`."""
    p = Path(path_str)
    return str(p if p.is_absolute() else (base / p))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Config-driven face extraction (FF++ / Celeb-DF v2 / DFDC).")
    p.add_argument("--dataset", required=True, choices=VIDEO_DATASETS)
    p.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    p.add_argument("--paths", default=str(ROOT / "paths.yaml"),
                   help="paths.yaml mapping dataset name -> root (copy from paths.example.yaml)")
    # root overrides (else resolved from paths.yaml)
    p.add_argument("--data-root", default=None)
    p.add_argument("--crops-root", default=None)
    # extraction overrides (None => take the value from base.yaml's preprocess block)
    p.add_argument("--predictor-path", default=None)
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--interp", default=None, choices=list(INTERP_ALIAS))
    p.add_argument("--border", default=None, choices=["constant", "reflect", "replicate"])
    p.add_argument("--mode", default=None, choices=["fixed_num_frames", "fixed_stride"])
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--no-align", action="store_true", help="ablation: square bbox crop, no rotation/warp")
    p.add_argument("--no-masks", action="store_true", help="do not save aligned FF++ manipulation masks")
    p.add_argument("--no-landmarks", action="store_true", help="do not save per-crop 81-landmark .npy")
    # dataset-specific
    p.add_argument("--comp", default="c23", choices=["raw", "c23", "c40"], help="FF++ compression")
    p.add_argument("--celebdf-test-list", default=None, help="override Celeb-DF test list path")
    p.add_argument("--dfdc-labels", default=None, help="override DFDC labels CSV path")
    p.add_argument("--limit", type=int, default=None, help="process only N videos (smoke test)")
    # manifest chaining
    p.add_argument("--build-manifest", action="store_true",
                   help="also write the unified manifest CSV after extraction")
    p.add_argument("--manifest-out", default=None)
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    cfg_yaml = load_yaml(Path(args.config))
    paths = load_yaml(Path(args.paths))
    pp = cfg_yaml.get("preprocess", {})

    data_root = args.data_root or paths.get(args.dataset)
    crops_root = args.crops_root or paths.get("crops_root")
    if not data_root:
        raise SystemExit(f"No data root for '{args.dataset}'. Add it to {args.paths} or pass --data-root.")
    if not crops_root:
        raise SystemExit(f"No 'crops_root' in {args.paths}; add it or pass --crops-root.")

    # base config -> kwargs, then apply CLI overrides
    kwargs = cfg_kwargs_from_preprocess(pp)
    if args.predictor_path is not None:
        kwargs["predictor_path"] = args.predictor_path
    if args.crop_size is not None:
        kwargs["crop_size"] = args.crop_size
    if args.scale is not None:
        kwargs["scale"] = kwargs["enlarge"] = args.scale
    if args.interp is not None:
        kwargs["interp"] = INTERP_ALIAS[args.interp]
    if args.border is not None:
        kwargs["border"] = args.border
    if args.mode is not None:
        kwargs["mode"] = args.mode
    if args.num_frames is not None:
        kwargs["num_frames"] = args.num_frames
    if args.stride is not None:
        kwargs["stride"] = args.stride
    if args.no_align:
        kwargs["align"] = False
    if args.no_masks:
        kwargs["save_masks"] = False
    if args.no_landmarks:
        kwargs["save_landmarks"] = False
    # resolve a relative predictor path against the repo root (so the command
    # works from any CWD), and test lists/labels against the dataset root
    kwargs["predictor_path"] = _resolve(kwargs["predictor_path"], ROOT)

    celebdf_default = pp.get("celebdf_test_list", "List_of_testing_videos.txt")
    dfdc_default = pp.get("dfdc_labels_file", "labels.csv")
    celebdf_test_list = (args.celebdf_test_list
                         or (_resolve(celebdf_default, Path(data_root)) if args.dataset == "celebdf_v2" else None))
    dfdc_labels = (args.dfdc_labels
                   or (_resolve(dfdc_default, Path(data_root)) if args.dataset == "dfdc" else None))

    manipulations = (cfg_yaml.get("train", {}) or {}).get("manipulations")  # None -> extractor default

    # deferred (needs dlib / cv2 / skimage — the isolated preprocess env)
    from sfdet.preprocess.extract_faces import ExtractConfig, run_extraction

    cfg = ExtractConfig(**kwargs)
    print(f"[preprocess] dataset={args.dataset}  data_root={data_root}  crops_root={crops_root}")
    print(f"[preprocess] crop={cfg.crop_size} align={cfg.align} border={cfg.border} "
          f"interp={cfg.interp} mode={cfg.mode} num_frames={cfg.num_frames} "
          f"masks={cfg.save_masks} landmarks={cfg.save_landmarks}")

    run_extraction(
        args.dataset, data_root, crops_root, cfg,
        comp=args.comp, manipulations=manipulations,
        celebdf_test_list=celebdf_test_list, dfdc_labels=dfdc_labels, limit=args.limit,
    )

    if args.build_manifest:
        from sfdet.preprocess.manifest import from_extraction_log, write_manifest
        rows = from_extraction_log(crops_root, args.dataset)
        out = args.manifest_out or str(Path(crops_root) / f"{args.dataset}_manifest.csv")
        write_manifest(rows, out)
        n_fake = sum(int(r["label"]) == 1 for r in rows)
        print(f"[preprocess] manifest -> {out}  ({len(rows)} crops; fake={n_fake}, real={len(rows) - n_fake})")


if __name__ == "__main__":
    main()
