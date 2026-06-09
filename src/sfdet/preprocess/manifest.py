#!/usr/bin/env python3
"""manifest.py — build the unified crop manifest for all five datasets.

One schema, two sources:
  * the three VIDEO datasets (FF++ c23, Celeb-DF v2, DFDC) — read the
    ``_extraction_log.jsonl`` that extract_faces.py wrote under
    ``crops_root/<dataset>/`` and FLATTEN it to one row per crop (videos that
    yielded zero faces are already marked excluded in the log and are dropped).
  * the two PRE-CROPPED datasets (WildDeepfake, DF40 diffusion subsets) — walk
    their provided crop folders directly (no extraction step).

The point of this file is that the manifest, not the on-disk folder shape, is the
unifying layer: datasets.py reads these columns and treats all five identically.

Unified columns:
    crop_path, label, dataset, subset, domain, source_video_id, frame,
    mask_path, landmark_path

  label    : 0 real / 1 fake (fake = positive class, AUC convention)
  subset   : FF++ manipulation ("original" for real) or DF40 method name
             (stable_diffusion_2_1 / ddpm / pixart_alpha / dit_xl_2); "real" for
             the DF40 real class; "" otherwise
  domain   : DF40 sub-domain "ff"/"cdf"; "" otherwise
  frame    : source frame index (video datasets) or image stem (pre-cropped)
  mask_path / landmark_path : populated for FF++ (the aligned mask feeds the E5
             GT-mask faithfulness check); "" otherwise

Paths are absolute (machine-local). The manifest is a DERIVED artifact and is
gitignored — rebuild it on each machine after extraction / download.

Run (matches the line extract_faces.py prints):
    # video datasets, after extract_faces.py:
    python -m sfdet.preprocess.manifest --dataset faceforensics_c23 --crops-root /data/crops
    python -m sfdet.preprocess.manifest --dataset celebdf_v2        --crops-root /data/crops
    python -m sfdet.preprocess.manifest --dataset dfdc              --crops-root /data/crops
    # pre-cropped datasets:
    python -m sfdet.preprocess.manifest --dataset wilddeepfake   --data-root /data/WildDeepfake
    python -m sfdet.preprocess.manifest --dataset df40_diffusion --data-root /data/DF40
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REAL, FAKE = 0, 1
FIELDS = ["crop_path", "label", "dataset", "subset", "domain",
          "source_video_id", "frame", "mask_path", "landmark_path"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

VIDEO_DATASETS = ("faceforensics_c23", "celebdf_v2", "dfdc")
PRECROPPED_DATASETS = ("wilddeepfake", "df40_diffusion")

# DF40 on-disk folder -> reporting name. The `ddim` folder IS DF40's "DDPM"
# (method #29); there is no `ddpm/` folder on disk. Keep this in sync with
# configs/data/df40_diffusion.yaml.
DF40_SUBSET_FOLDERS = {
    "sd2.1": "stable_diffusion_2_1",
    "ddim": "ddpm",
    "PixArt": "pixart_alpha",
    "DiT": "dit_xl_2",
}


def _row(crop_path, label, dataset, *, subset="", domain="",
         source_video_id="", frame="", mask_path="", landmark_path=""):
    return {
        "crop_path": str(crop_path), "label": int(label), "dataset": dataset,
        "subset": subset, "domain": domain, "source_video_id": source_video_id,
        "frame": frame, "mask_path": mask_path, "landmark_path": landmark_path,
    }


def _iter_images(base: Path):
    """Yield image files under `base` (recursively), in sorted order."""
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _ci_dir(parent: Path, name: str) -> Path:
    """Return parent's child directory matching `name` case-INSENSITIVELY; if none
    matches, fall back to parent/name so the existing 'not found' warning still fires.
    DF40 downloads vary in case (e.g. on-disk `pixart` vs the canonical `PixArt`, or
    `ff`/`FF`). Windows is case-insensitive so a build there hides the problem, but the
    training box (Linux) is case-sensitive — resolving case here keeps the SAME manifest
    command working on both and prevents a subset silently dropping out of the manifest."""
    parent = Path(parent)
    if parent.is_dir():
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    return parent / name


# --------------------------------------------------------------------------- #
# Source 1 — extraction log (the three video datasets)
# --------------------------------------------------------------------------- #
def from_extraction_log(crops_root, dataset) -> list:
    log_path = Path(crops_root) / dataset / "_extraction_log.jsonl"
    if not log_path.is_file():
        raise SystemExit(
            f"Extraction log not found: {log_path}\n"
            f"Run extract_faces.py for {dataset} first:\n"
            f"    python -m sfdet.preprocess.extract_faces --dataset {dataset} "
            f"--data-root <...> --crops-root {crops_root} --predictor-path <...>"
        )
    rows, n_excluded = [], 0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            if v.get("excluded_zero_faces") or not v.get("crops"):
                n_excluded += 1
                continue
            for c in v["crops"]:
                rows.append(_row(
                    c["crop_path"], v["label"], v["dataset"],
                    subset=v.get("subset", ""), domain=v.get("domain", ""),
                    source_video_id=v["source_video_id"], frame=c.get("frame", ""),
                    mask_path=c.get("mask_path", ""), landmark_path=c.get("landmark_path", ""),
                ))
    print(f"[{dataset}] {len(rows)} crops from {log_path.name} "
          f"({n_excluded} zero-face videos excluded)")
    return rows


# --------------------------------------------------------------------------- #
# Source 2a — WildDeepfake (pre-cropped face sequences)
# --------------------------------------------------------------------------- #
def from_wilddeepfake(root, real_dir="real_test", fake_dir="fake_test") -> list:
    """Walk real_test/<seq>/<frame>.png and fake_test/<seq>/<frame>.png. Each
    sequence folder is one source video (gives video-level grouping for free)."""
    root = Path(root)
    rows = []
    for sub, label in ((real_dir, REAL), (fake_dir, FAKE)):
        base = root / sub
        if not base.is_dir():
            print(f"WARNING: {base} not found; skipping (verify WildDeepfake layout).", file=sys.stderr)
            continue
        for img in _iter_images(base):
            rows.append(_row(img.resolve(), label, "wilddeepfake",
                             source_video_id=img.parent.name, frame=img.stem))
    print(f"[wilddeepfake] {len(rows)} crops")
    return rows


# --------------------------------------------------------------------------- #
# Source 2b — DF40 diffusion subsets (pre-cropped; fake-only + separate real)
# --------------------------------------------------------------------------- #
def from_df40(root, subset_folders=None, domains=("ff", "cdf"), real_dirname="real") -> list:
    """Walk <root>/<folder>/<domain>/**/*.png for fakes and <root>/<real>/<domain>/
    for the shared real class. `source_video_id` is the immediate parent folder —
    if DF40's crops are NOT nested per source video, this falls back to the domain
    dir and video-level reduction for DF40 is not meaningful (frame-level only;
    ties into the open frame-vs-video decision)."""
    root = Path(root)
    subset_folders = subset_folders or DF40_SUBSET_FOLDERS
    rows = []
    for folder, name in subset_folders.items():
        for dom in domains:
            base = _ci_dir(_ci_dir(root, folder), dom)
            if not base.is_dir():
                print(f"WARNING: {base} not found; skipping (verify DF40 layout / download).",
                      file=sys.stderr)
                continue
            for img in _iter_images(base):
                rows.append(_row(img.resolve(), FAKE, "df40_diffusion",
                                 subset=name, domain=dom,
                                 source_video_id=img.parent.name, frame=img.stem))
    for dom in domains:
        base = _ci_dir(_ci_dir(root, real_dirname), dom)
        if not base.is_dir():
            print(f"WARNING: DF40 real dir {base} not found — download the FF++ / Celeb-DF "
                  "real packs (DF40 ships fakes only).", file=sys.stderr)
            continue
        for img in _iter_images(base):
            rows.append(_row(img.resolve(), REAL, "df40_diffusion",
                             subset="real", domain=dom,
                             source_video_id=img.parent.name, frame=img.stem))
    print(f"[df40_diffusion] {len(rows)} crops "
          f"(subsets={list(subset_folders.values())}, domains={list(domains)}, + real)")
    return rows


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def write_manifest(rows, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def read_manifest(path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the unified crop manifest (CSV) for one dataset.")
    p.add_argument("--dataset", required=True,
                   choices=list(VIDEO_DATASETS) + list(PRECROPPED_DATASETS))
    p.add_argument("--crops-root", default=None,
                   help="VIDEO datasets: dir where extract_faces.py wrote crops + _extraction_log.jsonl")
    p.add_argument("--data-root", default=None,
                   help="PRE-CROPPED datasets (wilddeepfake / df40_diffusion): the dataset root")
    p.add_argument("--out", default=None, help="output manifest CSV path (default: alongside the data)")
    p.add_argument("--df40-domains", nargs="+", default=["ff", "cdf"],
                   help="DF40 sub-domains to include: ff, cdf, or both (default both)")
    args = p.parse_args(argv)

    if args.dataset in VIDEO_DATASETS:
        if not args.crops_root:
            p.error("--crops-root is required for the video datasets")
        rows = from_extraction_log(args.crops_root, args.dataset)
        default_out = Path(args.crops_root) / f"{args.dataset}_manifest.csv"
    elif args.dataset == "wilddeepfake":
        if not args.data_root:
            p.error("--data-root is required for wilddeepfake")
        rows = from_wilddeepfake(args.data_root)
        default_out = Path(args.data_root) / "wilddeepfake_manifest.csv"
    else:  # df40_diffusion
        if not args.data_root:
            p.error("--data-root is required for df40_diffusion")
        rows = from_df40(args.data_root, domains=tuple(args.df40_domains))
        default_out = Path(args.data_root) / "df40_diffusion_manifest.csv"

    if not rows:
        raise SystemExit(f"[{args.dataset}] no crops found — nothing written. Check the paths above.")

    out = write_manifest(rows, args.out or default_out)
    n_fake = sum(int(r["label"]) == FAKE for r in rows)
    print(f"[{args.dataset}] wrote {len(rows)} rows -> {out}  (fake={n_fake}, real={len(rows) - n_fake})")


if __name__ == "__main__":
    main()
