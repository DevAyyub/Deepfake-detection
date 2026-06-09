#!/usr/bin/env python
"""verify_dataloaders.py — smoke-test the data pipeline across all five datasets.

Builds the loaders via sfdet.data.dataset.get_dataloaders and pulls real batches
from each, asserting the batch CONTRACT the model code depends on:

    batch["spatial"]    [B, 3, S, S]   float32, ImageNet-normalized (NOT in [0,1])
    batch["frequency"]  [B, C, S, S]   float32, per-image normalized (C=1 gray / 3 rgb)
    batch["label"]      [B]            float32, values in {0, 1}
    + metadata lists (source_video_id / dataset / subset / domain / frame /
      crop_path / mask_path / landmark_path), each length B.

It also checks: no NaN/Inf (the FFT/log1p path is the main risk), that the TRAIN
sampler actually yields ~50/50 real:fake with the four FF++ methods represented,
that EVAL loaders are deterministic, and that train augmentation is live while
eval is not. Datasets whose manifest isn't on disk yet (e.g. DFDC before download)
are reported as NOT COVERED rather than treated as a pass.

RUN (from repo root, on the box that has the crops + FF++ split JSONs):
    python scripts/verify_dataloaders.py
    python scripts/verify_dataloaders.py --num-workers 4 --train-batches 16

Exit code 0 = every built loader passed; 1 = a contract failure; 2 = the loaders
could not be built (usually the FF++ manifest or the official split JSONs missing).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch  # the rest of the heavy imports are deferred into main()
from torch.utils.data import RandomSampler, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # work even without `pip install -e .`

REQUIRED_KEYS = ["spatial", "frequency", "label", "source_video_id", "dataset",
                 "subset", "domain", "frame", "crop_path", "mask_path", "landmark_path"]
META_KEYS = ["source_video_id", "dataset", "subset", "domain", "frame",
             "crop_path", "mask_path", "landmark_path"]
EXPECTED_FAMILIES = ["faceforensics_c23", "celebdf_v2", "dfdc", "wilddeepfake", "df40_diffusion"]


def family_of(loader_name: str) -> str:
    """Map a loader name to its dataset family for coverage reporting."""
    if loader_name.startswith("faceforensics_c23"):
        return "faceforensics_c23"
    if loader_name.startswith("df40"):
        return "df40_diffusion"
    return loader_name  # celebdf_v2 / dfdc / wilddeepfake


def verify_batch(batch: dict, image_size: int, freq_channels: int):
    """Check a single batch against the contract. Returns (ok, problems, stats).
    Kept dependency-light (a small, fixed set of tensor ops) so it is unit-testable."""
    problems, stats = [], {}

    missing = [k for k in REQUIRED_KEYS if k not in batch]
    if missing:
        return False, [f"missing keys: {missing}"], stats

    sp, fr, lb = batch["spatial"], batch["frequency"], batch["label"]
    B = int(sp.shape[0])
    stats["B"] = B

    # --- spatial: [B,3,S,S] float32, finite, normalized ---
    if sp.ndim != 4:
        problems.append(f"spatial ndim {sp.ndim} != 4")
    elif int(sp.shape[1]) != 3:
        problems.append(f"spatial channels {int(sp.shape[1])} != 3")
    elif tuple(int(d) for d in sp.shape[2:]) != (image_size, image_size):
        problems.append(f"spatial HxW {tuple(int(d) for d in sp.shape[2:])} != {(image_size, image_size)}")
    if sp.dtype != torch.float32:
        problems.append(f"spatial dtype {sp.dtype} != float32")
    if not bool(torch.isfinite(sp).all()):
        problems.append("spatial has NaN/Inf")
    smin, smax, smean = float(sp.min()), float(sp.max()), float(sp.mean())
    stats["spatial"] = (smin, smax, smean)
    if smin >= 0.0 and smax <= 1.0:
        problems.append("spatial looks un-normalized (all values in [0,1]) — ImageNet normalize missing?")

    # --- frequency: [B,C,S,S] float32, finite, ~per-image standardized ---
    if fr.ndim != 4:
        problems.append(f"frequency ndim {fr.ndim} != 4")
    elif int(fr.shape[1]) != freq_channels:
        problems.append(f"frequency channels {int(fr.shape[1])} != expected {freq_channels}")
    elif tuple(int(d) for d in fr.shape[2:]) != (image_size, image_size):
        problems.append(f"frequency HxW {tuple(int(d) for d in fr.shape[2:])} != {(image_size, image_size)}")
    if fr.dtype != torch.float32:
        problems.append(f"frequency dtype {fr.dtype} != float32")
    if not bool(torch.isfinite(fr).all()):
        problems.append("frequency has NaN/Inf (FFT/log1p/normalize blew up)")
    fmin, fmax, fmean, fstd = float(fr.min()), float(fr.max()), float(fr.mean()), float(fr.std())
    stats["frequency"] = (fmin, fmax, fmean, fstd)
    if not (0.3 < fstd < 3.0):  # per-image z-score -> pooled std ~1; flag a gross miss, don't fail hard
        problems.append(f"frequency std {fstd:.3f} far from ~1 (per-image normalization off?)")

    # --- label: [B] float32 in {0,1} ---
    if lb.ndim != 1 or int(lb.shape[0]) != B:
        problems.append(f"label shape {tuple(int(d) for d in lb.shape)} != ({B},)")
    if lb.dtype != torch.float32:
        problems.append(f"label dtype {lb.dtype} != float32 (BCEWithLogits expects float)")
    lab_list = [float(x) for x in lb.tolist()]
    bad = sorted({v for v in lab_list if v not in (0.0, 1.0)})
    if bad:
        problems.append(f"labels outside {{0,1}}: {bad[:5]}")
    stats["fake_frac"] = (sum(1 for v in lab_list if v == 1.0) / B) if B else 0.0

    # --- metadata list lengths ---
    for k in META_KEYS:
        if len(batch[k]) != B:
            problems.append(f"metadata '{k}' length {len(batch[k])} != B {B}")
    stats["datasets"] = sorted(set(batch["dataset"]))
    stats["subsets"] = sorted(set(s for s in batch["subset"] if s))
    stats["domains"] = sorted(set(d for d in batch["domain"] if d))
    return (len(problems) == 0), problems, stats


def _fmt(stats: dict) -> str:
    sp = stats.get("spatial"); fr = stats.get("frequency")
    out = [f"B={stats.get('B')}"]
    if sp:
        out.append(f"spatial[min={sp[0]:.2f} max={sp[1]:.2f} mean={sp[2]:.2f}]")
    if fr:
        out.append(f"freq[min={fr[0]:.2f} max={fr[1]:.2f} mean={fr[2]:.3f} std={fr[3]:.2f}]")
    out.append(f"fake={stats.get('fake_frac', 0):.2f}")
    if stats.get("subsets"):
        out.append(f"subsets={stats['subsets']}")
    if stats.get("domains"):
        out.append(f"domains={stats['domains']}")
    return "  ".join(out)


def check_loader(name, loader, image_size, freq_channels, *, n_batches, is_train):
    """Pull batches from one loader, verify the first, aggregate label stats over all.
    Returns (passed, n_samples_seen). Prints a per-loader block."""
    print(f"\n--- {name} ---")
    n_have = len(loader.dataset)
    print(f"    dataset rows: {n_have}")
    if n_have == 0:
        print("    [FAIL] loader has 0 rows")
        return False, 0

    t0 = time.time()
    it = iter(loader)
    try:
        first = next(it)
    except StopIteration:
        print("    [FAIL] loader yielded no batches")
        return False, 0
    dt = time.time() - t0

    ok, problems, stats = verify_batch(first, image_size, freq_channels)
    print(f"    first batch in {dt:.2f}s | {_fmt(stats)}")
    for p in problems:
        print(f"    [FAIL] {p}")
    passed = ok

    # aggregate label balance / method coverage across a few batches
    fake = total = 0
    methods = Counter()
    for b in [first] + [next(it, None) for _ in range(n_batches - 1)]:
        if b is None:
            break
        labs = [float(x) for x in b["label"].tolist()]
        total += len(labs)
        fake += sum(1 for v in labs if v == 1.0)
        for lab, sub in zip(labs, b["subset"]):
            if lab == 1.0 and sub:
                methods[sub] += 1
    frac = fake / total if total else 0.0

    if is_train:
        print(f"    sampler balance over {total} samples: fake={frac:.3f} (target ~0.5)")
        if methods:
            print(f"    fake methods seen: {dict(methods)}")
        if not (0.35 <= frac <= 0.65):
            print(f"    [WARN] train fake fraction {frac:.3f} outside [0.35, 0.65] — check balance='method'")
        if not isinstance(loader.sampler, WeightedRandomSampler):
            print(f"    [WARN] train loader sampler is {type(loader.sampler).__name__}; "
                  "expected WeightedRandomSampler (check balance='method')")
    else:
        if isinstance(loader.sampler, (RandomSampler, WeightedRandomSampler)):
            print(f"    [FAIL] eval loader uses {type(loader.sampler).__name__} "
                  "(should be a deterministic SequentialSampler)")
            passed = False
        # determinism: same first-batch order on a fresh pass
        try:
            again = next(iter(loader))
            if again["source_video_id"] != first["source_video_id"]:
                print("    [WARN] eval order changed between passes — not deterministic")
            else:
                print("    deterministic order: OK")
        except StopIteration:
            pass

    return passed, total


def check_augmentation(train_loader, val_loader):
    """Soft check: train transform is stochastic (jitter/flip) and eval is not.
    Reads a single crop twice via the dataset; skipped if files aren't reachable."""
    print("\n--- augmentation liveness (soft) ---")
    try:
        s0 = train_loader.dataset[0]["spatial"]
        s1 = train_loader.dataset[0]["spatial"]
        if torch.equal(s0, s1):
            print("    [WARN] two train reads of idx 0 are identical — augmentation may be off "
                  "(could be a no-flip + ~unit-jitter draw; re-run to confirm)")
        else:
            print("    train augmentation is live (idx 0 differs across reads): OK")
    except Exception as e:  # noqa: BLE001
        print(f"    train aug check skipped ({type(e).__name__}: {e})")
    try:
        e0 = val_loader.dataset[0]["spatial"]
        e1 = val_loader.dataset[0]["spatial"]
        print("    eval transform deterministic (idx 0 identical): OK" if torch.equal(e0, e1)
              else "    [FAIL] eval transform is NOT deterministic (idx 0 differs)")
    except Exception as e:  # noqa: BLE001
        print(f"    eval aug check skipped ({type(e).__name__}: {e})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test all five dataloaders.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--paths", default=str(REPO_ROOT / "paths.yaml"))
    ap.add_argument("--splits-dir", default=None, help="override FF++ splits dir (else paths.yaml/<ff_root>/splits)")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the smoke test (default 0)")
    ap.add_argument("--train-batches", type=int, default=8, help="batches to sample for the train balance check")
    ap.add_argument("--eval-batches", type=int, default=2, help="batches to sample per eval loader")
    args = ap.parse_args()

    from sfdet.data.dataset import get_dataloaders, load_configs

    cfg, paths = load_configs(args.config, args.paths)
    image_size = int(cfg["data"]["image_size"])
    freq_channels = 3 if str(cfg["data"].get("frequency", {}).get("channels", "gray")).lower() == "rgb" else 1
    print(f"config: image_size={image_size}, frequency_channels={freq_channels}, "
          f"num_workers={args.num_workers}")

    try:
        train_loader, val_loader, test_loaders = get_dataloaders(
            cfg, paths, splits_dir=args.splits_dir, num_workers=args.num_workers, verbose=True)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[ABORT] could not build loaders:\n  {e}\n"
              "Most often: the FF++ manifest isn't built yet (run scripts/preprocess_faces.py "
              "--dataset faceforensics_c23 --build-manifest) or the official FF++ split JSONs "
              "(train/val/test.json) aren't where paths.yaml['ffpp_splits'] points.")
        return 2

    loaders = [("faceforensics_c23 [train]", train_loader, True),
               ("faceforensics_c23 [val]", val_loader, False)]
    loaders += [(name, dl, False) for name, dl in test_loaders.items()]

    failures, covered = [], set()
    for name, dl, is_train in loaders:
        nb = args.train_batches if is_train else args.eval_batches
        passed, seen = check_loader(name, dl, image_size, freq_channels, n_batches=nb, is_train=is_train)
        covered.add(family_of(name))
        if not passed:
            failures.append(name)

    check_augmentation(train_loader, val_loader)

    # ---- summary ----
    print("\n" + "=" * 64)
    missing_families = [f for f in EXPECTED_FAMILIES if f not in covered]
    print(f"loaders checked: {len(loaders)} | families covered: "
          f"{len(covered)}/{len(EXPECTED_FAMILIES)} ({sorted(covered)})")
    if missing_families:
        print(f"NOT COVERED (manifest absent — download/extract, then re-run): {missing_families}")
    if failures:
        print(f"RESULT: FAIL — contract problems in: {failures}")
        return 1
    print("RESULT: PASS — every built loader satisfies the batch contract"
          + (" (some datasets not yet on disk; see above)" if missing_families else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
