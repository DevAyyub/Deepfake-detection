"""sfdet.data.dataset — DataLoaders for all five datasets (consolidated).

This single module holds what the package layout sketched as datasets / transforms
/ samplers / video_grouping. It is built around one fact that makes the usual
deepfake-loader playbook partly *wrong* here:

    The frequency branch reads a deterministic FFT-magnitude of the crop, and the
    signal C2 rests on is the generation-artifact structure in that spectrum. So
    any augmentation that resamples or filters the crop (blur, JPEG, resize,
    cutout) corrupts exactly what the frequency branch must read.

Design decisions encoded here (see the step-4 discussion):

  * MODALITY-SPLIT AUGMENTATION (default). The shared crop gets only lossless
    geometry (horizontal flip — a reflection, no interpolation, spectrum-preserving
    up to a theta mirror). The SPATIAL branch additionally gets mild photometric
    jitter (brightness/contrast). The FREQUENCY branch gets the FFT of the
    flip-only crop and nothing else. `AugConfig.shared_crop_aug=True` is the
    ablation that instead applies jitter to the shared crop (so the spectrum sees
    it) — kept togglable so the choice is measurable, not asserted.
    Because flip is shared and photometric ops do not move geometry, the spatial
    Grad-CAM and the (rho, theta) frequency saliency stay spatially registered.

  * FFT IS COMPUTED IN THE DATASET (CPU, parallel across workers), as a fixed,
    inspectable transform — fftshifted log1p magnitude, per-image normalised, in
    Cartesian grid (the polar (rho, theta) reading is applied at saliency time,
    step 10, not resampled here). Alternative: compute it on GPU inside the
    frequency branch; `FFTMagnitude` is a standalone class to make that move easy.

  * FF++ OFFICIAL SPLIT. train/val/test partition comes from the official
    FaceForensics++ JSON split files (identity-disjoint pairs), applied at LOAD
    time over the FF++ manifest. A manipulated clip inherits its SOURCE video's
    split (a fake is in train iff its original is), which is what keeps the split
    leakage-free and comparable to every published baseline. We read the official
    files rather than reconstruct counts — a wrong split silently breaks
    comparability.

  * TEST-ONLY for the other four. Only FF++ is split three ways. Celeb-DF v2,
    DFDC, WildDeepfake, and DF40 are eval-only: one train loader + one val loader
    (both FF++) + N eval loaders. DF40 yields one loader per subset x domain
    (per-subset AUC is C3's artifact); it ships fakes only, so each DF40 loader
    pairs the subset's fakes with the domain-matched real crops in the manifest.
    Eval loaders: no augmentation, no rebalancing, no shuffle — the natural
    distribution, or AUC/EER lose meaning.

  * CLASS IMBALANCE (train only). FF++ is ~1 real : 4 fake (four manipulation
    methods per original). A WeightedRandomSampler draws ~50/50 real/fake; the
    fake half is split evenly across the four methods (`balance='method'`) so no
    method dominates. AUC is threshold-free, so this is about a cleaner decision
    boundary, not rescuing the metric.

  * FRAME vs VIDEO (open, notes 1.10(a)). Train is frame-level; every sample
    carries source_video_id / dataset / subset / domain so the evaluator can
    reduce frame scores to a video score and report BOTH. Compare to SBI only at
    video-level and to Qiao only at frame-level — never mix in one column.

Filesystem roots come from paths.yaml; manifests are produced by
sfdet.preprocess.manifest. get_dataloaders takes already-loaded dicts so it is
import-light and testable; load_configs() is a convenience for the trainer.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF

from sfdet.preprocess.manifest import read_manifest

# ImageNet stats for the EfficientNet-B4 (ImageNet-pretrained) spatial branch.
# timm models carry their own data cfg; override via DualViewTransform if a model
# expects different normalisation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REAL, FAKE = 0, 1
VIDEO_DATASETS = ("faceforensics_c23", "celebdf_v2", "dfdc")
TEST_DATASETS = ("celebdf_v2", "dfdc", "wilddeepfake")  # DF40 handled separately (per subset)


# --------------------------------------------------------------------------- #
# Frequency-branch input: deterministic FFT-magnitude transform
# --------------------------------------------------------------------------- #
class FFTMagnitude:
    """crop ([3,H,W] in [0,1]) -> fftshifted log1p magnitude, per-image normalised.

    Cartesian grid (DC-centred). The polar (rho, theta) reading is applied to the
    *saliency*, not to this input, so the frequency CNN sees a standard image-like
    tensor. Grayscale (luminance) by default -> 1 channel; rgb -> 3 channels.
    """

    def __init__(self, grayscale: bool = True, log_scale: bool = True,
                 normalize: str = "per_image", eps: float = 1e-6):
        self.grayscale = grayscale
        self.log_scale = log_scale
        self.normalize = normalize
        self.eps = eps

    def __call__(self, img: "torch.Tensor") -> "torch.Tensor":
        x = img
        if self.grayscale:
            # BT.601 luminance; keeps a single, phase-free magnitude channel.
            x = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
        spec = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        mag = spec.abs()
        if self.log_scale:
            mag = torch.log1p(mag)                      # compress the huge DC/low-freq dynamic range
        if self.normalize == "per_image":
            m = mag.mean(dim=(-2, -1), keepdim=True)
            s = mag.std(dim=(-2, -1), keepdim=True)
            mag = (mag - m) / (s + self.eps)            # per-sample standardisation
        return mag.float()


# --------------------------------------------------------------------------- #
# Augmentation (train only) + the dual-view (spatial, frequency) transform
# --------------------------------------------------------------------------- #
@dataclass
class AugConfig:
    hflip: bool = True              # lossless geometry on the SHARED crop (spectrum-safe)
    brightness: float = 0.1         # spatial-branch-only photometric jitter (+/- fraction)
    contrast: float = 0.1
    shared_crop_aug: bool = False   # ABLATION: apply jitter to the shared crop so the
    #                                 frequency branch also sees it (vs modality-split default)


class DualViewTransform:
    """Produce (spatial_tensor, frequency_tensor) from one PIL crop.

    Train: optional shared horizontal flip, then spatial-only photometric jitter
    (unless shared_crop_aug), ImageNet-normalised spatial view + FFT of the
    flip-only crop. Eval: identity (no flip/jitter) -> normalised spatial + FFT.
    """

    def __init__(self, image_size: int, train: bool, aug: AugConfig, fft: FFTMagnitude,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.image_size = image_size
        self.train = train
        self.aug = aug
        self.fft = fft
        self.mean = list(mean)
        self.std = list(std)

    def _jitter(self, x: "torch.Tensor") -> "torch.Tensor":
        b = 1.0 + random.uniform(-self.aug.brightness, self.aug.brightness)
        c = 1.0 + random.uniform(-self.aug.contrast, self.aug.contrast)
        x = TF.adjust_brightness(x, max(b, 0.0))
        x = TF.adjust_contrast(x, max(c, 0.0))
        return x

    def __call__(self, img: "Image.Image"):
        x = TF.to_tensor(img)                                   # [3,H,W] float in [0,1], RGB
        if x.shape[-2:] != (self.image_size, self.image_size):
            # crops are already 256^2 from extraction / DF40; resize only when an
            # off-size dataset (e.g. WildDeepfake) requires it. ONE resample, fixed
            # interpolation, so the spectrum stays uniform within that dataset.
            x = TF.resize(x, [self.image_size, self.image_size], antialias=True)

        if self.train and self.aug.hflip and random.random() < 0.5:
            x = TF.hflip(x)                                     # SHARED, lossless

        if self.train and self.aug.shared_crop_aug:
            x = self._jitter(x)                                 # ablation: spectrum sees jitter too

        frequency = self.fft(x)                                 # from the flip-only [0,1] crop

        spatial = x
        if self.train and not self.aug.shared_crop_aug:
            spatial = self._jitter(spatial)                     # spatial-branch-only photometric
        spatial = TF.normalize(spatial, self.mean, self.std)
        return spatial, frequency


# --------------------------------------------------------------------------- #
# Dataset — one class for all five (the manifest is the unifying layer)
# --------------------------------------------------------------------------- #
class BaseDeepfakeDataset(Dataset):
    """Reads pre-extracted/pre-cropped face PNGs listed in a manifest and yields a
    dict with both branch inputs plus the metadata needed for video-level
    reduction, per-subset breakdown, and the E5 GT-mask check."""

    def __init__(self, rows: list, transform: DualViewTransform, image_size: int):
        self.rows = rows
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        img = Image.open(r["crop_path"]).convert("RGB")
        spatial, frequency = self.transform(img)
        return {
            "spatial": spatial,
            "frequency": frequency,
            "label": torch.tensor(float(int(r["label"])), dtype=torch.float32),
            "source_video_id": r.get("source_video_id", ""),
            "dataset": r.get("dataset", ""),
            "subset": r.get("subset", ""),
            "domain": r.get("domain", ""),
            "frame": str(r.get("frame", "")),
            "crop_path": r["crop_path"],
            "mask_path": r.get("mask_path", ""),
            "landmark_path": r.get("landmark_path", ""),
        }


def collate_batch(batch: list) -> dict:
    """Stack tensors; keep metadata as parallel lists (video grouping reads these)."""
    out = {
        "spatial": torch.stack([b["spatial"] for b in batch]),
        "frequency": torch.stack([b["frequency"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
    for k in ("source_video_id", "dataset", "subset", "domain",
              "frame", "crop_path", "mask_path", "landmark_path"):
        out[k] = [b[k] for b in batch]
    return out


# --------------------------------------------------------------------------- #
# FaceForensics++ official split (identity-disjoint; fake inherits source split)
# --------------------------------------------------------------------------- #
def _ffpp_origin_id(source_video_id: str) -> str:
    """'original__033' -> '033'; 'Deepfakes__033_097' -> '033' (the target id).
    Both ids of an FF++ pair live in the same split, so the target id suffices."""
    body = source_video_id.split("__", 1)[-1]
    return body.split("_", 1)[0]


def load_ffpp_split_ids(splits_dir, split: str) -> set:
    """Read the official <split>.json (list of [id, id] pairs) -> set of ids."""
    path = Path(splits_dir) / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"FaceForensics++ split file not found: {path}\n"
            "Download the official train.json / val.json / test.json from the "
            "FaceForensics++ repo (dataset/splits/) and point --splits-dir or "
            "paths.yaml['ffpp_splits'] at that directory. Do NOT hand-roll the "
            "split — it must match the published 720/140/140 identity partition."
        )
    pairs = json.loads(path.read_text())
    ids = set()
    for pair in pairs:
        for vid in pair:
            ids.add(str(vid))
    return ids


def apply_ffpp_split(rows: list, splits_dir, split: str) -> list:
    """Keep FF++ manifest rows whose origin id is in the requested split."""
    ids = load_ffpp_split_ids(splits_dir, split)
    return [r for r in rows
            if r.get("dataset") == "faceforensics_c23"
            and _ffpp_origin_id(r["source_video_id"]) in ids]


# --------------------------------------------------------------------------- #
# Class-imbalance sampling (train only)
# --------------------------------------------------------------------------- #
def balance_weights(rows: list, balance: str = "method") -> Optional[list]:
    """Per-sample weights for a WeightedRandomSampler.

    'binary' -> ~50/50 real/fake. 'method' -> 50% real, 50% fake with the fake mass
    split EVENLY across manipulation methods (the FF++ 'subset' field) so each of
    the four methods is equally represented. None -> no weighting (caller shuffles).
    """
    if balance is None:
        return None
    n = len(rows)
    if n == 0:
        return None
    labels = [int(r["label"]) for r in rows]
    n_fake = sum(labels)
    n_real = n - n_fake
    w = [0.0] * n
    if balance == "binary":
        for i, l in enumerate(labels):
            w[i] = (0.5 / n_real) if (l == REAL and n_real) else (0.5 / n_fake if n_fake else 0.0)
    elif balance == "method":
        method_counts = Counter(rows[i].get("subset", "") for i in range(n) if labels[i] == FAKE)
        n_methods = max(len(method_counts), 1)
        for i, l in enumerate(labels):
            if l == REAL:
                w[i] = 0.5 / max(n_real, 1)
            else:
                m = rows[i].get("subset", "")
                w[i] = 0.5 * (1.0 / n_methods) / max(method_counts[m], 1)
    else:
        raise ValueError(f"balance must be 'method', 'binary', or None; got {balance!r}")
    return w


def make_balanced_sampler(rows: list, balance: str = "method"):
    weights = balance_weights(rows, balance)
    if weights is None:
        return None
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


def seed_worker(worker_id: int) -> None:
    """Reproducible per-worker RNG for the flip/jitter draws."""
    base = torch.initial_seed() % 2 ** 32
    random.seed(base)
    try:
        import numpy as np
        np.random.seed(base % 2 ** 32)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# DF40: group fakes by (subset, domain), pair with domain-matched reals
# --------------------------------------------------------------------------- #
def group_df40(rows: list) -> dict:
    """{(subset, domain): fake_rows + domain-matched real_rows} for each diffusion
    subset present. DF40 ships fakes only; reals (subset=='real') were downloaded
    separately and are shared across subsets of the same domain."""
    reals_by_domain = defaultdict(list)
    fakes = defaultdict(list)
    for r in rows:
        if r.get("subset") == "real":
            reals_by_domain[r.get("domain", "")].append(r)
        else:
            fakes[(r.get("subset", ""), r.get("domain", ""))].append(r)
    groups = {}
    for (subset, domain), frows in sorted(fakes.items()):
        groups[(subset, domain)] = frows + reals_by_domain.get(domain, [])
    return groups


# --------------------------------------------------------------------------- #
# Path conventions (mirror sfdet.preprocess.manifest defaults)
# --------------------------------------------------------------------------- #
def _default_manifests(paths: dict) -> dict:
    crops = paths.get("crops_root", "")
    return {
        "faceforensics_c23": str(Path(crops) / "faceforensics_c23_manifest.csv"),
        "celebdf_v2": str(Path(crops) / "celebdf_v2_manifest.csv"),
        "dfdc": str(Path(crops) / "dfdc_manifest.csv"),
        "wilddeepfake": str(Path(paths.get("wilddeepfake", "")) / "wilddeepfake_manifest.csv"),
        "df40_diffusion": str(Path(paths.get("df40_diffusion", "")) / "df40_diffusion_manifest.csv"),
    }


def _default_splits_dir(paths: dict) -> str:
    return paths.get("ffpp_splits") or str(Path(paths.get("faceforensics_c23", "")) / "splits")


def load_configs(config_path="configs/base.yaml", paths_path="paths.yaml"):
    """Convenience for the trainer: (cfg, paths) dicts from the two YAML files."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    paths = yaml.safe_load(Path(paths_path).read_text())
    return cfg, paths


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #
def get_dataloaders(cfg: dict, paths: dict, *, splits_dir=None, manifests=None,
                    num_workers=None, balance: str = "method",
                    include_indomain_test: bool = False, verbose: bool = True):
    """Build (train_loader, val_loader, test_loaders).

    train/val are FF++ (val for checkpoint selection per save_best_on). test_loaders
    is {dataset_name: DataLoader} for Celeb-DF v2 / DFDC / WildDeepfake plus one
    'df40_<subset>_<domain>' loader per DF40 diffusion subset. Missing test
    manifests are skipped with a warning (so you can train before every test set is
    on disk); a missing FF++ manifest or split is a hard error.

    If include_indomain_test=True, the FF++ identity-disjoint *test* split is added
    as test_loaders['faceforensics_c23'] — the in-domain ceiling row for evaluation.
    """
    image_size = int(cfg["data"]["image_size"])
    fcfg = cfg["data"].get("frequency", {})
    grayscale = str(fcfg.get("channels", "gray")).lower() != "rgb"
    fft = FFTMagnitude(grayscale=grayscale,
                       log_scale=bool(fcfg.get("log_scale", True)),
                       normalize=str(fcfg.get("normalize", "per_image")))
    acfg = cfg["data"].get("augmentation", {})
    aug = AugConfig(
        hflip=bool(acfg.get("hflip", True)),
        brightness=float(acfg.get("brightness", 0.1)),
        contrast=float(acfg.get("contrast", 0.1)),
        shared_crop_aug=bool(acfg.get("shared_crop_aug", False)),
    )
    train_tf = DualViewTransform(image_size, True, aug, fft)
    eval_tf = DualViewTransform(image_size, False, aug, fft)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["data"].get("num_workers", 8)) if num_workers is None else int(num_workers)
    mpaths = manifests or _default_manifests(paths)
    sdir = splits_dir or _default_splits_dir(paths)

    def _loader(rows, *, train=False):
        sampler = make_balanced_sampler(rows, balance) if train else None
        return DataLoader(
            BaseDeepfakeDataset(rows, train_tf if train else eval_tf, image_size),
            batch_size=bs,
            sampler=sampler,
            shuffle=(train and sampler is None),
            num_workers=nw,
            pin_memory=True,
            drop_last=train,
            persistent_workers=(nw > 0),
            collate_fn=collate_batch,
            worker_init_fn=seed_worker if train else None,
        )

    # --- train / val (FaceForensics++) ---
    ffpp_path = mpaths["faceforensics_c23"]
    if not Path(ffpp_path).is_file():
        raise FileNotFoundError(
            f"FF++ manifest not found: {ffpp_path}. Run extraction + manifest for "
            "faceforensics_c23 first (scripts/preprocess_faces.py --build-manifest)."
        )
    ffpp_rows = read_manifest(ffpp_path)
    train_rows = apply_ffpp_split(ffpp_rows, sdir, "train")
    val_rows = apply_ffpp_split(ffpp_rows, sdir, "val")
    if not train_rows:
        raise RuntimeError(
            f"FF++ train split is empty. Check that {sdir}/train.json matches the "
            f"video ids in {ffpp_path} (source_video_id like 'Deepfakes__033_097')."
        )
    train_loader = _loader(train_rows, train=True)
    val_loader = _loader(val_rows)

    # --- eval-only datasets ---
    test_loaders = {}
    # in-domain FF++ test split (the ceiling row); identity-disjoint from train/val
    if include_indomain_test:
        test_rows = apply_ffpp_split(ffpp_rows, sdir, "test")
        if test_rows:
            test_loaders["faceforensics_c23"] = _loader(test_rows)
        elif verbose:
            print("[get_dataloaders] skip faceforensics_c23 in-domain test: empty test split")
    for name in TEST_DATASETS:
        p = mpaths.get(name)
        if not p or not Path(p).is_file():
            if verbose:
                print(f"[get_dataloaders] skip {name}: manifest not found ({p})")
            continue
        test_loaders[name] = _loader(read_manifest(p))

    # --- DF40 diffusion: one loader per subset x domain (fakes + domain reals) ---
    p = mpaths.get("df40_diffusion")
    if p and Path(p).is_file():
        for (subset, domain), rows in group_df40(read_manifest(p)).items():
            test_loaders[f"df40_{subset}_{domain}"] = _loader(rows)
    elif verbose:
        print(f"[get_dataloaders] skip df40_diffusion: manifest not found ({p})")

    if verbose:
        n_fake = sum(int(r["label"]) == FAKE for r in train_rows)
        print(f"[get_dataloaders] FF++ train={len(train_rows)} frames "
              f"(fake={n_fake}, real={len(train_rows) - n_fake}); val={len(val_rows)}")
        print(f"[get_dataloaders] eval loaders: {list(test_loaders)}")
    return train_loader, val_loader, test_loaders
