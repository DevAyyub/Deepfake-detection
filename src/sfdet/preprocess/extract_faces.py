#!/usr/bin/env python3
"""extract_faces.py — dlib detection + 81-landmark alignment + cropping.

Runs in the ISOLATED preprocess env (requirements-preprocess.txt: dlib, opencv,
scikit-image, numpy, tqdm). Produces aligned face crops for the three VIDEO
datasets (FaceForensics++ c23, Celeb-DF v2, DFDC) plus a per-video extraction log
(``_extraction_log.jsonl``) that manifest.py turns into the unified manifest.
WildDeepfake and DF40 ship pre-cropped faces and are NOT processed here.

WHY THIS EXACT PIPELINE (notes 1.8). The crops feed a frequency branch, and the
DF40 diffusion crops we evaluate against were produced by DeepfakeBench's
preprocessing, which we cannot re-run. Matching it keeps the FFT spectra
comparable across all five datasets and matches every 256^2 baseline. This is a
faithful re-implementation of DeepfakeBench's ``img_align_crop``:

  1. dlib HOG frontal detector on the grayscale frame.
  2. dlib 81-point predictor; five keypoints are taken from the standard indices
     (parts 37/44/30/49/55 = left-eye, right-eye, nose, left-mouth, right-mouth)
     exactly as DeepfakeBench's ``get_keypts`` does.
  3. The largest face is kept. (Datasets are mostly single-subject; DFDC has
     multi-face frames where the fake identity is not known at the video-level
     label, so largest-face is the standard, documented compromise.)
  4. ONE resample: a skimage SimilarityTransform maps the 5 keypoints onto the
     ArcFace 5-point template (scaled to crop_size, with a margin of scale-1), and
     cv2.warpAffine applies it once. Border is BORDER_CONSTANT (black) to MATCH
     DF40's crops. (Reflect padding is gentler on the FFT and is available via
     --border reflect as a frequency-hygiene ABLATION, but the headline run
     matches the un-redoable DF40 crops.)
  5. Crops are written as PNG (lossless) so no JPEG artifacts re-enter the spectrum.

align=False is the minimal-resampling path for the alignment ablation: a square
enlarged-bbox crop with no rotation/warp.

Missing faces are skipped, never blank-filled (a near-zero image has a degenerate
all-DC spectrum). Videos that yield zero faces are recorded and excluded, not
silently dropped. For FF++ fakes, the manipulation-mask video is warped with the
SAME transform and saved alongside each crop (for the E5 GT-mask faithfulness
check); landmarks are saved per crop when enabled.

Run (in the preprocess env):
    python -m sfdet.preprocess.extract_faces \
        --dataset faceforensics_c23 \
        --data-root /data/FaceForensics++/c23 \
        --crops-root /data/crops \
        --predictor-path dlib_tools/shape_predictor_81_face_landmarks.dat
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv is required for face extraction (preprocess env).\n"
        "Install:  pip install -r requirements-preprocess.txt\n"
        f"(import error: {exc})"
    )

try:
    import dlib
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "dlib is required for face extraction and is part of the ISOLATED preprocess\n"
        "env (matches DeepfakeBench/DF40). Install it there:\n"
        "    pip install -r requirements-preprocess.txt\n"
        f"(import error: {exc})"
    )

try:
    from skimage.transform import SimilarityTransform
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-image is required for the alignment (same SimilarityTransform math as\n"
        "DeepfakeBench). Install the preprocess env:\n"
        "    pip install -r requirements-preprocess.txt\n"
        f"(import error: {exc})"
    )

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_):
        return it


# --- DeepfakeBench-matching constants ---------------------------------------- #
# Five keypoints, taken from the SAME dlib indices DeepfakeBench's get_keypts uses
# (the 81-point predictor keeps the classic 68-point indices; parts 68-80 are the
# extra forehead points, unused here). Order: L-eye, R-eye, nose, L-mouth, R-mouth.
KP_PART_INDICES = (37, 44, 30, 49, 55)

# ArcFace/insightface 5-point reference template, defined at 112x112, SAME order as
# KP_PART_INDICES. Used by DeepfakeBench's img_align_crop.
ARCFACE_112 = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)

INTERP = {
    "bilinear": cv2.INTER_LINEAR,     # DeepfakeBench default (warpAffine default)
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "nearest": cv2.INTER_NEAREST,
}
BORDER = {
    "constant": cv2.BORDER_CONSTANT,  # black fill — MATCHES DF40 (headline)
    "reflect": cv2.BORDER_REFLECT_101,  # gentler on the FFT — frequency-hygiene ABLATION
    "replicate": cv2.BORDER_REPLICATE,
}

REAL, FAKE = 0, 1                      # fake = positive class (AUC convention)
LABEL_NAME = {REAL: "real", FAKE: "fake"}

VIDEO_DATASETS = ("faceforensics_c23", "celebdf_v2", "dfdc")
DEFAULT_MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


@dataclass
class ExtractConfig:
    """Every knob an ablation might touch. Defaults MATCH DeepfakeBench/DF40 and
    configs/base.yaml's `preprocess` block."""
    # detector / landmarks
    predictor_path: str = "dlib_tools/shape_predictor_81_face_landmarks.dat"
    det_upsample: int = 1               # dlib pyramid upsample; >0 finds smaller faces (slower)
    det_threshold: float = 0.0          # dlib HOG score threshold; raise to reject weak detections
    face_select: str = "largest"        # "largest" or "confident" when several faces are found
    # crop geometry
    crop_size: int = 256                # == data.image_size; matches DeepfakeBench/DF40
    align: bool = True                  # True: ArcFace similarity-transform align (match DF40)
    #                                     False: square enlarged-bbox crop, no rotation (ablation)
    scale: float = 1.3                  # ArcFace-template margin (margin_rate = scale-1 = 0.3)
    enlarge: float = 1.3                # unaligned-mode bbox expansion (FF++ 1.3x convention)
    interp: str = "bilinear"            # the SINGLE documented interpolation for the one resample
    border: str = "constant"            # "constant" (black; match DF40) | "reflect" (ablation)
    # frame sampling (mirrors DeepfakeBench preprocessing/config.yaml)
    mode: str = "fixed_num_frames"      # or "fixed_stride"
    num_frames: int = 32
    stride: int = 10
    # extra artifacts
    save_masks: bool = True             # warp+save aligned manipulation masks (FF++ fakes) for E5
    save_landmarks: bool = True         # save the 81 raw landmarks (.npy) per crop


class FaceExtractor:
    """dlib detect -> 81 landmarks -> single-resample aligned crop.

    One instance is reused across a dataset (loads detector + landmark model once).
    Call it on a BGR frame; returns ``(crop, M, landmarks81)`` or ``None`` if no
    face is detected. ``M`` (the 2x3 affine actually applied) is returned so the
    caller can warp the manipulation mask with the identical transform.
    """

    def __init__(self, cfg: ExtractConfig):
        self.cfg = cfg
        if cfg.interp not in INTERP:
            raise ValueError(f"interp must be one of {list(INTERP)}; got {cfg.interp!r}")
        if cfg.border not in BORDER:
            raise ValueError(f"border must be one of {list(BORDER)}; got {cfg.border!r}")
        if not Path(cfg.predictor_path).is_file():
            raise SystemExit(
                f"Landmark model not found: {cfg.predictor_path}\n"
                "Download shape_predictor_81_face_landmarks.dat (linked from the "
                "DeepfakeBench repo, ./preprocessing/dlib_tools) and pass --predictor-path."
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(cfg.predictor_path)
        self._interp = INTERP[cfg.interp]
        self._border = BORDER[cfg.border]
        self._dst = self._build_template()   # cached destination template (crop_size, +margin)

    # -- template -----------------------------------------------------------
    def _build_template(self) -> np.ndarray:
        """ArcFace template scaled to crop_size with a (scale-1) margin — exactly the
        sequence in DeepfakeBench's img_align_crop."""
        s = float(self.cfg.crop_size)
        dst = ARCFACE_112.copy()
        dst[:, 0] += 8.0                       # insightface 112x96 -> 112 x-offset (ref target = 112)
        dst[:, 0] *= s / 112.0
        dst[:, 1] *= s / 112.0
        margin = self.cfg.scale - 1.0
        xm = s * margin / 2.0
        ym = s * margin / 2.0
        dst[:, 0] += xm
        dst[:, 1] += ym
        dst[:, 0] *= s / (s + 2.0 * xm)
        dst[:, 1] *= s / (s + 2.0 * ym)
        return dst.astype(np.float32)

    # -- detection / landmarks ----------------------------------------------
    def _detect(self, gray: np.ndarray):
        rects, scores, _ = self.detector.run(gray, self.cfg.det_upsample, self.cfg.det_threshold)
        if not rects:
            return None
        if self.cfg.face_select == "confident":
            return rects[int(np.argmax(scores))]
        return max(rects, key=lambda r: r.width() * r.height())   # default: largest

    @staticmethod
    def _all_landmarks(shape) -> np.ndarray:
        return np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)],
            dtype=np.float32,
        )

    @staticmethod
    def _five_keypoints(shape) -> np.ndarray:
        return np.array(
            [[shape.part(i).x, shape.part(i).y] for i in KP_PART_INDICES],
            dtype=np.float32,
        )

    # -- transforms (each returns a 2x3 affine; warp is applied exactly ONCE) --
    def _aligned_matrix(self, kp5: np.ndarray) -> np.ndarray:
        tform = SimilarityTransform()
        tform.estimate(kp5, self._dst)         # similarity: rotation + uniform scale + translation
        return tform.params[0:2, :]

    def _bbox_matrix(self, lm: np.ndarray) -> np.ndarray:
        x0, y0 = lm.min(axis=0)
        x1, y1 = lm.max(axis=0)
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        side = max(x1 - x0, y1 - y0) * self.cfg.enlarge              # square + enlarge
        scale = self.cfg.crop_size / max(side, 1e-6)
        return np.array(
            [[scale, 0.0, self.cfg.crop_size * 0.5 - scale * cx],
             [0.0, scale, self.cfg.crop_size * 0.5 - scale * cy]],
            dtype=np.float64,
        )

    def _warp(self, img: np.ndarray, M: np.ndarray, *, interp=None, border=None,
              border_value=0) -> np.ndarray:
        return cv2.warpAffine(
            img, M, (self.cfg.crop_size, self.cfg.crop_size),
            flags=interp if interp is not None else self._interp,
            borderMode=border if border is not None else self._border,
            borderValue=border_value,
        )

    def __call__(self, frame_bgr: np.ndarray):
        """Return (crop_bgr, M, landmarks81) or None."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rect = self._detect(gray)
        if rect is None:
            return None
        shape = self.predictor(gray, rect)
        lm81 = self._all_landmarks(shape)
        if self.cfg.align:
            M = self._aligned_matrix(self._five_keypoints(shape))
        else:
            M = self._bbox_matrix(lm81)
        crop = self._warp(frame_bgr, M)         # ONE resample
        return crop, M, lm81

    def warp_mask(self, mask_bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Warp a manipulation mask with the SAME M. Binary content -> nearest +
        black (0) border, so out-of-frame is 'not manipulated'."""
        if mask_bgr.ndim == 3:
            mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        return self._warp(mask_bgr, M, interp=cv2.INTER_NEAREST,
                          border=cv2.BORDER_CONSTANT, border_value=0)


# --------------------------------------------------------------------------- #
# Frame sampling
# --------------------------------------------------------------------------- #
def target_frame_indices(n_total: int, cfg: ExtractConfig) -> list:
    """Indices to sample. fixed_num_frames -> uniform across the clip (reproducible;
    DeepfakeBench samples 32); fixed_stride -> every `stride`-th frame. Never
    consecutive (avoids near-duplicate frames)."""
    if n_total <= 0:
        return []
    if cfg.mode == "fixed_stride":
        return list(range(0, n_total, max(cfg.stride, 1)))
    if cfg.mode != "fixed_num_frames":
        raise ValueError(f"mode must be fixed_num_frames or fixed_stride; got {cfg.mode!r}")
    if n_total <= cfg.num_frames:
        return list(range(n_total))
    return [int(i) for i in np.linspace(0, n_total - 1, cfg.num_frames)]


def extract_video(video_path: Path, out_dir: Path, extractor: FaceExtractor,
                  cfg: ExtractConfig, mask_video_path: Optional[Path] = None) -> dict:
    """Sample frames from one video, write crops (+ aligned masks + landmarks),
    return a per-video record. Crops/masks/landmarks for a sampled frame are
    co-located in out_dir as <idx>.png / <idx>_mask.png / <idx>.npy."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"opened": False, "n_sampled": 0, "n_faces": 0, "crops": []}
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = set(target_frame_indices(n_total, cfg))

    mask_cap = None
    if mask_video_path is not None and cfg.save_masks and Path(mask_video_path).is_file():
        mc = cv2.VideoCapture(str(mask_video_path))
        mask_cap = mc if mc.isOpened() else None

    crops: list = []
    n_sampled = 0
    idx = 0
    max_t = max(targets) if targets else -1
    # Sequential read with early stop: frame-accurate seeking is not guaranteed
    # across codecs, so we read in order and act on target indices. The mask video
    # (if any) is read in lockstep to stay frame-aligned with the face video.
    while idx <= max_t:
        ok, frame = cap.read()
        mask_frame = None
        if mask_cap is not None:
            m_ok, m_frame = mask_cap.read()
            mask_frame = m_frame if m_ok else None
        if not ok:
            break
        if idx in targets:
            n_sampled += 1
            result = extractor(frame)
            if result is not None:
                crop, M, lm81 = result
                out_dir.mkdir(parents=True, exist_ok=True)
                crop_fp = out_dir / f"{idx:06d}.png"            # PNG = lossless
                cv2.imwrite(str(crop_fp), crop)
                rec = {"frame": idx, "crop_path": str(crop_fp.resolve()),
                       "mask_path": "", "landmark_path": ""}
                if cfg.save_landmarks:
                    lm_fp = out_dir / f"{idx:06d}.npy"
                    np.save(str(lm_fp), lm81)
                    rec["landmark_path"] = str(lm_fp.resolve())
                if mask_frame is not None:
                    mask_fp = out_dir / f"{idx:06d}_mask.png"
                    cv2.imwrite(str(mask_fp), extractor.warp_mask(mask_frame, M))
                    rec["mask_path"] = str(mask_fp.resolve())
                crops.append(rec)
        idx += 1
    cap.release()
    if mask_cap is not None:
        mask_cap.release()
    return {"opened": True, "n_sampled": n_sampled, "n_faces": len(crops), "crops": crops}


# --------------------------------------------------------------------------- #
# Per-dataset video enumeration + label assignment
# --------------------------------------------------------------------------- #
def _load_celebdf_test_list(path: Path) -> Optional[set]:
    if not path.is_file():
        print(f"WARNING: Celeb-DF test list not found at {path}; processing ALL videos. "
              "The standard protocol evaluates ONLY the official 518-video test split.",
              file=sys.stderr)
        return None
    keep = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            keep.add(line.split()[-1].replace("\\", "/"))     # last token is the video path
    return keep


def _load_dfdc_labels(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"DFDC labels file not found: {path}\n"
            "DFDC labels come from a CSV (filename,label) shipped with the test set; "
            "pass --dfdc-labels."
        )
    mapping: dict = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)                                    # header
        for row in reader:
            if len(row) < 2:
                continue
            fn, lab = row[0].strip(), row[1].strip().lower()
            mapping[fn] = FAKE if lab in ("fake", "1", "1.0") else REAL
    return mapping


def _ffpp_mask_path(root: Path, method: str, comp: str, stem: str) -> Optional[Path]:
    """FF++ manipulation-mask video for a fake clip, if present."""
    mp = root / "manipulated_sequences" / method / "masks" / "videos" / f"{stem}.mp4"
    return mp if mp.is_file() else None


def enumerate_videos(dataset: str, root: Path, *, comp: str = "c23",
                     manipulations: Optional[list] = None,
                     celebdf_test_list: Optional[Path] = None,
                     dfdc_labels: Optional[Path] = None
                     ) -> Iterator[tuple]:
    """Yield (video_path, label, subset, source_video_id, mask_video_path) for a
    video dataset. mask_video_path is set only for FF++ fakes (else None)."""
    root = Path(root)
    if dataset == "faceforensics_c23":
        manips = manipulations or DEFAULT_MANIPULATIONS
        real_dir = root / "original_sequences" / "youtube" / comp / "videos"
        for vp in sorted(real_dir.glob("*.mp4")):
            yield vp, REAL, "original", vp.stem, None        # real has no manipulation mask
        for m in manips:
            mdir = root / "manipulated_sequences" / m / comp / "videos"
            for vp in sorted(mdir.glob("*.mp4")):
                # video ids repeat across manipulations (same identity pair) -> prefix
                yield vp, FAKE, m, f"{m}__{vp.stem}", _ffpp_mask_path(root, m, comp, vp.stem)
    elif dataset == "celebdf_v2":
        test = _load_celebdf_test_list(celebdf_test_list or root / "List_of_testing_videos.txt")
        for sub, label in (("Celeb-real", REAL), ("YouTube-real", REAL),
                           ("Celeb-synthesis", FAKE)):
            for vp in sorted((root / sub).glob("*.mp4")):
                if test is not None and f"{sub}/{vp.name}" not in test:
                    continue                                  # keep only official test videos
                yield vp, label, sub, vp.stem, None
    elif dataset == "dfdc":
        labels = _load_dfdc_labels(dfdc_labels) if dfdc_labels else None
        if labels is None:
            raise SystemExit("DFDC is label-from-CSV; pass --dfdc-labels (filename,label).")
        candidates = sorted((root / "test").glob("*.mp4")) or sorted(root.glob("*.mp4"))
        for vp in candidates:
            label = labels.get(vp.name)
            if label is None:                                 # no label -> can't evaluate
                print(f"WARNING: no DFDC label for {vp.name}; skipping.", file=sys.stderr)
                continue
            yield vp, label, "dfdc", vp.stem, None
    else:
        raise ValueError(
            f"{dataset!r} is not a video dataset. WildDeepfake and DF40 ship pre-cropped "
            "faces and are handled directly by manifest.py, not extracted here."
        )


def run_extraction(dataset: str, data_root: str, crops_root: str, cfg: ExtractConfig, *,
                   comp: str = "c23", manipulations: Optional[list] = None,
                   celebdf_test_list: Optional[str] = None,
                   dfdc_labels: Optional[str] = None, limit: Optional[int] = None) -> Path:
    """Extract one video dataset to crops_root/<dataset>/ and write an extraction log."""
    if dataset not in VIDEO_DATASETS:
        raise ValueError(f"{dataset!r} is not one of {VIDEO_DATASETS}")
    out_root = Path(crops_root) / dataset
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "_extraction_log.jsonl"
    extractor = FaceExtractor(cfg)

    videos = list(enumerate_videos(
        dataset, Path(data_root), comp=comp, manipulations=manipulations,
        celebdf_test_list=Path(celebdf_test_list) if celebdf_test_list else None,
        dfdc_labels=Path(dfdc_labels) if dfdc_labels else None,
    ))
    if limit:
        videos = videos[:limit]
    if not videos:
        raise SystemExit(f"No videos found for {dataset} under {data_root}. Check the layout.")

    # Resume support. A prior (interrupted) run leaves its completed videos in the log;
    # read them, skip them, and APPEND new results — so re-running the SAME command
    # continues where it left off instead of restarting from video 1. A video's log line
    # is written only after all its crops are saved, so each video is either fully logged
    # (skip it) or absent (redo it); flushing per line makes that durable across a normal
    # shutdown / Ctrl+C. To force a clean re-run instead, delete crops_root/<dataset> first.
    done = set()
    if log_path.is_file():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["video_path"])
                except (ValueError, KeyError):
                    continue
    todo = [v for v in videos if str(v[0]) not in done]
    n_skip = len(videos) - len(todo)
    if n_skip:
        print(f"[{dataset}] resuming: {n_skip} videos already done, {len(todo)} remaining")

    n_zero = 0
    with open(log_path, "a") as logf:
        for vp, label, subset, vid, mask_vp in tqdm(todo, desc=f"extract {dataset}"):
            rec = extract_video(vp, out_root / LABEL_NAME[label] / vid, extractor, cfg,
                                mask_video_path=mask_vp)
            n_zero += rec["n_faces"] == 0
            logf.write(json.dumps({
                "dataset": dataset, "source_video_id": vid, "label": int(label),
                "subset": subset, "domain": "", "video_path": str(vp),
                "n_sampled": rec["n_sampled"], "n_faces": rec["n_faces"],
                "opened": rec["opened"], "excluded_zero_faces": rec["n_faces"] == 0,
                "crops": rec["crops"],
            }) + "\n")
            logf.flush()  # durable per-video so a shutdown loses at most the in-progress one
    print(f"[{dataset}] processed {len(todo)} this run ({n_skip} skipped)  "
          f"zero-face(excluded)={n_zero}  log={log_path}")
    print(f"   next: python -m sfdet.preprocess.manifest --dataset {dataset} "
          f"--crops-root {crops_root}")
    return log_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="dlib face extraction for the three video datasets.")
    p.add_argument("--dataset", required=True, choices=VIDEO_DATASETS)
    p.add_argument("--data-root", required=True,
                   help="dataset root (copy the value from your paths.yaml)")
    p.add_argument("--crops-root", required=True,
                   help="where crops are written (paths.yaml crops_root)")
    p.add_argument("--predictor-path", default=ExtractConfig.predictor_path)
    p.add_argument("--crop-size", type=int, default=ExtractConfig.crop_size)
    p.add_argument("--no-align", action="store_true",
                   help="ablation: square enlarged-bbox crop, no rotation/warp")
    p.add_argument("--scale", type=float, default=ExtractConfig.scale,
                   help="ArcFace-template margin (margin_rate = scale-1); DeepfakeBench uses 1.3")
    p.add_argument("--interp", default=ExtractConfig.interp, choices=list(INTERP))
    p.add_argument("--border", default=ExtractConfig.border, choices=list(BORDER),
                   help="warp border: 'constant' (black; match DF40) or 'reflect' (FFT-hygiene ablation)")
    p.add_argument("--enlarge", type=float, default=ExtractConfig.enlarge)
    p.add_argument("--det-upsample", type=int, default=ExtractConfig.det_upsample)
    p.add_argument("--det-threshold", type=float, default=ExtractConfig.det_threshold)
    p.add_argument("--mode", default=ExtractConfig.mode,
                   choices=["fixed_num_frames", "fixed_stride"])
    p.add_argument("--num-frames", type=int, default=ExtractConfig.num_frames)
    p.add_argument("--stride", type=int, default=ExtractConfig.stride)
    p.add_argument("--no-masks", action="store_true",
                   help="do not save aligned manipulation masks (FF++)")
    p.add_argument("--no-landmarks", action="store_true",
                   help="do not save per-crop 81-landmark .npy files")
    p.add_argument("--comp", default="c23", choices=["raw", "c23", "c40"],
                   help="FaceForensics++ compression (c23 only for the headline run)")
    p.add_argument("--manipulations", nargs="+", default=None,
                   help="FF++ fake methods (default: the standard four)")
    p.add_argument("--celebdf-test-list", default=None,
                   help="path to Celeb-DF List_of_testing_videos.txt (default: <root>/...)")
    p.add_argument("--dfdc-labels", default=None,
                   help="DFDC labels CSV (filename,label) — required for dfdc")
    p.add_argument("--limit", type=int, default=None, help="process only N videos (smoke test)")
    return p


def main(argv: Optional[list] = None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = ExtractConfig(
        predictor_path=args.predictor_path, det_upsample=args.det_upsample,
        det_threshold=args.det_threshold, crop_size=args.crop_size,
        align=not args.no_align, scale=args.scale, enlarge=args.enlarge,
        interp=args.interp, border=args.border, mode=args.mode,
        num_frames=args.num_frames, stride=args.stride,
        save_masks=not args.no_masks, save_landmarks=not args.no_landmarks,
    )
    run_extraction(
        args.dataset, args.data_root, args.crops_root, cfg, comp=args.comp,
        manipulations=args.manipulations, celebdf_test_list=args.celebdf_test_list,
        dfdc_labels=args.dfdc_labels, limit=args.limit,
    )


if __name__ == "__main__":
    main()
