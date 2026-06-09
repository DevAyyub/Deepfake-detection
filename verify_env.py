#!/usr/bin/env python3
"""verify_env.py — environment sanity check for the dual spatial-frequency detector.

Run this immediately after `pip install -r requirements.txt && pip install -e .`
(locally and again on the remote GPU). It:

  1. prints Python / platform info,
  2. imports every core library and prints its version (collecting failures
     instead of crashing on the first one),
  3. reports CUDA availability, the torch-bundled CUDA version, and each visible
     GPU,
  4. runs a tiny EfficientNet-B4 forward pass (the spatial-branch backbone) and
     checks the output shape,
  5. runs a tiny 2D-FFT magnitude sanity check (the frequency-branch front end),
  6. prints a PASS/FAIL summary and exits non-zero if any CRITICAL check failed.

Usage:
    python verify_env.py                # fast, offline (random EfficientNet-B4 weights)
    python verify_env.py --pretrained   # also exercises the ImageNet weight download
    python verify_env.py --image-size 256
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys

# Core libraries whose absence should fail the check (the model/eval pipeline
# cannot run without these). Optional extras are checked but never fatal.
CRITICAL = ["torch", "torchvision", "timm", "numpy", "cv2", "sklearn"]
OPTIONAL = [
    "scipy",
    "PIL",
    "pandas",
    "matplotlib",
    "albumentations",
    "einops",
    "yaml",
    "tqdm",
    "tensorboard",
    "open_clip",
]

# Map import name -> the distribution whose __version__/metadata we report.
_VERSION_HINT = {
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "open_clip": "open-clip-torch",
}


def _version_of(mod_name: str, module) -> str:
    """Best-effort version string for an imported module."""
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(module, attr, None)
        if isinstance(v, str):
            return v
    # Fall back to installed-distribution metadata.
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version(_VERSION_HINT.get(mod_name, mod_name))
    except Exception:
        return "unknown"


def check_imports() -> tuple[list[str], list[str]]:
    """Import every library; return (critical_failures, optional_failures)."""
    crit_fail: list[str] = []
    opt_fail: list[str] = []

    print("-- Library imports " + "-" * 41)
    for name in CRITICAL + OPTIONAL:
        try:
            module = importlib.import_module(name)
            print(f"  [ ok ] {name:<16} {_version_of(name, module)}")
        except Exception as exc:  # noqa: BLE001 - we want to report, not crash
            bucket = crit_fail if name in CRITICAL else opt_fail
            bucket.append(name)
            tag = "FAIL" if name in CRITICAL else "warn"
            print(f"  [{tag}] {name:<16} import error: {exc}")
    return crit_fail, opt_fail


def check_devices() -> bool:
    """Report CUDA/device info. Returns True if a CUDA device is usable."""
    print("\n-- Compute devices " + "-" * 41)
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] torch unavailable: {exc}")
        return False

    print(f"  torch              {torch.__version__}")
    print(f"  torch CUDA build   {torch.version.cuda}")
    print(f"  cuDNN              {torch.backends.cudnn.version()}")

    available = torch.cuda.is_available()
    print(f"  cuda.is_available  {available}")
    if not available:
        print("  [warn] No CUDA device visible — fine for local CPU dev, "
              "but training must run on the remote GPU.")
        return False

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} (sm_{cap[0]}{cap[1]}, {total_gb:.1f} GiB)")
    return True


def check_efficientnet(image_size: int, pretrained: bool, use_cuda: bool) -> bool:
    """Build EfficientNet-B4 (spatial backbone) and run one forward pass."""
    print("\n-- Spatial branch (EfficientNet-B4) " + "-" * 24)
    try:
        import timm
        import torch

        device = torch.device("cuda" if use_cuda else "cpu")
        # num_classes=2 -> binary real/fake head; weights random unless --pretrained.
        model = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=2
        ).to(device).eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        dummy = torch.randn(2, 3, image_size, image_size, device=device)
        with torch.no_grad():
            out = model(dummy)

        ok = tuple(out.shape) == (2, 2)
        print(f"  params             {n_params:.1f}M")
        print(f"  pretrained         {pretrained}")
        print(f"  input              {tuple(dummy.shape)}  on {device.type}")
        print(f"  logits             {tuple(out.shape)}  (expected (2, 2))")
        print(f"  [{'ok' if ok else 'FAIL'}] forward pass")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] EfficientNet-B4 forward failed: {exc}")
        return False


def check_fft_frontend(image_size: int, use_cuda: bool) -> bool:
    """Sanity-check the frequency-branch front end: 2D FFT -> shifted log-magnitude.

    This is only a numerical/shape check of the transform that will feed the
    frequency branch — not the branch itself (that is implemented later).
    """
    print("\n-- Frequency branch front end (2D FFT magnitude) " + "-" * 11)
    try:
        import torch

        device = torch.device("cuda" if use_cuda else "cpu")
        # A single grayscale-like crop: (B, 1, H, W).
        x = torch.randn(2, 1, image_size, image_size, device=device)
        spec = torch.fft.fft2(x, norm="ortho")          # complex spectrum
        spec = torch.fft.fftshift(spec, dim=(-2, -1))    # DC -> centre, so (rho, theta) is centred
        mag = torch.log1p(spec.abs())                    # log-magnitude (what the CNN will see)

        ok = (
            tuple(mag.shape) == (2, 1, image_size, image_size)
            and torch.isfinite(mag).all().item()
        )
        print(f"  input              {tuple(x.shape)}  on {device.type}")
        print(f"  log-magnitude      {tuple(mag.shape)}  finite={torch.isfinite(mag).all().item()}")
        print(f"  [{'ok' if ok else 'FAIL'}] FFT magnitude transform")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] FFT front-end check failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="download ImageNet weights for EfficientNet-B4 (tests the download path)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="square input size for the forward-pass checks (matches data.image_size = 256)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Environment verification — dual spatial-frequency detector")
    print("=" * 60)
    print(f"Python   {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"Platform {platform.platform()}")

    crit_fail, _opt_fail = check_imports()

    # If core libs are missing there's no point running the forward passes.
    if crit_fail:
        print("\n" + "=" * 60)
        print(f"RESULT: FAIL — missing critical libraries: {', '.join(crit_fail)}")
        print("Fix: pip install -r requirements.txt  (and  pip install -e .)")
        print("=" * 60)
        return 1

    has_cuda = check_devices()
    enet_ok = check_efficientnet(args.image_size, args.pretrained, has_cuda)
    fft_ok = check_fft_frontend(args.image_size, has_cuda)

    print("\n" + "=" * 60)
    forward_ok = enet_ok and fft_ok
    if forward_ok:
        gpu_note = "GPU detected" if has_cuda else "CPU only (no GPU visible)"
        print(f"RESULT: PASS — core libraries import and forward passes run. [{gpu_note}]")
        # Not having a GPU locally is expected; it's only a hard requirement on
        # the remote, so it does not fail the check here.
        print("=" * 60)
        return 0

    failed = [n for n, ok in (("EfficientNet-B4", enet_ok), ("FFT front end", fft_ok)) if not ok]
    print(f"RESULT: FAIL — forward-pass check(s) failed: {', '.join(failed)}")
    print("=" * 60)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
