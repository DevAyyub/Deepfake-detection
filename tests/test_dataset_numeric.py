"""Numeric tests that need a real PyTorch install: the FFT-magnitude transform,
the dual-view transform, and the batch-contract checker run on genuine tensors.

The whole module is marked needs_real_torch, so it is skipped automatically when
the conftest torch stub is active (no torch installed) and runs on your box / a
torch-enabled CI job."""
import importlib.util
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.needs_real_torch

import torch  # noqa: E402  (real torch in the environments where this module runs)

from sfdet.data.dataset import AugConfig, DualViewTransform, FFTMagnitude  # noqa: E402


def test_fft_magnitude_shape_dtype_and_norm():
    x = torch.rand(3, 32, 32)                          # [3,H,W] in [0,1]
    out = FFTMagnitude(grayscale=True)(x)
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert abs(float(out.mean())) < 1e-4               # per-image zero-mean
    assert abs(float(out.std()) - 1.0) < 0.1           # per-image ~unit-std
    assert FFTMagnitude(grayscale=False)(x).shape == (3, 32, 32)


def test_fft_magnitude_constant_crop_is_finite():
    # a flat crop has zero spectral variance; the eps guard must avoid NaN/Inf
    out = FFTMagnitude(grayscale=True)(torch.full((3, 16, 16), 0.5))
    assert torch.isfinite(out).all()


def test_dualview_shapes_and_eval_determinism():
    from PIL import Image
    img = Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8"))
    fft = FFTMagnitude()
    train_tf = DualViewTransform(32, True, AugConfig(), fft)
    eval_tf = DualViewTransform(32, False, AugConfig(), fft)

    sp, fr = train_tf(img)
    assert sp.shape == (3, 32, 32) and fr.shape == (1, 32, 32)
    assert sp.dtype == torch.float32 and fr.dtype == torch.float32
    assert torch.isfinite(sp).all() and torch.isfinite(fr).all()

    e0, _ = eval_tf(img)
    e1, _ = eval_tf(img)
    assert torch.equal(e0, e1)                          # eval transform is deterministic


def _load_verify_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vd", root / "scripts" / "verify_dataloaders.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verify_batch_contract_on_real_tensors():
    vd = _load_verify_module()
    B, S = 4, 16
    good = {
        "spatial": torch.randn(B, 3, S, S), "frequency": torch.randn(B, 1, S, S),
        "label": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "source_video_id": ["a"] * B, "dataset": ["faceforensics_c23"] * B, "subset": ["real"] * B,
        "domain": [""] * B, "frame": ["0"] * B, "crop_path": ["p"] * B,
        "mask_path": [""] * B, "landmark_path": [""] * B,
    }
    ok, problems, _ = vd.verify_batch(good, S, 1)
    assert ok and problems == []

    bad = dict(good)
    bad["label"] = torch.tensor([0.0, 1.0, 2.0, 0.0])   # 2.0 is illegal
    assert not vd.verify_batch(bad, S, 1)[0]
