"""pytest config + a minimal torch/torchvision/PIL stub for environments without
a real PyTorch install.

Why: the pure-logic tests import sfdet.data.dataset, which imports
torch/torchvision/PIL at module load. We want those tests to run EVERYWHERE —
the dev sandbox and fast CI with no torch installed — so when real torch is
absent we register stubs that satisfy the imports and the torch-free code paths
those tests actually exercise (FF++ splits, balancing, DF40 grouping, the
get_dataloaders wiring — none of which call tensor math).

Numeric tests that need genuine tensors (real FFT, transforms) are marked with
@pytest.mark.needs_real_torch and are skipped automatically while the stub is
active. On your box / a torch-enabled CI job, real torch is used and they run.
"""
import sys
import types

import pytest

try:
    import torch  # noqa: F401
    STUBBED_TORCH = False
except ImportError:
    STUBBED_TORCH = True

    def _module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    def _passthrough(x, *_, **__):
        return x

    class _StubDataset:                       # base class for BaseDeepfakeDataset
        pass

    class _StubDataLoader:                     # records what get_dataloaders configured
        def __init__(self, dataset, **kw):
            self.dataset = dataset
            self.__dict__.update(kw)           # batch_size, sampler, shuffle, drop_last, ...

    class _StubWRS:                            # WeightedRandomSampler
        def __init__(self, weights, num_samples=None, replacement=True, generator=None, **kw):
            self.weights = list(weights)
            self.num_samples = num_samples if num_samples is not None else len(self.weights)
            self.replacement = replacement

        def __len__(self):
            return self.num_samples

    class _StubGenerator:
        def manual_seed(self, *_):
            return self

    class _StubImage:                          # PIL.Image
        @staticmethod
        def open(*_, **__):
            return None

        @staticmethod
        def fromarray(*_, **__):
            return None

    torch = _module(
        "torch",
        float32="float32", float64="float64", double="double",
        tensor=lambda *a, **k: (a[0] if a else None),
        stack=lambda xs, *a, **k: list(xs),
        as_tensor=lambda x, *a, **k: x,
        log1p=_passthrough,
        isfinite=lambda t: types.SimpleNamespace(all=lambda: True),
        equal=lambda a, b: a is b,
        initial_seed=lambda: 0,
        Generator=_StubGenerator,
    )
    _module("torch.fft", fftshift=_passthrough, fft2=_passthrough)
    torch.fft = sys.modules["torch.fft"]
    _tud = _module("torch.utils.data", Dataset=_StubDataset,
                   DataLoader=_StubDataLoader, WeightedRandomSampler=_StubWRS)
    _tu = _module("torch.utils")
    _tu.data = _tud
    torch.utils = _tu

    _tvf = _module("torchvision.transforms.functional",
                   to_tensor=_passthrough, resize=_passthrough, hflip=_passthrough,
                   adjust_brightness=_passthrough, adjust_contrast=_passthrough,
                   normalize=_passthrough)
    _tvt = _module("torchvision.transforms")
    _tvt.functional = _tvf
    _tv = _module("torchvision")
    _tv.transforms = _tvt

    _module("PIL", Image=_StubImage)           # `from PIL import Image` -> _StubImage


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_real_torch: requires a real PyTorch install; skipped under the torch stub",
    )


def pytest_collection_modifyitems(config, items):
    if not STUBBED_TORCH:
        return
    skip = pytest.mark.skip(reason="needs a real PyTorch install (torch stub active)")
    for item in items:
        if "needs_real_torch" in item.keywords:
            item.add_marker(skip)
