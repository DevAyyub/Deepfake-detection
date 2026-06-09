"""factory.py — config-driven model + forward-adapter dispatch for training/eval.

A model "variant" (``model.variant`` in the merged config) selects BOTH which network
to build and how a batch maps onto its forward call. Keeping the two together here is
what lets the trainer and evaluator stay model-agnostic: they never hard-code an input
signature, so switching architectures (the ablation arms) is a pure config swap.

    spatial_only   -> SpatialClassifier(spatial)              m(b["spatial"])
    frequency_only -> FrequencyClassifier(frequency)          m(b["frequency"])        [ABL3]
    full           -> DeepfakeDetector(spatial, frequency)    m(b["spatial"], b["frequency"])

All three builders already exist and read the same merged config; this module only
routes to them and pairs each with the matching forward adapter.
"""
from typing import Callable, Tuple

from torch import nn

from .spatial_branch import build_spatial_classifier
from .frequency_branch import build_frequency_classifier
from .model import build_detector


# --- forward adapters: the single place that knows each variant's input signature --- #
def _spatial_forward(model: nn.Module, batch: dict):
    return model(batch["spatial"])


def _frequency_forward(model: nn.Module, batch: dict):
    return model(batch["frequency"])


def _dual_forward(model: nn.Module, batch: dict):
    return model(batch["spatial"], batch["frequency"])


# variant name -> (builder, forward adapter)
_VARIANTS = {
    "spatial_only":   (build_spatial_classifier,   _spatial_forward),
    "frequency_only": (build_frequency_classifier, _frequency_forward),
    "full":           (build_detector,             _dual_forward),
}


def variant_names():
    """Sorted list of known variant names (for CLIs / error messages)."""
    return sorted(_VARIANTS)


def build_model(cfg: dict) -> Tuple[nn.Module, Callable, str]:
    """Build the model named by ``cfg['model']['variant']`` and return
    ``(model, forward_fn, variant)``.

    ``forward_fn(model, batch) -> logits [B]`` is the adapter the trainer/evaluator call
    so a single loop drives every variant; it pulls exactly the inputs that variant
    consumes from the batch dict. Defaults to ``full`` (the dual model) when no variant
    key is present.
    """
    variant = str((cfg or {}).get("model", {}).get("variant", "full")).lower()
    if variant not in _VARIANTS:
        raise ValueError(
            f"unknown model.variant '{variant}'; choose one of {variant_names()}")
    builder, forward_fn = _VARIANTS[variant]
    return builder(cfg), forward_fn, variant
