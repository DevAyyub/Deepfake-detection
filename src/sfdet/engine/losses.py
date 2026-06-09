"""sfdet.engine.losses — objective functions, kept out of the model class so the
loss is swappable from config without touching the architecture.

BCEWithLogitsLoss is the default: the model emits one logit per sample and labels
are {0,1} floats (see the batch contract), and BCE-with-logits is the numerically
stable form (it fuses the sigmoid). Class imbalance is already handled upstream by
the balanced sampler in the train loader, so no pos_weight is applied here (that
would double-correct).
"""
from __future__ import annotations

import torch.nn as nn


def build_loss(cfg: dict) -> nn.Module:
    name = ((cfg or {}).get("train", {}).get("loss", {}) or {}).get("name", "bce").lower()
    if name in ("bce", "bce_with_logits", "bcewithlogits"):
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"unknown loss '{name}' (only 'bce' is wired so far)")
