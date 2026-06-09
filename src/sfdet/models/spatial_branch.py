"""spatial_branch.py — the spatial encoder of the dual spatial-frequency detector.

EfficientNet-B4 (ImageNet-pretrained, via timm). This is deliberately a standard,
off-the-shelf encoder: the dual-branch architecture (C1) exists to ENABLE the
frequency attribution, it is not claimed as a standalone novelty, so nothing
here is exotic.

Two classes:
  SpatialBranch      the encoder. forward(x) -> the last /32 convolutional
                     feature map [B, C, H/32, W/32] (C = backbone width = 1792 for
                     B4). This single map is BOTH (a) the fusion query source and
                     (b) the Grad-CAM target, so the spatial explanation is over
                     exactly the representation that enters fusion. It is
                     modality-pure: no frequency information is mixed in here.
  SpatialClassifier  thin standalone wrapper: global average pool -> dropout ->
                     linear -> one real/fake logit. This IS the spatial-only model
                     trained for E1's spatial-only ablation arm. The same
                     SpatialBranch is reused unchanged by the full detector later
                     (detector.py); only the head differs across variants.

Input: the batch contract's ``spatial`` tensor, [B, 3, 256, 256], ImageNet-
normalized. Because the data pipeline bakes in ImageNet mean/std, the backbone
weights must expect the SAME normalization. SpatialBranch asserts this against
the data pipeline's constants when pretrained=True: timm's ported
``tf_efficientnet_b4*`` use inception 0.5/0.5 and would silently degrade
transfer, whereas the native ``efficientnet_b4`` uses ImageNet stats.

Grad-CAM itself lives in sfdet.explain (step 10). Here we only EXPOSE the target
(the feature map and the module that produces it); the hook attaches there.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from sfdet.data.dataset import IMAGENET_MEAN, IMAGENET_STD  # single source of truth for input norm

# timm-native weights use ImageNet mean/std (NOT the 0.5/0.5 of the tf_ ports).
DEFAULT_BACKBONE = "efficientnet_b4"


class SpatialBranch(nn.Module):
    """EfficientNet-B4 trunk that exposes its last /32 feature map.

    forward(x: [B,3,H,W]) -> feature map [B, out_channels, H/32, W/32].

    The pooled vector and the classification head are intentionally NOT here —
    they live in SpatialClassifier (standalone) and classifier_head.py (fused),
    so this trunk is shared verbatim across the spatial-only and dual variants.
    """

    REDUCTION = 32

    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 drop_path_rate: float = 0.0, in_chans: int = 3,
                 check_normalization: bool = True):
        super().__init__()
        # num_classes=0 + global_pool='' makes forward return the feature MAP
        # (timm's forward_features output, after conv_head) rather than a pooled
        # vector — that map is the fusion query and the Grad-CAM target.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="",
            drop_path_rate=drop_path_rate, in_chans=in_chans,
        )
        # Read the width from timm (1792 for B4) rather than hardcoding it.
        self.out_channels = int(self.backbone.num_features)

        if pretrained and check_normalization and in_chans == 3:
            self._assert_normalization_matches()

    def _assert_normalization_matches(self) -> None:
        """Guard the silent-degradation bug: the backbone's expected input
        normalization must match what the data pipeline applied."""
        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        mean, std = cfg.get("mean"), cfg.get("std")
        if mean is None or std is None:
            return
        tol = 1e-3
        ok = (all(abs(a - b) < tol for a, b in zip(mean, IMAGENET_MEAN))
              and all(abs(a - b) < tol for a, b in zip(std, IMAGENET_STD)))
        if not ok:
            raise ValueError(
                f"Normalization mismatch: backbone '{cfg.get('architecture', '?')}' expects "
                f"mean={tuple(round(m, 3) for m in mean)} std={tuple(round(s, 3) for s in std)}, "
                f"but the data pipeline normalizes with mean={IMAGENET_MEAN} std={IMAGENET_STD}. "
                "Use a variant with ImageNet stats (e.g. 'efficientnet_b4'), or set the data "
                "pipeline's normalization from the backbone's pretrained_cfg. Mismatched "
                "normalization silently degrades the pretrained features."
            )

    @property
    def gradcam_target(self) -> nn.Module:
        """Module whose output is the tapped /32 feature map — the Grad-CAM target
        the step-10 hook attaches to. For timm EfficientNet this is the
        post-conv_head activation (``bn2``/``act2``, a fused BatchNormAct2d in
        recent timm). Falls back to the branch itself, whose forward output is the
        same map, so the contract is version-independent."""
        bb = self.backbone
        for name in ("act2", "bn2", "conv_head"):
            mod = getattr(bb, name, None)
            if isinstance(mod, nn.Module):
                return mod
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, out_channels, H/32, W/32]


class SpatialClassifier(nn.Module):
    """Standalone spatial-only model (E1 ablation arm).

    SpatialBranch -> global average pool -> dropout -> linear -> one logit.
    Single-logit + BCEWithLogitsLoss matches base.yaml (loss: bce) and the [B]
    float labels in the batch contract. Kept deliberately minimal (one linear, no
    MLP) so this arm mirrors standard EfficientNet fine-tuning and stays a clean,
    comparable baseline.
    """

    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 dropout: float = 0.3, drop_path_rate: float = 0.0,
                 num_classes: int = 1, check_normalization: bool = True):
        super().__init__()
        self.branch = SpatialBranch(
            backbone, pretrained=pretrained, drop_path_rate=drop_path_rate,
            check_normalization=check_normalization,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.branch.out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.branch(x)                        # [B, C, H/32, W/32]
        pooled = torch.flatten(self.pool(feat), 1)   # [B, C]
        logits = self.fc(self.dropout(pooled))       # [B, num_classes]
        if self.num_classes == 1:
            logits = logits.squeeze(1)               # [B] — pairs with BCEWithLogitsLoss
        return (logits, feat) if return_features else logits


def build_spatial_classifier(cfg: dict) -> SpatialClassifier:
    """Construct from a merged config dict (configs/model/spatial_only.yaml)."""
    m = (cfg or {}).get("model", {})
    return SpatialClassifier(
        backbone=m.get("backbone", DEFAULT_BACKBONE),
        pretrained=bool(m.get("pretrained", True)),
        dropout=float(m.get("dropout", 0.3)),
        drop_path_rate=float(m.get("drop_path_rate", 0.0)),
        num_classes=int(m.get("num_classes", 1)),
    )


if __name__ == "__main__":
    # Offline shape smoke test: pretrained=False so no weights are downloaded.
    torch.manual_seed(0)
    model = SpatialClassifier(pretrained=False, check_normalization=False).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        feat = model.branch(x)
        logits = model(x)
    print(f"input             : {tuple(x.shape)}")
    print(f"branch feature map: {tuple(feat.shape)}   <- fusion query + Grad-CAM target")
    print(f"branch out_channels: {model.branch.out_channels}")
    print(f"gradcam_target    : {type(model.branch.gradcam_target).__name__}")
    print(f"classifier logits : {tuple(logits.shape)}   <- one real/fake logit per sample")
