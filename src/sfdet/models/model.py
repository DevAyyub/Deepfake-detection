"""model.py — the assembled DeepfakeDetector (Steps 5/8/9/10 integrated).

Forward path (this is the integration-time picture of the invariant):

    spatial  ─► spatial_branch    ─► spatial_feat [B,1792,8,8]   (pre-fusion; query + spatial CAM target)
    frequency─► frequency_branch  ─► freq_feat     [B,256,16,16]  ◄── *** PRE-FUSION SALIENCY TAP ***
                                       (= frequency_branch.tap_block output)
                          │
                          ▼   (freq_feat and spatial_feat are PASSED INTO fusion)
                       fusion(spatial_feat, freq_feat) ─► pooled [B,512]   ◄── the ONE cross-modal mixing point
                          │
                          ▼
                       head(pooled, spatial_feat) ─► logit [B]

⚠ INTEGRATION-TIME INVARIANT (re-verified on THIS assembled model — not assumed from Step 10):
the frequency-saliency target `self.frequency_branch.gradcam_target` (-> tap_block) executes
BEFORE `self.fusion`. In `forward`, `freq_feat` is COMPUTED by the frequency branch and THEN
passed into fusion; `tap_block` lives inside `frequency_branch`, while `fusion` is a sibling
submodule — so the tap is strictly pre-fusion in the integrated path. Hooking the saliency here
attributes to frequency features alone (C2); a post-fusion hook entangles the branches and
breaks C1/C2 SILENTLY. tests/test_prefusion_hook.py asserts this on the assembled model two ways
— structural containment (target ∈ frequency_branch, target ∉ fusion) and a forward-order probe
(tap's forward hook fires strictly before fusion's). The fusion module is not novel; the branches
are modality-pure precisely so the per-branch attribution (C2) is well-defined.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sfdet.models.spatial_branch import SpatialBranch, DEFAULT_BACKBONE
from sfdet.models.frequency_branch import build_frequency_branch
from sfdet.models.fusion import build_fusion, build_head
from sfdet.explain.explainability import DualExplainer, FrequencyExplainer, SpatialExplainer


class DeepfakeDetector(nn.Module):
    """Dual spatial-frequency detector: two modality-pure branches meeting at ONE single-stage
    cross-attention fusion block, then a single-logit head. Exposes `.spatial_branch` /
    `.frequency_branch` so the explainers resolve each PRE-FUSION Grad-CAM target."""

    def __init__(self, spatial_branch, frequency_branch, fusion, head):
        super().__init__()
        self.spatial_branch = spatial_branch
        self.frequency_branch = frequency_branch
        self.fusion = fusion
        self.head = head

    # --- the pre-fusion saliency targets (stable handles the explainers resolve) ---
    @property
    def spatial_gradcam_target(self) -> nn.Module:
        """Spatial Grad-CAM target (the /32 map); pre-fusion (the query source)."""
        return self.spatial_branch.gradcam_target

    @property
    def frequency_saliency_target(self) -> nn.Module:
        """Frequency-saliency Grad-CAM target. PRE-FUSION (see module docstring): it is
        frequency_branch.tap_block, whose output is computed BEFORE fusion in `forward`."""
        return self.frequency_branch.gradcam_target

    def forward(self, spatial, frequency, return_attn: bool = False):
        spatial_feat = self.spatial_branch(spatial)              # [B,1792,8,8]  (pre-fusion)
        freq_feat = self.frequency_branch(frequency)             # [B,256,16,16] (*** PRE-FUSION TAP ***)
        # ---- the SINGLE cross-modal mixing point; everything above is modality-pure ----
        if return_attn:
            pooled, attn = self.fusion(spatial_feat, freq_feat, return_attn=True)
        else:
            pooled, attn = self.fusion(spatial_feat, freq_feat), None  # [B,512]
        logits = self.head(pooled, spatial_feat)                 # [B]  (head ignores spatial_feat
        return (logits, attn) if return_attn else logits          #       unless include_spatial=True)

    # --- integrated explainability (per-sample maps from the detector's own fake logit) ---
    def explain(self, spatial, frequency, normalize: bool = True):
        """Spatial Grad-CAM + frequency (rho, theta) saliency, both from the PRE-FUSION taps,
        in one forward + one backward. Returns {'spatial': {...}, 'frequency': {...}}."""
        return DualExplainer(self)(spatial, frequency, normalize=normalize)

    def spatial_explainer(self) -> SpatialExplainer:
        return SpatialExplainer(self)

    def frequency_explainer(self) -> FrequencyExplainer:
        return FrequencyExplainer(self)


def build_detector(cfg: dict) -> DeepfakeDetector:
    """Assemble the full detector from a merged config, wiring dimensions from each branch.
    Mirrors build_spatial_classifier's model-key reading for the spatial branch."""
    m = (cfg or {}).get("model", {})
    data = (cfg or {}).get("data", {})
    image_size = int(data.get("image_size", 256))

    spatial = SpatialBranch(
        backbone=m.get("backbone", DEFAULT_BACKBONE),
        pretrained=bool(m.get("pretrained", True)),
        drop_path_rate=float(m.get("drop_path_rate", 0.0)),
        check_normalization=bool(m.get("check_normalization", True)),
    )
    spatial_hw = (image_size // spatial.REDUCTION, image_size // spatial.REDUCTION)   # (8, 8) @256
    frequency = build_frequency_branch(cfg)
    fusion = build_fusion(cfg, spatial_channels=spatial.out_channels,
                          freq_channels=frequency.out_channels,
                          spatial_hw=spatial_hw, freq_hw=frequency.tap_hw)
    head = build_head(cfg, d_model=fusion.out_dim, spatial_channels=spatial.out_channels)
    return DeepfakeDetector(spatial, frequency, fusion, head)


def _selftest():
    """Runs on real torch (pretrained=False so no weight download): full forward, attention
    diagnostic, and the integrated explainers; re-checks the pre-fusion forward order."""
    cfg = {"model": {"pretrained": False, "check_normalization": False},
           "data": {"image_size": 256}}
    model = build_detector(cfg).eval()
    B = 2
    spatial = torch.randn(B, 3, 256, 256)
    frequency = torch.randn(B, 1, 256, 256)

    with torch.no_grad():
        logits = model(spatial, frequency)
        logits2, attn = model(spatial, frequency, return_attn=True)
    print(f"logits {tuple(logits.shape)} | attention {tuple(attn.shape)}")
    assert tuple(logits.shape) == (B,)

    # integration-time invariant: forward order (tap strictly before fusion)
    order = []
    h1 = model.frequency_saliency_target.register_forward_hook(lambda *a: order.append("tap"))
    h2 = model.fusion.register_forward_hook(lambda *a: order.append("fusion"))
    with torch.no_grad():
        model(spatial, frequency)
    h1.remove(); h2.remove()
    print(f"forward order: {order}")
    assert order == ["tap", "fusion"], order

    maps = model.explain(spatial, frequency)
    print(f"frequency saliency {tuple(maps['frequency']['cam'].shape)} | "
          f"spatial CAM {tuple(maps['spatial']['cam'].shape)}")
    assert tuple(maps["frequency"]["cam"].shape) == (B, 16, 16)
    print("OK: detector assembled; frequency saliency target is pre-fusion (order tap->fusion)")


if __name__ == "__main__":
    _selftest()
