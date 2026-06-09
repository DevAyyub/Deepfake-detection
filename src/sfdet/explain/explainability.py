"""explainability.py — integrated Grad-CAM explainers for the dual detector.

Two explainers, both producing PER-SAMPLE, CLASS-DISCRIMINATIVE attribution from the detector's
own fake logit:
  * SpatialExplainer    — Grad-CAM over the spatial branch's /32 map (image space).
  * FrequencyExplainer  — Grad-CAM over the frequency branch's PRE-FUSION (rho, theta) tap,
                          a saliency map over the 2D-FFT magnitude spectrum indexed by (rho, theta).
  * DualExplainer       — both at once from the assembled DualBranchDetector (Step 11).

⚠ BINDING INVARIANT (frequency saliency): the Grad-CAM target is a FREQUENCY-BRANCH layer
registered BEFORE the cross-attention fusion — FrequencyBranch.gradcam_target (-> tap_block).
See the hook-site comment in FrequencyExplainer.

SCOPE: these maps LOCALIZE DISCRIMINATIVE SPECTRAL / SPATIAL CONTENT (where gradient x
activation is high for the fake decision). They are NOT a faithfulness claim — Grad-CAM
faithfulness is contested (Adebayo 2018), and the two maps are a per-branch *attribution*,
computed inside the detector over its own features. The actual faithfulness probe (spectral
occlusion / GT mask) is E5, a separate experiment; this module only GENERATES the maps.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def grad_cam_map(activation, gradient, normalize: bool = True, eps: float = 1e-8):
    """Grad-CAM from an activation A [B,C,H,W] and its gradient G [B,C,H,W] w.r.t. the target
    scalar: w = GAP(G); CAM = ReLU(sum_c w_c * A_c) -> [B,H,W].

    Per-sample [0,1] normalization is for DISPLAY only — a visualization step, not a magnitude
    or faithfulness statement (pass normalize=False for raw importance, e.g. for E5 occlusion)."""
    weights = gradient.mean(dim=(2, 3), keepdim=True)          # [B,C,1,1]
    cam = torch.relu((weights * activation).sum(dim=1))        # [B,H,W]
    if not normalize:
        return cam
    B = cam.shape[0]
    flat = cam.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    return ((flat - mn) / (mx - mn + eps)).reshape(cam.shape)


class GradCAM:
    """Generic Grad-CAM: hooks `target_layer`, runs one forward + one backward from a scalar
    target, returns {'cam','cam_raw'}. Hooks are registered per call and removed in `finally`
    (no leaked hooks).

    Runs with the model in eval() so BN uses running statistics and samples do not couple
    through batch stats; backprop of `score.sum()` then yields, for each sample, exactly
    d(its logit)/d(its activation) — so the whole batch's per-sample CAMs come from one pass."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, *forward_args, sign: float = 1.0, normalize: bool = True, **forward_kwargs):
        store = {}

        def fwd_hook(_module, _inp, out):
            store["A"] = out
            out.retain_grad()                      # keep grad on this non-leaf activation

        handle = self.target_layer.register_forward_hook(fwd_hook)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                logits = self.model(*forward_args, **forward_kwargs)
                if logits.ndim > 1 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                score = sign * logits              # fake logit; sign=-1 for a "real"-class map
                self.model.zero_grad(set_to_none=True)
                score.sum().backward()
            A = store["A"]
            cam_raw = grad_cam_map(A, A.grad, normalize=False)
            cam = grad_cam_map(A, A.grad, normalize=normalize)
        finally:
            handle.remove()
            if was_training:
                self.model.train()
        return {"cam": cam.detach(), "cam_raw": cam_raw.detach()}


def _resolve_target(model, which: str) -> nn.Module:
    """Locate the pre-fusion gradcam_target for 'spatial' or 'frequency' across: the assembled
    DualBranchDetector (model.spatial_branch / model.frequency_branch), the single-branch
    SpatialClassifier / FrequencyClassifier (model.branch), or a bare branch (model.gradcam_target)."""
    branch_attr = "spatial_branch" if which == "spatial" else "frequency_branch"
    if hasattr(model, branch_attr):
        return getattr(model, branch_attr).gradcam_target
    if hasattr(model, "branch") and hasattr(getattr(model, "branch"), "gradcam_target"):
        return model.branch.gradcam_target
    if hasattr(model, "gradcam_target"):
        return model.gradcam_target
    raise AttributeError(f"could not locate a {which} gradcam_target on {type(model).__name__}")


class SpatialExplainer:
    """Grad-CAM over the spatial branch /32 map ([B,1792,8,8]) -> image-space saliency [B,8,8]
    (upsample to the crop for display). The target is pre-fusion (the query source, upstream of
    the fusion module)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or _resolve_target(model, "spatial")
        self.gradcam = GradCAM(model, self.target_layer)

    def __call__(self, *forward_args, normalize: bool = True, **kw):
        out = self.gradcam(*forward_args, sign=1.0, normalize=normalize, **kw)
        return {"cam": out["cam"], "cam_raw": out["cam_raw"], "space": "image"}


class FrequencyExplainer:
    """Per-sample, class-discriminative frequency saliency over the 2D-FFT magnitude, indexed by
    (rho, theta). Grad-CAM over the frequency branch's pre-fusion tap ([B,256,16,16])."""

    def __init__(self, model, target_layer=None, coords: str = "polar"):
        self.model = model
        # ------------------------------------------------------------------------------------ #
        # ⚠ PRE-FUSION HOOK SITE — BINDING INVARIANT (load-bearing for C2).
        # The target is the frequency branch's OWN last feature map
        # (FrequencyBranch.gradcam_target -> tap_block, [B,256,16,16]), produced BEFORE the
        # cross-attention fusion (fusion is a separate, DOWNSTREAM module — neither branch
        # imports it). The activation is therefore a pure function of the FFT magnitude, so the
        # resulting (rho, theta) map attributes to FREQUENCY features ALONE — that is C2.
        #
        # If this hook is placed on ANY post-fusion tensor (the fused tokens, the pooled vector,
        # the head input, or the spatial query stream — which carries spatial content forward
        # via the residual), the attribution becomes JOINT (spatial+frequency entangled) and C1
        # and C2 contradict. It fails SILENTLY: the code runs and a map still renders. So we
        # resolve the target via the .gradcam_target PROPERTY (never a re-derived module path),
        # and tests/test_prefusion_hook.py pins this on the assembled model.
        #
        # The Grad-CAM GRADIENT flows from the dual fake logit back THROUGH fusion to this tap
        # (so the map reflects the end-to-end decision), while the ACTIVATION stays modality-
        # pure — that is the whole design, and the reason the tap must be pre-fusion.
        # ------------------------------------------------------------------------------------ #
        self.target_layer = target_layer or _resolve_target(model, "frequency")
        self.gradcam = GradCAM(model, self.target_layer)
        self.coords = self._infer_coords(coords)

    def _infer_coords(self, default):
        for attr in ("frequency_branch", "branch"):
            obj = getattr(self.model, attr, None)
            if obj is not None and hasattr(obj, "coords"):
                return obj.coords
        return getattr(self.model, "coords", default)

    def __call__(self, *forward_args, normalize: bool = True, **kw):
        out = self.gradcam(*forward_args, sign=1.0, normalize=normalize, **kw)
        # cam is [B,16,16]. In polar mode it is ALREADY indexed by (rho, theta) (rows=rho,
        # cols=theta). In cartesian mode it is (kx,ky); visualize.to_cartesian_spectrum / a
        # polar resample maps it to (rho, theta) downstream.
        layout = "rho_theta" if self.coords == "polar" else "cartesian"
        return {"cam": out["cam"], "cam_raw": out["cam_raw"], "layout": layout, "coords": self.coords}


class DualExplainer:
    """Runs BOTH explainers from the assembled DualBranchDetector's single fake logit, in ONE
    forward + ONE backward (both pre-fusion taps hooked simultaneously). For the Step-11 model;
    until then use SpatialExplainer / FrequencyExplainer on the single-branch classifiers."""

    def __init__(self, detector):
        self.detector = detector
        self.spatial_target = _resolve_target(detector, "spatial")
        self.freq_target = _resolve_target(detector, "frequency")     # pre-fusion (see above)

    def __call__(self, spatial, frequency, normalize: bool = True):
        store = {}

        def mk(name):
            def hook(_m, _i, out):
                store[name] = out
                out.retain_grad()
            return hook

        hs = self.spatial_target.register_forward_hook(mk("spatial"))
        hf = self.freq_target.register_forward_hook(mk("freq"))
        was_training = self.detector.training
        self.detector.eval()
        try:
            with torch.enable_grad():
                logits = self.detector(spatial, frequency)
                if logits.ndim > 1 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                self.detector.zero_grad(set_to_none=True)
                logits.sum().backward()
            sp, fr = store["spatial"], store["freq"]
            spatial_cam = grad_cam_map(sp, sp.grad, normalize=normalize)
            freq_cam = grad_cam_map(fr, fr.grad, normalize=normalize)
        finally:
            hs.remove()
            hf.remove()
            if was_training:
                self.detector.train()
        return {"spatial": {"cam": spatial_cam.detach(), "space": "image"},
                "frequency": {"cam": freq_cam.detach(), "layout": "rho_theta"}}


def _selftest():
    """Runs on real torch: dummy inputs through the UN-FUSED frequency and spatial branches,
    real hooks + real backward, checks CAM shapes and the (rho, theta) layout."""
    from sfdet.models.frequency_branch import FrequencyBranch, FrequencyClassifier
    from sfdet.models.spatial_branch import SpatialClassifier

    fclf = FrequencyClassifier(FrequencyBranch(in_channels=1, image_size=256, coords="polar"))
    fx = torch.randn(2, 1, 256, 256)
    fout = FrequencyExplainer(fclf)(fx)
    print(f"frequency saliency: cam {tuple(fout['cam'].shape)} layout={fout['layout']}")
    assert tuple(fout["cam"].shape) == (2, 16, 16)
    assert fout["layout"] == "rho_theta"

    sclf = SpatialClassifier(pretrained=False)   # builds its own efficientnet_b4 branch (name, not instance)
    sx = torch.randn(2, 3, 256, 256)
    sout = SpatialExplainer(sclf)(sx)
    print(f"spatial grad-cam: cam {tuple(sout['cam'].shape)}")
    assert sout["cam"].shape[0] == 2 and sout["cam"].ndim == 3
    print("OK: explainers produce per-sample maps (frequency (rho,theta) + spatial)")


if __name__ == "__main__":
    _selftest()
