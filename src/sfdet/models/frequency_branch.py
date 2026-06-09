"""frequency_branch.py — CNN over the 2D-FFT magnitude spectrum (the frequency branch).

Consumes the loader's frequency view: a **fftshifted** (DC-centered), log1p, per-image
z-scored magnitude spectrum, [B, in_channels, S, S] (in_channels=1 grayscale by default,
S = data.image_size = 256). Because the spectrum is DC-centered, radius from the center is
spatial-frequency magnitude (rho) and angle is orientation (theta).

Design (see the §2.x design note):
  * A magnitude spectrum is NOT a natural image. Image translation -> phase only, and we kept
    magnitude, so the spectrum is shift-invariant: ABSOLUTE position is the signal (a peak at a
    given (rho, theta) means a specific periodicity from the generator's upsampling). So this
    branch is position-sensitive, from-scratch, and compact — deliberately NOT a pretrained
    natural-image backbone (that prior is wrong here, and it would break modality-purity).
  * (rho, theta): we resample the centered magnitude to a polar grid (rows = rho, cols = theta)
    so radial structure becomes horizontal bands and angular structure becomes columns; the
    Grad-CAM map is then literally a (rho, theta) heatmap. theta is periodic, so convolutions
    pad the theta axis CIRCULARLY (and the rho axis with zeros). Cartesian (no resample) is a
    one-flag ablation.
  * Representation choice (DFT magnitude over block-DCT / Haar) is for representational FIT to
    the per-sample (rho, theta) attribution (Corvi 2023: generator/upsampling traces are most
    directly localizable in the global DFT magnitude) — NOT an architectural novelty. This
    branch EXISTS TO ENABLE C2; nothing here is "novel" or a faithfulness claim.

Spectral framing (carry into any prose): subset differences are a mixture of shared and
distinct spectral signatures whose balance tracks how much convolutional upsampling sits in the
generation path — not latent-vs-pixel, not qualitatively different mechanisms.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Fixed geometric buffers (built with numpy, stored as torch buffers)
# --------------------------------------------------------------------------- #
def _build_polar_grid(image_size: int, n_rho: int, n_theta: int) -> "torch.Tensor":
    """Sampling grid [n_rho, n_theta, 2] in [-1, 1] (x, y) for grid_sample, mapping each
    output (rho, theta) cell to the centered Cartesian spectrum about the DC bin.

    rho in [0, r_max] along rows, theta in [0, 2*pi) along columns. Centered on the DC bin
    (index S//2 after fftshift); r_max = S//2 - 1 keeps every sample in-bounds along the axes
    (no zero-fan)."""
    S = image_size
    center = S // 2                 # DC bin after fftshift (even S)
    r_max = S // 2 - 1              # in-bounds along +x / +y axes
    rho = np.linspace(0.0, r_max, n_rho)[:, None]                       # [n_rho, 1]
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)[None, :]  # [1, n_theta]
    x = center + rho * np.cos(theta)        # [n_rho, n_theta] pixel coords (cols / width)
    y = center + rho * np.sin(theta)        # [n_rho, n_theta] pixel coords (rows / height)
    gx = x / (S - 1) * 2.0 - 1.0            # normalize to [-1, 1] for grid_sample (align_corners)
    gy = y / (S - 1) * 2.0 - 1.0
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)              # [n_rho, n_theta, 2] = (x, y)
    return torch.from_numpy(grid)


def _build_radial_map_polar(n_rho: int, n_theta: int) -> "torch.Tensor":
    """rho-ramp coordinate channel for the polar layout (row index normalized to [0, 1])."""
    r = (np.linspace(0.0, 1.0, n_rho)[:, None] * np.ones((1, n_theta))).astype(np.float32)
    return torch.from_numpy(r[None, None])                            # [1, 1, n_rho, n_theta]


def _build_radial_map_cart(image_size: int) -> "torch.Tensor":
    """radial-distance coordinate channel for the Cartesian layout (distance from the DC bin)."""
    S = image_size
    center = S // 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    r = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    r = (r / (r.max() + 1e-8)).astype(np.float32)
    return torch.from_numpy(r[None, None])                            # [1, 1, S, S]


# --------------------------------------------------------------------------- #
# Conv block with theta-periodic padding
# --------------------------------------------------------------------------- #
class _ConvBlock(nn.Module):
    """conv -> BN -> ReLU. In polar mode the width axis is theta (periodic), so we pad it
    CIRCULARLY and pad the height (rho) with zeros; we F.pad manually, then conv with
    padding=0. In Cartesian mode both axes get zero padding."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, k: int = 3,
                 periodic_w: bool = True):
        super().__init__()
        self.k = k
        self.stride = stride
        self.pad = k // 2
        self.periodic_w = periodic_w
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        p = self.pad
        if self.periodic_w:
            x = F.pad(x, (p, p, 0, 0), mode="circular")    # width = theta (periodic seam)
            x = F.pad(x, (0, 0, p, p), mode="constant")    # height = rho (zeros)
        else:
            x = F.pad(x, (p, p, p, p), mode="constant")
        return self.act(self.bn(self.conv(x)))


# --------------------------------------------------------------------------- #
# The frequency branch
# --------------------------------------------------------------------------- #
class FrequencyBranch(nn.Module):
    """From-scratch CNN over the (DC-centered, log1p, per-image z-scored) magnitude spectrum.

    Pipeline: [polar (rho, theta) resample (default)] -> [radial-coord channel] -> stem ->
    4 downsampling stages (÷16) -> PRE-FUSION TAP.

    forward(x) returns the last conv feature map `freq_feat` [B, out_channels, Hf, Wf]
    (16x16 at image_size=256).
    """

    def __init__(self, in_channels: int = 1, image_size: int = 256, coords: str = "polar",
                 radial_coord: bool = True, widths=(32, 64, 128, 256)):
        super().__init__()
        assert coords in ("polar", "cartesian"), f"coords must be polar|cartesian, got {coords!r}"
        self.in_channels = in_channels
        self.image_size = image_size
        self.coords = coords
        self.radial_coord = radial_coord

        n_rho = n_theta = image_size
        if coords == "polar":
            self.register_buffer("polar_grid", _build_polar_grid(image_size, n_rho, n_theta),
                                 persistent=False)
            if radial_coord:
                self.register_buffer("radial_map", _build_radial_map_polar(n_rho, n_theta),
                                     persistent=False)
        else:
            self.polar_grid = None
            if radial_coord:
                self.register_buffer("radial_map", _build_radial_map_cart(image_size),
                                     persistent=False)

        c_in = in_channels + (1 if radial_coord else 0)
        w0, w1, w2, w3 = widths
        pw = (coords == "polar")        # circular theta-padding only meaningful in polar mode

        # stem keeps resolution and lifts channels
        self.stem = _ConvBlock(c_in, w0, stride=1, periodic_w=pw)
        # 4 downsampling stages -> ÷16 ; each = one stride-2 conv + one stride-1 refine conv
        self.stage1 = nn.Sequential(_ConvBlock(w0, w1, stride=2, periodic_w=pw),
                                    _ConvBlock(w1, w1, stride=1, periodic_w=pw))
        self.stage2 = nn.Sequential(_ConvBlock(w1, w2, stride=2, periodic_w=pw),
                                    _ConvBlock(w2, w2, stride=1, periodic_w=pw))
        self.stage3 = nn.Sequential(_ConvBlock(w2, w3, stride=2, periodic_w=pw),
                                    _ConvBlock(w3, w3, stride=1, periodic_w=pw))
        # ---------------------------------------------------------------------------------- #
        # *** PRE-FUSION TAP (BINDING INVARIANT for C2) ***
        # `tap_block`'s output is the frequency branch's OWN last feature map, produced BEFORE
        # any cross-attention (fusion lives OUTSIDE this module). It serves BOTH roles:
        #   (1) the key/value source for the single-stage cross-attention, and
        #   (2) the Grad-CAM target for the frequency saliency (see `gradcam_target`).
        # Registering the saliency hook here — not on the FUSED representation — is what keeps
        # the attribution per-branch. A post-fusion hook entangles spatial+frequency and
        # SILENTLY collapses the C1/C2 separation (the code would still render a map). Do not
        # move the hook past fusion; tests/test_prefusion_hook.py guards this.
        # ---------------------------------------------------------------------------------- #
        self.tap_block = nn.Sequential(_ConvBlock(w3, w3, stride=2, periodic_w=pw),
                                       _ConvBlock(w3, w3, stride=1, periodic_w=pw))

        self.out_channels = w3
        self.reduction = 16
        self.tap_hw = (n_rho // 16, n_theta // 16)      # (16, 16) at image_size=256

    @property
    def gradcam_target(self) -> nn.Module:
        """The module the frequency-saliency Grad-CAM hooks. MUST be pre-fusion: this is the
        frequency branch's own final feature map, before cross-attention."""
        return self.tap_block

    def _to_polar(self, x):
        grid = self.polar_grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)   # [B, n_rho, n_theta, 2]
        return F.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="border")

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: [B, in_channels, S, S] — the loader's DC-centered log-magnitude (polar resample
        # assumes DC is at the center, which fftshift guarantees).
        if self.coords == "polar":
            x = self._to_polar(x)                        # [B, in_channels, n_rho, n_theta]
        if self.radial_coord:
            rc = self.radial_map.expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, rc], dim=1)                # + 1 coordinate channel
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        freq_feat = self.tap_block(x)                    # *** PRE-FUSION TAP ***
        return freq_feat


class FrequencyClassifier(nn.Module):
    """Frequency branch + global pool + linear head. This is the FREQUENCY-ONLY ablation
    (ABL3) and a standalone-trainable detector; it mirrors SpatialClassifier so the training/
    eval engine treats the two identically. Returns one logit per sample."""

    def __init__(self, branch: FrequencyBranch, dropout: float = 0.3, num_classes: int = 1):
        super().__init__()
        self.branch = branch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(branch.out_channels, 1 if num_classes <= 2 else num_classes)

    def forward(self, x, return_features: bool = False):
        feat = self.branch(x)                            # [B, C_f, Hf, Wf] (pre-fusion tap)
        pooled = self.pool(feat).flatten(1)
        logits = self.fc(self.drop(pooled)).squeeze(-1)  # [B]
        return (logits, feat) if return_features else logits


# --------------------------------------------------------------------------- #
# Builders (read the merged config; mirror build_spatial_classifier)
# --------------------------------------------------------------------------- #
def build_frequency_branch(cfg: dict) -> FrequencyBranch:
    data = cfg.get("data", {})
    freq = data.get("frequency", {})
    model = cfg.get("model", {})
    in_channels = 3 if str(freq.get("channels", "gray")).lower() == "rgb" else 1
    return FrequencyBranch(
        in_channels=in_channels,
        image_size=int(data.get("image_size", 256)),
        coords=str(model.get("freq_coords", "polar")),          # ablation: "cartesian"
        radial_coord=bool(model.get("freq_radial_coord", True)),
    )


def build_frequency_classifier(cfg: dict) -> FrequencyClassifier:
    model = cfg.get("model", {})
    return FrequencyClassifier(
        build_frequency_branch(cfg),
        dropout=float(model.get("dropout", 0.3)),
        num_classes=int(model.get("num_classes", 1)),
    )


# --------------------------------------------------------------------------- #
# Self-test (random weights; runs under real torch and the shape-stub)
# --------------------------------------------------------------------------- #
def _selftest():
    for coords in ("polar", "cartesian"):
        branch = FrequencyBranch(in_channels=1, image_size=256, coords=coords, radial_coord=True)
        x = torch.randn(2, 1, 256, 256)                 # dummy FFT-magnitude batch
        feat = branch(x)
        print(f"[{coords}] input {tuple(x.shape)} -> pre-fusion tap freq_feat {tuple(feat.shape)}"
              f"  (out_channels={branch.out_channels}, reduction={branch.reduction},"
              f" tap_hw={branch.tap_hw})")
        assert tuple(feat.shape) == (2, branch.out_channels, 16, 16)
        assert branch.gradcam_target is branch.tap_block           # tap resolves, pre-fusion

        clf = FrequencyClassifier(branch)
        logits = clf(x)
        print(f"[{coords}] FrequencyClassifier logits {tuple(logits.shape)}")
        assert tuple(logits.shape) == (2,)
        logits2, feat2 = clf(x, return_features=True)
        assert tuple(feat2.shape) == (2, branch.out_channels, 16, 16)
    print("OK: frequency branch shapes verified (polar + cartesian)")


if __name__ == "__main__":
    _selftest()
