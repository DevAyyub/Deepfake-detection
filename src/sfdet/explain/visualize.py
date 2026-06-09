"""visualize.py — render the (rho, theta) frequency saliency (and spatial Grad-CAM).

No torch: operates on numpy CAM arrays (call .detach().cpu().numpy() upstream). cv2 handles
resize / colormap / inverse-polar warp; matplotlib is optional (lazy) for labeled (rho, theta)
figures.

The PRIMARY frequency artifact is the (rho, theta) heatmap (rows = radial frequency, cols =
orientation) — the frequency branch's native polar layout, so it is exact by construction. The
Cartesian overlay is a SECONDARY rendering: an inverse-polar warp of that map back onto the
2D-FFT magnitude, for showing 'which spectral region drove this sample' over the actual
spectrum. The inverse warp matches FrequencyBranch's forward polar convention
(center = S//2, r_max = S//2 - 1, theta in [0, 2*pi)).
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:                         # pragma: no cover
    cv2 = None


def _need_cv2():
    if cv2 is None:
        raise RuntimeError("cv2 (opencv-python-headless) is required for this function")


def _to_uint8(m01):
    return np.clip(np.asarray(m01, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


def _norm01(a, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    return (a - a.min()) / (a.max() - a.min() + eps)


def rho_theta_map(cam, out_hw=(256, 256)):
    """Upsample a [Hc, Wc] (rho, theta) CAM to a smooth (rho, theta) heatmap in [0,1].
    Rows = rho (DC at row 0), cols = theta in [0, 2*pi)."""
    _need_cv2()
    cam = np.asarray(cam, dtype=np.float32)
    up = cv2.resize(cam, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    return _norm01(up)


def colorize(map01, cmap=None):
    """[H, W] in [0,1] -> [H, W, 3] uint8 RGB heatmap."""
    _need_cv2()
    cmap = cv2.COLORMAP_JET if cmap is None else cmap
    bgr = cv2.applyColorMap(_to_uint8(map01), cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _inverse_polar_maps(image_size, n_rho, n_theta):
    """cv2.remap maps to warp a (rho, theta) image [n_rho, n_theta] back onto the centered
    Cartesian spectrum [S, S], matching FrequencyBranch's forward convention."""
    S = image_size
    center = S // 2
    r_max = float(S // 2 - 1)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx = xx - center
    dy = yy - center
    rho = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    map_y = ((rho / r_max) * (n_rho - 1)).astype(np.float32)          # row index = rho
    map_x = (np.mod((theta / (2.0 * np.pi)) * n_theta, n_theta)).astype(np.float32)  # col = theta
    return map_x, map_y, (rho > r_max)                               # mask: outside sampled radius


def to_cartesian_spectrum(cam, image_size=256):
    """Inverse-polar warp a (rho, theta) CAM to the centered Cartesian spectrum [S, S] in [0,1].
    Corners beyond r_max (not represented in the polar map) are set to 0. Accepts a CAM at any
    resolution (it is treated as the polar grid)."""
    _need_cv2()
    cam = np.asarray(cam, dtype=np.float32)
    n_rho, n_theta = cam.shape
    map_x, map_y, outside = _inverse_polar_maps(image_size, n_rho, n_theta)
    cart = cv2.remap(cam, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    cart[outside] = 0.0
    return _norm01(cart)


def overlay(base, map01, alpha=0.5, cmap=None):
    """Alpha-blend a [0,1] saliency map (colorized) over a base image (grayscale [H,W] or RGB
    [H,W,3]) -> [H,W,3] uint8 RGB."""
    _need_cv2()
    base = np.asarray(base)
    if base.ndim == 2:
        base = cv2.cvtColor(_to_uint8(_norm01(base)), cv2.COLOR_GRAY2RGB)
    elif base.dtype != np.uint8:
        base = _to_uint8(_norm01(base))
    heat = colorize(_norm01(map01), cmap)
    if heat.shape[:2] != base.shape[:2]:
        heat = cv2.resize(heat, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0.0)


def spatial_overlay(cam, image_rgb, alpha=0.5):
    """Spatial Grad-CAM ([Hc, Wc]) blended over the input crop ([H, W, 3])."""
    return overlay(image_rgb, np.asarray(cam, dtype=np.float32), alpha=alpha)


def frequency_panels(cam, magnitude=None, image_size=256, out_hw=(256, 256)):
    """Convenience: returns the standard frequency-saliency panels as RGB uint8 arrays —
    {'rho_theta': (rho,theta) heatmap, 'cartesian': inverse-warped spectrum heatmap,
     'overlay': cartesian heatmap over `magnitude` if provided}."""
    rt = rho_theta_map(cam, out_hw=out_hw)
    cart = to_cartesian_spectrum(cam, image_size=image_size)
    panels = {"rho_theta": colorize(rt), "cartesian": colorize(cart)}
    if magnitude is not None:
        panels["overlay"] = overlay(magnitude, cart)
    return panels


def save_panels(path, panels):
    """hstack a list of [H, W, 3] RGB uint8 panels and write to `path` (PNG)."""
    _need_cv2()
    h = max(p.shape[0] for p in panels)
    res = []
    for p in panels:
        if p.shape[0] != h:
            p = cv2.resize(p, (int(round(p.shape[1] * h / p.shape[0])), h))
        res.append(p)
    grid = np.hstack(res)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return str(path)


def plot_rho_theta(cam, title="frequency saliency (ρ, θ)", out_path=None):
    """Optional labeled (rho, theta) figure (matplotlib, imported lazily). Rows = rho
    (DC -> Nyquist), cols = theta in degrees. For paper figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_rho_theta") from e
    cam = _norm01(np.asarray(cam, dtype=np.float32))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cam, origin="lower", aspect="auto", cmap="jet",
                   extent=[0, 360, 0, cam.shape[0]])
    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("ρ (radial frequency: DC → Nyquist)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="saliency (normalized; display only)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
