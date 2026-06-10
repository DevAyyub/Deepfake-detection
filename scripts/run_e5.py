#!/usr/bin/env python
"""run_e5.py — E5: the frequency-saliency faithfulness PROBE (C2's load-bearing experiment).

WHAT THIS DOES
--------------
For the FULL dual-branch detector, it asks a causal question about the per-sample,
class-discriminative frequency saliency (the pre-fusion (rho, theta) Grad-CAM, C2):

    "Are the (rho, theta) regions the saliency marks as important actually the regions
     the detector's fake decision DEPENDS ON?"

It answers it by spectral occlusion: rank the (rho, theta) cells by saliency, replace the
top-k with a neutral baseline IN THE POLAR SPECTRUM the frequency branch convolves over,
re-run the WHOLE model, and watch the fake decision degrade. The signal is not the raw
drop (removing anything lowers a logit) but the *selectivity*: saliency-ordered occlusion
vs control orderings.

  * deletion : start from the true spectrum, progressively occlude top-saliency cells.
  * insertion: start from the all-baseline spectrum, progressively restore top-saliency cells.
  * controls : random cells (essential — shares the masking OOD artifact, so the
               saliency-vs-random gap isolates saliency quality), highest-ENERGY cells
               (essential — tests "is the saliency just an energy/peak detector?"), and
               anti-saliency / least-salient (cheap monotonicity check).

REPORTED METRICS (per dataset, separately)
------------------------------------------
  * auc_full      : AUC with no occlusion (deletion @ frac 0).
  * auc_nofreq    : AUC with the WHOLE spectrum replaced by the baseline (deletion @ frac 1
                    == insertion @ frac 0); the frequency branch is neutralized.
  * freq_reliance : auc_full - auc_nofreq. *** Read this FIRST (see "HOW TO READ", below). ***
  * deletion / insertion AUC curves vs occluded fraction, per method.
  * Delta-AUC @ k : auc_full - AUC_deletion(method, k), saliency vs random vs energy, and the GAP.
  * area gaps     : normalized area between the saliency curve and each control curve.
  * radial saliency profile: mean saliency vs rho (a light spectral-fingerprint corroboration).

HOW TO READ IT AGAINST C2 (honest version)
-------------------------------------------
C2 is scoped to "frequency-domain attribution that LOCALIZES DISCRIMINATIVE SPECTRAL CONTENT".
It is NOT a faithfulness-by-construction claim: the attribution is still JOINT at the logit
(the Grad-CAM gradient flows back through fusion), and Grad-CAM faithfulness is contested
(Adebayo 2018). So:

  * A POSITIVE result (saliency-ordered occlusion drops AUC / fake-logit substantially MORE
    than random AND energy; insertion recovers faster) SUPPORTS the scoped claim: the
    highlighted (rho, theta) regions are the ones the decision depends on, and the map is not
    vacuous nor merely an energy detector.
  * It does NOT license the words "faithful", "proves", or "more faithful than DF-P2E". The
    DF-P2E advantage stays CATEGORICAL (attribution computed inside the detector over its own
    frequency features), never a faithfulness comparison.
  * freq_reliance is the honesty gate. If auc_full ~= auc_nofreq the model barely uses the
    frequency branch on THIS data, so the deletion curve is near-flat for EVERY method and a
    null saliency-vs-random gap means "frequency is not load-bearing here", NOT "the saliency
    is unfaithful". This is the expected story on FF++ (recall spatial-only matched/beat the
    full model in-domain), and exactly why E5 must also run on the DF40 diffusion subsets,
    where frequency is load-bearing. Per the spectral framing, expect a larger gap on the
    strong-upsampling subsets (DDPM, SD-2.1) than the weak-peak ones (PixArt-alpha, DiT) —
    reported correlationally (the balance tracks convolutional upsampling), never as
    "different mechanisms".

BINDING INVARIANT: the saliency comes from sfdet.explain.FrequencyExplainer, whose Grad-CAM
target is the PRE-FUSION FrequencyBranch.gradcam_target (tap_block). E5 never re-derives a
hook; it reuses that explainer, so the attribution stays per-branch.

RUN (on Colab, after cells 1-8; the full-model best.pt is on Drive)
-------------------------------------------------------------------
    # First pass on FF++ val (fast; --max-batches caps the sample count):
    python scripts/run_e5.py \
        --checkpoint /content/drive/MyDrive/deepfake/runs/e1_full/best.pt \
        --out-dir   /content/drive/MyDrive/deepfake/runs/e1_full/e5 \
        --max-batches 64

    # Final pass, full FF++ val + DF40 diffusion subsets (once those manifests exist on Colab):
    python scripts/run_e5.py --checkpoint .../e1_full/best.pt --out-dir .../e1_full/e5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# default occlusion grid (fraction of the 16x16 = 256 (rho, theta) cells).
# 0.0 and 1.0 are endpoints: deletion@0 = full spectrum (auc_full),
# deletion@1 = baseline spectrum (auc_nofreq); insertion mirrors them.
DEFAULT_FRACS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
CELL_GRID = 16              # saliency / tap resolution at image_size=256 (256 / reduction(16))
REPORT_AT = [0.10, 0.15]    # headline Delta-AUC operating points


# --------------------------------------------------------------------------- #
# config merge (mirrors scripts/train.py so behaviour is identical)
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config(base_path, model_path, data_path=None) -> dict:
    cfg = yaml.safe_load(Path(base_path).read_text()) or {}
    for overlay_path in (data_path, model_path):       # base <- data <- model
        if overlay_path:
            cfg = deep_merge(cfg, yaml.safe_load(Path(overlay_path).read_text()) or {})
    return cfg


# --------------------------------------------------------------------------- #
# polar-space occlusion: temporarily wrap FrequencyBranch._to_polar so the
# branch sees an occluded / inserted polar spectrum. Self-contained (no edits to
# committed model code); the saliency map lives in (rho, theta), so we intervene
# in exactly that space. _to_polar is a method (not a hookable nn.Module), hence
# the wrap rather than a forward-hook.
# --------------------------------------------------------------------------- #
class PolarOcclusion:
    """Context manager. Per batch: precompute the true polar spectrum + a baseline,
    then for each (mask, mode) set them and run the model; the wrapped _to_polar
    returns the blended spectrum instead of resampling the (now ignored) input."""

    def __init__(self, branch):
        self.branch = branch
        self._orig = branch._to_polar          # bound method; used to build polar_true
        self.polar_true = None                 # [B,1,Hp,Wp]
        self.baseline = None                   # [B,1,Hp,Wp]
        self.mask = None                       # [B,1,Hp,Wp] in {0,1}; 1 == "selected cell"
        self.mode = "deletion"

    def __enter__(self):
        self.branch._to_polar = self._patched
        return self

    def __exit__(self, *exc):
        self.branch._to_polar = self._orig
        self.mask = None
        return False

    def polar_of(self, frequency):
        """The true polar spectrum for this batch, via the ORIGINAL resample."""
        return self._orig(frequency)

    def set_batch(self, polar_true, fill: str = "mean"):
        self.polar_true = polar_true
        if fill == "mean":                     # per-image mean (z-scored => ~0): a neutral, no-hole baseline
            mu = polar_true.mean(dim=(1, 2, 3), keepdim=True)
            self.baseline = mu.expand_as(polar_true)
        elif fill == "zero":
            import torch
            self.baseline = torch.zeros_like(polar_true)
        else:
            raise ValueError(f"fill must be mean|zero, got {fill!r}")

    def _patched(self, _x):
        # ignore the (cartesian) input; return the blended polar spectrum for the current mask/mode
        if self.mask is None:
            return self.polar_true
        m = self.mask
        if self.mode == "deletion":            # remove selected cells -> baseline
            return self.polar_true * (1.0 - m) + self.baseline * m
        return self.baseline * (1.0 - m) + self.polar_true * m   # insertion: add selected cells back


def topk_cell_mask(scores, frac: float, out_hw):
    """scores [B,G,G] (higher = selected first) -> upsampled cell mask [B,1,*out_hw] in {0,1}.

    All methods select at the SAME (rho, theta) cell granularity (G x G), so 'k% of cells'
    means the same thing for saliency, random, energy and anti — a fair comparison."""
    import torch
    import torch.nn.functional as F
    B, G, _ = scores.shape
    N = G * G
    k = int(round(frac * N))
    flat = scores.reshape(B, N)
    mask = torch.zeros_like(flat)
    if k > 0:
        idx = flat.topk(k, dim=1).indices
        mask.scatter_(1, idx, 1.0)
    mask = mask.reshape(B, 1, G, G)
    return F.interpolate(mask, size=tuple(out_hw), mode="nearest")


# --------------------------------------------------------------------------- #
# per-dataset E5
# --------------------------------------------------------------------------- #
def run_e5_on_loader(model, loader, device, *, fracs, do_insertion, max_batches,
                     fill="mean", verbose=True):
    """Returns a dict of curves + headline gaps for one loader."""
    import torch
    import torch.nn.functional as F
    from sfdet.explain.explainability import FrequencyExplainer

    branch = getattr(model, "frequency_branch", None)
    if branch is None:
        raise AttributeError("model has no .frequency_branch — E5 needs the FULL dual model.")
    if getattr(branch, "coords", "polar") != "polar":
        raise ValueError("polar-space occlusion requires freq_coords: polar (full.yaml default).")

    fx = FrequencyExplainer(model)             # pre-fusion (rho, theta) saliency; returns cam_raw
    occl = PolarOcclusion(branch)
    methods = ["saliency", "random", "energy", "anti"]
    modes = ["deletion"] + (["insertion"] if do_insertion else [])

    # accumulators: probs[mode][method][frac] -> [chunks]; fake-logit means on fakes likewise
    probs = {mo: {m: {f: [] for f in fracs} for m in methods} for mo in modes}
    fakeprob = {m: {f: [] for f in fracs} for m in methods}     # deletion only, on fake samples
    labels_all = []
    radial_sum = np.zeros(CELL_GRID, dtype=np.float64)          # mean saliency vs rho
    radial_n = 0

    model.eval()
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        spatial = batch["spatial"].to(device, non_blocking=True)
        freq = batch["frequency"].to(device, non_blocking=True)
        y = np.asarray([int(v) for v in batch["label"].tolist()])
        labels_all.extend(y.tolist())
        is_fake = (y == 1)

        # (1) per-sample fake saliency — clean pass, NO occlusion patch installed.
        out = fx(spatial, freq, normalize=False)               # one fwd + one bwd
        cam = out["cam_raw"].to(device).float()                # [B,16,16] >= 0, (rho rows, theta cols)
        model.zero_grad(set_to_none=True)
        radial_sum += cam.mean(dim=2).sum(dim=0).detach().cpu().numpy()   # mean over theta, sum over batch
        radial_n += cam.shape[0]

        # (2) ranking scores for every method, at cell granularity.
        with torch.no_grad():
            polar_true = occl.polar_of(freq)                   # [B,1,Hp,Wp]
            energy = F.adaptive_avg_pool2d(polar_true, CELL_GRID).squeeze(1)  # [B,16,16] brightness
        out_hw = polar_true.shape[-2:]
        scores = {"saliency": cam, "random": torch.rand_like(cam),
                  "energy": energy, "anti": -cam}

        # (3) occlusion forwards (no grad).
        with occl:
            occl.set_batch(polar_true, fill=fill)
            for mode in modes:
                occl.mode = mode
                for m in methods:
                    for f in fracs:
                        occl.mask = topk_cell_mask(scores[m], f, out_hw)
                        with torch.no_grad():
                            logits = model(spatial, freq)
                            if logits.ndim > 1 and logits.shape[-1] == 1:
                                logits = logits.squeeze(-1)
                            p = torch.sigmoid(logits.float()).cpu().numpy()
                        probs[mode][m][f].append(p)
                        if mode == "deletion" and is_fake.any():
                            fakeprob[m][f].append(p[is_fake])
        if verbose and (bi % 25 == 0):
            print(f"    batch {bi}  (n so far {len(labels_all)})", flush=True)

    labels = np.asarray(labels_all, dtype=int)
    return _summarize(labels, probs, fakeprob, fracs, modes,
                      radial_profile=(radial_sum / max(radial_n, 1)).tolist())


def _summarize(labels, probs, fakeprob, fracs, modes, radial_profile):
    from sfdet.metrics.classification import roc_auc

    def curve(mode, method):
        return {f: roc_auc(labels, np.concatenate(probs[mode][method][f])) for f in fracs}

    def area(c):                               # normalized area under AUC-vs-frac.
        xs = np.asarray(fracs, dtype=float)    # manual trapezoid (np.trapz was removed in NumPy 2.0,
        ys = np.asarray([c[f] for f in fracs], dtype=float)   # so this stays version-portable).
        integral = float(np.sum((ys[:-1] + ys[1:]) * 0.5 * np.diff(xs)))
        return integral / (xs[-1] - xs[0])

    deletion = {m: curve("deletion", m) for m in probs["deletion"]}
    insertion = {m: curve("insertion", m) for m in probs["insertion"]} if "insertion" in modes else {}

    auc_full = deletion["saliency"][fracs[0]]                  # deletion @ 0.0
    auc_nofreq = deletion["saliency"][fracs[-1]]               # deletion @ 1.0 (freq neutralized)

    headline = {
        "auc_full": auc_full,
        "auc_nofreq": auc_nofreq,
        "freq_reliance": float(auc_full - auc_nofreq),
        "deletion_area_gap_vs_random": float(area(deletion["random"]) - area(deletion["saliency"])),
        "deletion_area_gap_vs_energy": float(area(deletion["energy"]) - area(deletion["saliency"])),
    }
    if insertion:
        headline["insertion_area_gap_vs_random"] = float(
            area(insertion["saliency"]) - area(insertion["random"]))
    for k in REPORT_AT:
        if k in deletion["saliency"]:
            d_sal = auc_full - deletion["saliency"][k]
            d_rnd = auc_full - deletion["random"][k]
            d_eng = auc_full - deletion["energy"][k]
            headline[f"delta_auc@{k}"] = {
                "saliency_drop": float(d_sal), "random_drop": float(d_rnd),
                "energy_drop": float(d_eng), "gap_vs_random": float(d_sal - d_rnd),
                "gap_vs_energy": float(d_sal - d_eng)}

    # mean fake-prob on fake samples vs frac (the per-sample causal view; deletion).
    fake_curve = {m: {f: float(np.mean(np.concatenate(fakeprob[m][f]))) if fakeprob[m][f] else float("nan")
                      for f in fracs} for m in fakeprob}

    return {"n_samples": int(labels.size),
            "frac_grid": list(fracs),
            "headline": headline,
            "deletion_auc": deletion,
            "insertion_auc": insertion,
            "fake_prob_deletion": fake_curve,
            "radial_saliency_profile": radial_profile}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_report(all_results: dict):
    print("\n================ E5 frequency-saliency faithfulness probe ================")
    hdr = ("dataset", "N", "AUCfull", "AUCnofq", "freqRel",
           "dAUC@.15 sal", "rand", "GAP", "delArea(s-r)", "verdict")
    rows = [hdr]
    for name, r in all_results.items():
        h = r["headline"]
        d15 = h.get("delta_auc@0.15", {})
        gap = d15.get("gap_vs_random", float("nan"))
        rel = h["freq_reliance"]
        if rel < 0.01:
            verdict = "freq not load-bearing here"
        elif gap > 0.02:
            verdict = "saliency selective (supports C2)"
        elif gap > 0.0:
            verdict = "weakly selective"
        else:
            verdict = "no selectivity"
        rows.append((name, str(r["n_samples"]),
                     f"{h['auc_full']:.3f}", f"{h['auc_nofreq']:.3f}", f"{rel:+.3f}",
                     f"{d15.get('saliency_drop', float('nan')):.3f}",
                     f"{d15.get('random_drop', float('nan')):.3f}",
                     f"{gap:+.3f}",
                     f"{h['deletion_area_gap_vs_random']:+.3f}", verdict))
    w = [max(len(rw[i]) for rw in rows) for i in range(len(hdr))]
    for j, rw in enumerate(rows):
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(rw)))
        if j == 0:
            print("  ".join("-" * w[i] for i in range(len(w))))
    print("\nRead freqRel (= AUCfull - AUCnofq) FIRST: if ~0, the model barely uses the frequency")
    print("branch on that dataset, so a flat/zero GAP means 'frequency not load-bearing', NOT")
    print("'saliency unfaithful'. GAP>0 (and beating the energy control) supports C2's scoped")
    print("claim that the (rho,theta) saliency localizes discriminative spectral content.")
    print("==========================================================================\n")


def save_results(all_results: dict, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e5.json").write_text(json.dumps(all_results, indent=2))
    cols = ["dataset", "n_samples", "auc_full", "auc_nofreq", "freq_reliance",
            "dauc15_saliency", "dauc15_random", "dauc15_gap_vs_random", "dauc15_gap_vs_energy",
            "deletion_area_gap_vs_random", "deletion_area_gap_vs_energy",
            "insertion_area_gap_vs_random"]
    with open(out_dir / "e5.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, r in all_results.items():
            h = r["headline"]
            d15 = h.get("delta_auc@0.15", {})
            w.writerow([name, r["n_samples"], h["auc_full"], h["auc_nofreq"], h["freq_reliance"],
                        d15.get("saliency_drop"), d15.get("random_drop"),
                        d15.get("gap_vs_random"), d15.get("gap_vs_energy"),
                        h["deletion_area_gap_vs_random"], h["deletion_area_gap_vs_energy"],
                        h.get("insertion_area_gap_vs_random")])
    return {"json": str(out_dir / "e5.json"), "csv": str(out_dir / "e5.csv")}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="E5 frequency-saliency faithfulness probe (full model).")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "model" / "full.yaml"))
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--paths", default=str(REPO_ROOT / "paths.yaml"))
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--checkpoint", required=True, help="full-model best.pt (from Step 12 / E1)")
    ap.add_argument("--out-dir", default=None, help="where to write e5.json / e5.csv")
    ap.add_argument("--datasets", default=None,
                    help="comma-sep subset of loader names; default = FF++ val + all df40_* subsets")
    ap.add_argument("--fill", default="mean", choices=["mean", "zero"], help="occlusion baseline")
    ap.add_argument("--no-insertion", action="store_true", help="skip the insertion curve (faster)")
    ap.add_argument("--fracs", default=None, help="comma-sep occlusion fractions (override default grid)")
    ap.add_argument("--batch-size", type=int, default=None, help="override eval batch size")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches per dataset (sampling)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from sfdet.data.dataset import get_dataloaders
    from sfdet.engine.trainer import set_seed
    from sfdet.models.factory import build_model

    cfg = load_merged_config(args.config, args.model_config, args.data_config)
    variant = str(cfg.get("model", {}).get("variant", "full")).lower()
    if variant != "full":
        raise SystemExit(f"E5 requires the FULL dual model; model.variant is '{variant}'. "
                         f"Use --model-config configs/model/full.yaml.")
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    paths = yaml.safe_load(Path(args.paths).read_text()) or {}
    fracs = [float(x) for x in args.fracs.split(",")] if args.fracs else list(DEFAULT_FRACS)
    fracs = sorted(set(fracs) | {0.0, 1.0})    # ensure endpoints present for auc_full / auc_nofreq

    set_seed(int(cfg.get("seed", 1337)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # build model + load the converged full-model checkpoint
    model, _, _ = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] note: {len(missing)} missing / {len(unexpected)} unexpected keys "
              f"(expected 0 for a matching full-model checkpoint).")
    model = model.to(device)
    print(f"[load] full model from {args.checkpoint}  | device={device}")

    # loaders: FF++ val (in-domain reference) + the cross-dataset test loaders that have manifests.
    _, val_loader, test_loaders = get_dataloaders(
        cfg, paths, splits_dir=args.splits_dir, num_workers=args.num_workers,
        balance=None, verbose=True)
    candidates = {"faceforensics_c23_val": val_loader, **test_loaders}
    if args.datasets:
        wanted = [s.strip() for s in args.datasets.split(",")]
        loaders = {k: candidates[k] for k in wanted if k in candidates}
    else:                                       # default per the design: FF++ val + diffusion subsets
        loaders = {k: v for k, v in candidates.items()
                   if k == "faceforensics_c23_val" or k.startswith("df40")}
    if not loaders:
        raise SystemExit("no E5 loaders available (no manifests found). "
                         "FF++ val needs the FF++ manifest; df40_* need the DF40 manifests on this machine.")

    do_insertion = not args.no_insertion
    print(f"[e5] datasets: {list(loaders)} | fracs={fracs} | insertion={do_insertion} "
          f"| max_batches={args.max_batches}")

    all_results = {}
    for name, loader in loaders.items():
        print(f"\n[e5] === {name} ===", flush=True)
        all_results[name] = run_e5_on_loader(
            model, loader, device, fracs=fracs, do_insertion=do_insertion,
            max_batches=args.max_batches, fill=args.fill, verbose=True)

    print_report(all_results)
    if args.out_dir:
        paths_out = save_results(all_results, args.out_dir)
        print(f"[e5] wrote {paths_out['json']} and {paths_out['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
