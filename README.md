# Explainable Cross-Dataset Deepfake Detection
### Dual Spatial–Frequency Architecture with Integrated Frequency Saliency

A deepfake detector that pairs a spatial encoder with a frequency-domain encoder
and adds attribution in **both** domains. Python package: `sfdet`
(spatial-frequency detection).

---

## What this is

The model has three parts (see research notes §1.6 / §2.5):

- **Spatial branch** — EfficientNet-B4 over the face crop.
- **Frequency branch** — a CNN over the 2D-FFT magnitude spectrum of the same crop.
- **Single-stage cross-attention fusion** joining them, then a real/fake head.

On top of the detector it produces two attribution views of a single sample:
Grad-CAM over the spatial branch, and a class-discriminative **saliency map over
the FFT magnitude spectrum** indexed in polar `(ρ, θ)` coordinates.

### How to talk about it (framing rules — please keep to these in code, docstrings, and the paper)

These are deliberate scoping choices, baked in so comments and commit messages
stay consistent with the write-up:

- The cross-attention fusion is **established technique**, not a standalone
  contribution. Its job here is to *enable* the per-branch frequency attribution.
  Avoid "novel" for the fusion and avoid bare "first" for the method — state what
  it *does* (a capability statement), not a priority claim.
- The frequency-saliency map **localises the discriminative spectral content**
  the model responded to. It is **not** advanced as "faithful", as "proof", or as
  "more faithful than" prior explainable detectors — the attribution is joint and
  Grad-CAM faithfulness is contested. The advantage over prior work is
  *categorical* (attribution placed in the frequency domain, separated per
  branch), not a faithfulness comparison.
- Describe generator spectra as a **mixture of shared and distinct signatures**
  whose balance tracks **how much convolutional upsampling sits in the generation
  path**. Do **not** frame it as a "latent vs pixel" split or as "qualitatively
  different mechanisms" (a pixel-space DDPM out-scores a latent DiT in the
  reference numbers, which sinks the latent-vs-pixel story).
- WildDeepfake is the **real-world, internet-collected** axis — never call it
  cross-family or diffusion.

---

## ⚠️ Binding invariant — the frequency-saliency hook is PRE-FUSION

**The frequency-saliency target layer must be a pre-fusion layer of the frequency
branch.** The saliency map is computed on the frequency branch's own features
*before* cross-attention mixes in the spatial branch.

A post-fusion hook would yield a *joint* attribution and quietly break the
separation between the dual encoder and the frequency attribution that the method
claims — and it would fail **silently**, still producing a plausible heatmap. The
frequency branch is therefore structured to keep its pre-fusion layers cleanly
reachable, the fusion is kept single-stage (each branch modality-pure up to
fusion), and `tests/test_prefusion_hook.py` guards that the registered layer is
actually pre-fusion. Multi-stage fusion exists only as an explicit ablation.

---

## Data & evaluation protocol

**Train on FaceForensics++ (c23, standard four-manipulation) ONLY.** Everything
else is evaluation-only:

| Axis | Datasets |
|---|---|
| Conventional lab | Celeb-DF v2, DFDC (full DFDC, not DFDCP) |
| Real-world | WildDeepfake |
| Diffusion (test-only) | DF40 subsets: SD-2.1, **DDPM (pixel-space, not DDIM)**, PixArt-α, DiT-XL/2 |

**AUC-ROC is the anchor metric** (threshold-free, and what every comparison row is
reported in). Accuracy and EER are reported alongside. Report **both frame-level
and video-level** numbers and keep them separated — frame-vs-video is an open
decision to settle from the first results, not something to mix in one column.

Datasets are **not** in this repo. They live on the remote GPU and are referenced
through `paths.yaml` (see setup). DF40 ships pre-extracted crops; the other four
are cropped once by `scripts/preprocess_faces.py`.

---

## Setup

This is a **src-layout** package. There are **two** requirements files on purpose.

### 1. Training / evaluation environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # editable install of `sfdet`
python verify_env.py             # imports + CUDA + a dummy EfficientNet-B4 forward pass
```

On Linux/Colab the default PyPI `torch` wheel is already a CUDA build, so
`requirements.txt` installs GPU torch as-is. If you hit a driver/CUDA mismatch,
install `torch`/`torchvision` from the [PyTorch index](https://pytorch.org/get-started/locally/)
first (matching your runtime), then run the rest. Keep `torch` and `torchvision`
on matching minor versions.

### 2. Face-extraction environment (isolated — see below)

```bash
python -m venv .venv-preprocess
source .venv-preprocess/bin/activate
pip install -r requirements-preprocess.txt
python scripts/preprocess_faces.py --dataset faceforensics_c23 ...
```

**Why two files:** the MTCNN detector (`facenet-pytorch`) hard-pins
`torch<2.3` / `numpy<2.0`, which would drag the whole training stack back to
early-2024 versions. Face extraction is a one-time job that writes plain image
crops to disk — there is no torch-version coupling back to training — so it gets
its own environment instead of holding the main stack hostage.

There is no `grad-cam` package: `pytorch-grad-cam` declares the GUI `opencv-python`
(which clashes with the headless build used here), and the pre-fusion hook plus
the custom `(ρ, θ)` saliency are bespoke anyway, so Grad-CAM is hand-rolled in
`src/sfdet/explain/`.

### 3. Paths

```bash
cp paths.example.yaml paths.yaml   # paths.yaml is gitignored; edit per machine
```

`sfdet.utils.paths` reads `paths.yaml` and injects the dataset roots at runtime,
so no absolute path or machine-specific layout ever enters version control.

---

## Repository layout

`configs/base.yaml` holds the shared defaults and constraints; per-dataset,
per-model, and per-experiment configs layer on top
(`base ← data/ ← model/ ← experiment/`). Ablations — spatial-only,
frequency-only, fusion variants — are toggled by swapping the `model/` config,
with no code changes. The full annotated tree is in the project chat; the model,
data, explainability, and engine module files are implemented in later component
chats, and per-variant config files are added alongside the code that reads them.

---

## Experiments

The Day-1 experiment set (E1–E5) and its two open decisions are described in
[`experiments/README.md`](experiments/README.md). Results dumps under
`experiments/results/` are gitignored; the small `MODELS.md` ledger that records
where final weights live (and their hashes) is tracked.

---

## Sharing trained weights

Weights are gitignored (`*.pth` etc.). The final checkpoints needed for paper
figures are stored in shared cloud storage (Google Drive, already mounted on
Colab), **not** committed and **not** in Git LFS — see the rationale in the
project chat. Record each canonical checkpoint's Drive path, `sha256`, and
producing commit hash in `experiments/results/MODELS.md`.
