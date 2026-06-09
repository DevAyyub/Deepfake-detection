"""sfdet — Spatial-Frequency Deepfake Detection.

An explainable cross-dataset deepfake detector built around a dual
spatial-frequency encoder with integrated frequency saliency.

Architecture (see research notes 1.6 / 2.5):
  * a spatial branch (EfficientNet-B4),
  * a frequency branch (2D-FFT magnitude -> CNN),
  * joined by a SINGLE-STAGE cross-attention fusion, then a classifier head.

The single-stage join is a deliberate design constraint, not an arbitrary
choice: it keeps each branch modality-pure up to the point of fusion, which is
the precondition for computing a per-branch frequency-domain saliency map. See
``sfdet.models`` and ``sfdet.explain`` for the binding pre-fusion invariant that
follows from this.

Training is on FaceForensics++ (c23, four-manipulation) ONLY. Celeb-DF v2, DFDC,
WildDeepfake, and the DF40 diffusion subsets are evaluation-only. The anchor
evaluation metric is AUC-ROC, with accuracy and EER reported alongside, at both
frame and video granularity.

Layout: this is a src-layout package. Install editable with `pip install -e .`
after `pip install -r requirements.txt`.
"""

__version__ = "0.1.0"
