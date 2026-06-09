"""sfdet.data — datasets, transforms, sampling, and video grouping.

Implemented as a single consolidated module:

  dataset.py   The data layer for all five datasets. Contains:
                 * FFTMagnitude — deterministic 2D-FFT-magnitude transform for the
                   frequency branch (log1p + per-image z-score), computed on the
                   SAME crop as the spatial view so no extra resampling artifacts
                   enter the spectrum.
                 * DualViewTransform / AugConfig — spatial-only augmentation for the
                   EfficientNet branch; the frequency input is a fixed transform.
                 * BaseDeepfakeDataset + collate_batch — read pre-extracted face
                   crops (via the manifest) and yield {spatial, frequency, label,
                   source_video_id, ...}, the id giving video-level grouping.
                 * balanced real/fake sampling, FF++ identity-disjoint split logic,
                   DF40 subset/domain grouping, and get_dataloaders() (the entry
                   point returning train/val/test loaders).

Filesystem roots come from paths.yaml via load_configs, never hardcoded.
"""
