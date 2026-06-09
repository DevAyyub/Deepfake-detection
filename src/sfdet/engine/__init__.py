"""sfdet.engine — training and evaluation loops.

  trainer.py    FF++ c23 training loop: AMP, optimizer/scheduler from config,
                checkpoint selection on validation AUC-ROC. Trains on c23
                four-manipulation data only.
  evaluator.py  Cross-dataset evaluation. Produces BOTH frame-level and
                video-level metrics per dataset and keeps them separated (never
                mixes the two in one column). Video score = reduction over the
                sampled frames of each source video.
  losses.py     Loss functions (BCE default), kept out of the model class so the
                objective is swappable without touching the architecture.

The engine is config-driven; swapping the model/ config (e.g. spatial_only)
changes what gets trained/evaluated with no loop changes, which is what keeps
ablations cheap. Implementations land in a later training/eval chat.
"""
