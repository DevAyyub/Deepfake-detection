"""sfdet.metrics — detection metrics.

  classification.py  AUC-ROC (the ANCHOR metric — every results-table row is
                     compared in AUC), accuracy, and EER derived from the ROC.
                     Computed at both frame and video granularity.

AUC-ROC is the anchor because it is threshold-free and is the metric the
baselines and targets are reported in, so it is what must be comparable across
datasets. Accuracy and EER are reported alongside it. Implementations land in a
later training/eval chat.
"""
