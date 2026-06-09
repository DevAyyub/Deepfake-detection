"""sfdet.models.baselines — comparison models (NOT the proposed architecture).

  ojha_lc.py          E2: frozen CLIP ViT-L/14 image encoder + a single linear
                      (logistic-regression) probe. Trained on FF++ c23, evaluated
                      on all five datasets. Uses open-clip-torch.
  sbi_efficientnet.py E3: Self-Blended Images detector (EfficientNet backbone)
                      run on the DF40 four diffusion subsets.

These exist to populate the comparison rows of the results table on the same
c23 training protocol; they are deliberately separate from sfdet.models.model
so the proposed model stays uncluttered. Implementations land in a later
baselines chat.
"""
