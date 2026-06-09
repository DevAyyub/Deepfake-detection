"""sfdet.models — model components for the dual spatial-frequency detector.

Module slots (research notes 2.5), each implemented in its own file so that
ablations are toggled by config rather than by editing code:

  spatial_branch.py    EfficientNet-B4 spatial encoder.
  frequency_branch.py  2D-FFT magnitude -> CNN encoder. EXPOSES its last
                       pre-fusion feature map for the frequency-saliency hook.
  fusion.py            Single-stage cross-attention fusion (spatial features as
                       the query; frequency features as keys/values).
  classifier_head.py   Pooling + linear real/fake head on the fused representation.
  detector.py          Assembles the branches + fusion + head and reads the model
                       config to switch between variants (dual / spatial-only /
                       frequency-only / fusion variants) for ablations.
  baselines/           Comparison models that are NOT the proposed architecture
                       (Ojha frozen-CLIP linear probe; SBI EfficientNet).

================================ BINDING INVARIANT ============================
The frequency-saliency target layer MUST be a PRE-FUSION layer of the frequency
branch. The saliency map is computed on the frequency branch's own features
*before* any cross-attention with the spatial branch mixes the two modalities.

Why this is load-bearing, not stylistic: after fusion, a feature map is a joint
spatial+frequency representation. A saliency map taken there would be a joint
attribution and could not be described as "spectral content the model relied
on" — it would silently undercut the separation of the two contributions
(dual encoder vs. frequency attribution) that the method claims. A post-fusion
hook fails silently: it still produces a plausible-looking heatmap. So the
frequency branch is structured to keep its pre-fusion layers cleanly reachable,
and ``tests/test_prefusion_hook.py`` guards that the registered layer is in fact
pre-fusion.

The fusion stage is kept single-stage for the same reason: each branch stays
modality-pure up to fusion. Multi-stage fusion is available only as an explicit
ablation, never as the default path.
==============================================================================

Framing note (carry into any docstrings/strings here): the cross-attention
fusion is established technique, not a standalone novelty — it is the enabler of
the frequency attribution. Describe the spectral behaviour of generators as a
mixture of shared and distinct signatures whose balance tracks how much
convolutional upsampling sits in the generation path; do not frame it as a
latent-vs-pixel split or as qualitatively different mechanisms.

Implementations of the modules above land in later component chats; this package
currently defines only the layout and the invariant they must respect.
"""
