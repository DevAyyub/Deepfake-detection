"""sfdet.explain — integrated explainability for the dual-branch detector.

Two attribution views over a single sample, hence "integrated":

  gradcam_spatial.py  Grad-CAM over the spatial (EfficientNet-B4) branch —
                      a spatial heatmap on the face crop.
  freq_saliency.py    Class-discriminative saliency over the 2D-FFT magnitude
                      spectrum, indexed in polar (rho, theta) coordinates so it
                      reads as radial-frequency / orientation structure rather
                      than raw pixel grid position.
  hooks.py            Forward/backward activation hooks. The frequency-saliency
                      hook attaches to a PRE-FUSION frequency-branch layer (see
                      the binding invariant in sfdet.models). hooks live here so
                      the dependency runs explain -> models, never the reverse.
  spectral_utils.py   FFT helpers: fftshift, log1p magnitude, and Cartesian->polar
                      (rho, theta) binning shared by the frequency branch input
                      and the saliency visualisation.

Scope of the frequency-saliency claim (keep all wording here within this):
it LOCALISES the discriminative spectral content the model responded to for a
given sample. It is NOT advanced as a faithfulness guarantee. The attribution is
joint (Grad-CAM-style gradients on learned features), and Grad-CAM faithfulness
is contested in the literature, so do not describe the map as "faithful", as
"proof", or as "more faithful than" prior work. The contribution relative to
prior explainable detectors is categorical — attribution placed in the frequency
domain and separated per branch — not a faithfulness comparison.

A separate frequency-saliency faithfulness *check* (spectral occlusion / GT-mask,
reporting a delta) is a planned experiment; it quantifies and reports, it does
not license a faithfulness claim in the method's framing.

Implementations land in a later explainability chat; this package currently
records the layout, the pre-fusion hook target, and the claim scope.
"""
