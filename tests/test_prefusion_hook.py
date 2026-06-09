"""Guards the BINDING INVARIANT on the ASSEMBLED model: the frequency-saliency Grad-CAM target
executes BEFORE the cross-attention fusion. This is the INTEGRATION-TIME check — it does not
assume the invariant from the frequency-branch/explainer code; it asserts it on the wired
DeepfakeDetector via (1) structural containment, (2) a forward-order probe, (3) activation
identity, and (4) the DualExplainer producing a (rho, theta) map.

Needs a real PyTorch + timm install (it builds the actual detector), so the whole module is
marked needs_real_torch and is skipped under the conftest torch stub. Heavy imports are deferred
INTO the tests so the module still collects cleanly in CI.
"""
import pytest

pytestmark = pytest.mark.needs_real_torch


def _model():
    from sfdet.models.model import build_detector
    cfg = {"model": {"pretrained": False, "check_normalization": False},
           "data": {"image_size": 256}}
    return build_detector(cfg).eval()


def test_frequency_target_is_prefusion_submodule():
    """The saliency target lives inside frequency_branch and NOT inside fusion."""
    from sfdet.explain.explainability import _resolve_target
    m = _model()
    target = _resolve_target(m, "frequency")
    assert target is m.frequency_saliency_target            # the documented handle...
    assert target is m.frequency_branch.gradcam_target      # ...resolves to the branch property
    assert target is m.frequency_branch.tap_block           # ...which is the tap block
    freq_ids = {id(x) for x in m.frequency_branch.modules()}
    fusion_ids = {id(x) for x in m.fusion.modules()}
    assert id(target) in freq_ids                           # contained in the frequency branch
    assert id(target) not in fusion_ids                     # NOT contained in fusion


def test_forward_order_tap_strictly_before_fusion():
    """In the integrated forward, the tap's hook fires strictly before fusion's hook."""
    import torch
    m = _model()
    order = []
    h1 = m.frequency_saliency_target.register_forward_hook(lambda *a: order.append("tap"))
    h2 = m.fusion.register_forward_hook(lambda *a: order.append("fusion"))
    try:
        with torch.no_grad():
            m(torch.randn(2, 3, 256, 256), torch.randn(2, 1, 256, 256))
    finally:
        h1.remove()
        h2.remove()
    assert order == ["tap", "fusion"], order


def test_hooked_activation_equals_prefusion_tap_output():
    """The activation the saliency hook captures equals the frequency branch's own pre-fusion
    output (so it is the modality-pure tap, not a post-fusion tensor)."""
    import torch
    m = _model()
    seen = {}
    h = m.frequency_saliency_target.register_forward_hook(
        lambda _mod, _i, out: seen.__setitem__("A", out.detach()))
    freq = torch.randn(2, 1, 256, 256)
    try:
        with torch.no_grad():
            m(torch.randn(2, 3, 256, 256), freq)
            ff = m.frequency_branch(freq)                   # the branch's own pre-fusion tap
    finally:
        h.remove()
    assert seen["A"].shape == ff.shape == (2, m.frequency_branch.out_channels, 16, 16)
    assert torch.allclose(seen["A"], ff)


def test_dualexplainer_produces_prefusion_maps():
    """End to end: the explainer attributes through fusion back to the pre-fusion taps and
    yields a (rho, theta) frequency map + a spatial map."""
    import torch
    from sfdet.explain.explainability import DualExplainer
    m = _model()
    maps = DualExplainer(m)(torch.randn(2, 3, 256, 256), torch.randn(2, 1, 256, 256))
    assert maps["frequency"]["cam"].shape == (2, 16, 16)
    assert maps["spatial"]["cam"].shape[0] == 2
