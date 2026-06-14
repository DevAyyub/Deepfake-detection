"""fusion.py — single-stage cross-attention fusion + classification head.

This is the ONE point where the two modality-pure branches meet:
  * spatial features  [B, 1792, 8, 8]   (EfficientNet-B4 /32 map; 64 query tokens)
  * frequency tap     [B, 256, 16, 16]  (pre-fusion (rho, theta) map; 256 key/value tokens)

Direction: spatial = Query, frequency = Key/Value. The output is query-shaped, so each
spatial token is augmented by the frequency evidence most relevant to it (spatial backbone +
frequency modulation). The Q/K/V direction is irrelevant to explainability — BOTH Grad-CAM
saliency targets live on the *pre-fusion* branch features, upstream of this module.

MODALITY PURITY (C1 enabler; precondition for per-branch saliency): each branch's features are
a function of ONLY its own modality, and they first interact here, at this single block. Using
ONE stage is a deliberate choice to keep that property — multi-stage / multi-depth interleaving
(Qiao / TSFF-Net) mixes the modalities earlier, which would leave NO modality-pure frequency
representation to attribute over and would make C2's (rho, theta) saliency ill-defined.
This is explicitly NOT a capacity claim: multi-stage fusion is MORE expressive (more mixing,
more parameters); that expressiveness is exactly what it buys by sacrificing branch purity, so
it is available only as the ABL2 ablation. (Joint training still backprops through both
branches — that shapes the weights but not the forward separability the per-branch claim rests
on.) Cross-attention fusion is prior art; this module exists to ENABLE C2, not as novelty.

NOTE: the attention weights ([B, heads, 64, 256]) may be returned as a DIAGNOSTIC (which
frequency tokens a spatial token attends to). They are NOT the C2 saliency and NOT a
faithfulness signal — C2 is Grad-CAM over the pre-fusion frequency tap.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _FusionBlock(nn.Module):
    """One pre-norm cross-attention block: LN -> MHA(q; kv) -> +res -> LN -> FFN -> +res.
    `q` is the evolving query stream; `kv` is the fixed frequency key/value stream."""

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int, dropout: float,
                 residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)   # <1 handicaps the spatial query residual (Route-1 gate C)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, need_weights: bool = False):
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, attn_w = self.attn(q_n, kv_n, kv_n, need_weights=need_weights,
                                     average_attn_weights=False)
        q = self.residual_scale * q + self.dropout(attn_out)                       # residual carries spatial content
        q = q + self.dropout(self.ffn(self.norm_ffn(q)))
        return q, attn_w


class CrossAttentionFusion(nn.Module):
    """Single-stage cross-attention fusion (spatial Query, frequency Key/Value).

    forward(spatial_feat, freq_feat) -> pooled [B, d_model]  (the post-fusion representation
    fed to the head), optionally with attention weights [B, heads, Nq, Nkv] as a diagnostic.

    `n_blocks` > 1 stacks fusion blocks (an expressiveness/depth ablation). Note this stacks
    fusion DEPTH while the frequency key/value stream stays the pre-fusion tap, so the branch
    taps remain pure; the stronger ABL2 (interleaving fusion at multiple branch depths, which
    sacrifices that purity) is a separate architectural variant.
    """

    def __init__(self, spatial_channels: int = 1792, freq_channels: int = 256,
                 spatial_hw=(8, 8), freq_hw=(16, 16), d_model: int = 512, n_heads: int = 8,
                 ffn_ratio: int = 4, dropout: float = 0.1, n_blocks: int = 1,
                 return_attn: bool = False, residual_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.out_dim = d_model
        self.n_heads = n_heads
        self.return_attn = return_attn
        n_q = spatial_hw[0] * spatial_hw[1]      # 64
        n_kv = freq_hw[0] * freq_hw[1]           # 256

        # project each modality to the common d_model, then add learnable positional embeddings
        # (position is meaningful on both grids — the 8x8 spatial map and the (rho, theta) map).
        self.q_proj = nn.Linear(spatial_channels, d_model)
        self.kv_proj = nn.Linear(freq_channels, d_model)
        self.q_pos = nn.Parameter(torch.zeros(1, n_q, d_model))
        self.kv_pos = nn.Parameter(torch.zeros(1, n_kv, d_model))

        self.blocks = nn.ModuleList(
            [_FusionBlock(d_model, n_heads, ffn_ratio, dropout, residual_scale=residual_scale)
             for _ in range(n_blocks)])
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.q_pos, std=0.02)
        nn.init.trunc_normal_(self.kv_pos, std=0.02)

    def forward(self, spatial_feat, freq_feat, return_attn=None):
        return_attn = self.return_attn if return_attn is None else return_attn
        # flatten grids -> token sequences, project to d_model, add positional embeddings
        q = self.q_proj(spatial_feat.flatten(2).transpose(1, 2)) + self.q_pos     # [B, 64, d]
        kv = self.kv_proj(freq_feat.flatten(2).transpose(1, 2)) + self.kv_pos     # [B, 256, d]

        attn_w = None
        last = len(self.blocks) - 1
        for i, blk in enumerate(self.blocks):
            need = bool(return_attn) and (i == last)
            q, w = blk(q, kv, need_weights=need)
            if need:
                attn_w = w
        pooled = q.mean(dim=1)                                                    # GAP -> [B, d]
        return (pooled, attn_w) if return_attn else pooled


class ClassificationHead(nn.Module):
    """Post-fusion head: Dropout -> Linear -> one logit. Same minimal shape as the spatial-only
    head, so the C1 ablation compares representations, not head capacity.

    include_spatial=True additionally concatenates GAP(spatial_feat) so the dual model is an
    information SUPERSET of the spatial-only arm by construction (any dual-vs-spatial-only gap
    is then attributable to the frequency branch, not to a spatial bottleneck at projection)."""

    def __init__(self, d_model: int = 512, spatial_channels: int = 1792, dropout: float = 0.3,
                 num_classes: int = 1, include_spatial: bool = False):
        super().__init__()
        self.include_spatial = include_spatial
        in_dim = d_model + (spatial_channels if include_spatial else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, 1 if num_classes <= 2 else num_classes)

    def forward(self, fused_pooled, spatial_feat=None):
        x = fused_pooled
        if self.include_spatial:
            assert spatial_feat is not None, "include_spatial=True needs spatial_feat"
            x = torch.cat([x, spatial_feat.flatten(2).mean(-1)], dim=1)            # + GAP spatial
        return self.fc(self.drop(x)).squeeze(-1)                                   # [B]


# --------------------------------------------------------------------------- #
# Builders (read the merged config; mirror the other branch builders)
# --------------------------------------------------------------------------- #
def build_fusion(cfg: dict, *, spatial_channels: int = 1792, freq_channels: int = 256,
                 spatial_hw=(8, 8), freq_hw=(16, 16)) -> CrossAttentionFusion:
    f = cfg.get("model", {}).get("fusion", {})
    return CrossAttentionFusion(
        spatial_channels=spatial_channels, freq_channels=freq_channels,
        spatial_hw=spatial_hw, freq_hw=freq_hw,
        d_model=int(f.get("d_model", 512)), n_heads=int(f.get("n_heads", 8)),
        ffn_ratio=int(f.get("ffn_ratio", 4)), dropout=float(f.get("dropout", 0.1)),
        n_blocks=int(f.get("n_blocks", 1)),                 # ABL2: stack > 1
        return_attn=bool(f.get("return_attn", False)),
        residual_scale=float(f.get("residual_scale", 1.0)),  # Route-1 gate; 1.0 = unchanged
    )


def build_head(cfg: dict, *, d_model: int = 512, spatial_channels: int = 1792) -> ClassificationHead:
    m = cfg.get("model", {})
    return ClassificationHead(
        d_model=d_model, spatial_channels=spatial_channels,
        dropout=float(m.get("dropout", 0.3)), num_classes=int(m.get("num_classes", 1)),
        include_spatial=bool(m.get("head_includes_spatial", False)),
    )


# --------------------------------------------------------------------------- #
# Self-test (random weights; runs under real torch and the shape-stub)
# --------------------------------------------------------------------------- #
def _selftest():
    B = 2
    spatial = torch.randn(B, 1792, 8, 8)        # dummy spatial-branch output
    freq = torch.randn(B, 256, 16, 16)          # dummy frequency-branch pre-fusion tap

    fusion = CrossAttentionFusion()
    pooled = fusion(spatial, freq)
    print(f"fusion: spatial {tuple(spatial.shape)} + freq {tuple(freq.shape)} -> pooled "
          f"{tuple(pooled.shape)} (d_model={fusion.d_model})")
    assert tuple(pooled.shape) == (B, fusion.d_model)

    pooled2, attn = fusion(spatial, freq, return_attn=True)
    print(f"attention weights (diagnostic, NOT C2 saliency): {tuple(attn.shape)}")
    assert tuple(attn.shape) == (B, fusion.n_heads, 64, 256)

    head = build_head({})
    logits = head(pooled)
    print(f"head (GAP-fused): logits {tuple(logits.shape)}")
    assert tuple(logits.shape) == (B,)

    head_sup = ClassificationHead(include_spatial=True)
    logits_sup = head_sup(pooled, spatial_feat=spatial)
    print(f"head (concat GAP-spatial superset): logits {tuple(logits_sup.shape)}")
    assert tuple(logits_sup.shape) == (B,)

    deep = CrossAttentionFusion(n_blocks=2)     # ABL2 depth toggle
    assert tuple(deep(spatial, freq).shape) == (B, deep.d_model)
    print("ABL2 n_blocks=2 -> pooled", tuple(deep(spatial, freq).shape))

    print("OK: fusion + head shapes verified")


if __name__ == "__main__":
    _selftest()
