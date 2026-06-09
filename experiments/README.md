# Experiments

The Day-1 experiment set from research notes §1.10. Every experiment trains on
**FaceForensics++ c23 (four-manipulation) only** and is reported in **AUC-ROC**
(anchor), with accuracy and EER alongside, at **frame and video** granularity.

Each experiment is a config under `configs/experiment/` layered on
`base ← data/ ← model/`. The commands below are the intended interface; the
`scripts/` entry points and the per-experiment config files are implemented in
later chats. Results write to `experiments/results/` (gitignored).

| ID | What | Model / source | Eval datasets |
|----|------|----------------|---------------|
| **E1** | The proposed detector — fills every cell of the results table | `sfdet` dual spatial–frequency | all five |
| **E2** | Ojha-style linear probe: frozen CLIP ViT-L/14 + linear head | `models/baselines/ojha_lc.py` | all five |
| **E3** | SBI (Self-Blended Images) on the diffusion subsets | `models/baselines/sbi_efficientnet.py` | DF40 ×4 |
| **E4** | RECCE re-run on **c23** — *gated by open decision (b)* | external RECCE | per its protocol |
| **E5** | Frequency-saliency **faithfulness check** (quantitative) | `sfdet.explain` | held-out subset |

### E1 — proposed model
```bash
python scripts/train.py    --config configs/experiment/e1_main.yaml
python scripts/evaluate.py --config configs/experiment/e1_main.yaml --ckpt <best.pth>
```
Produces both frame- and video-level AUC/accuracy/EER for Celeb-DF v2, DFDC,
WildDeepfake, and each DF40 diffusion subset.

### E2 — Ojha frozen-CLIP linear probe
```bash
python scripts/train.py    --config configs/experiment/e2_ojha_lc.yaml
python scripts/evaluate.py --config configs/experiment/e2_ojha_lc.yaml --ckpt <best.pth>
```

### E3 — SBI on DF40 diffusion subsets
```bash
python scripts/train.py    --config configs/experiment/e3_sbi_df40.yaml
python scripts/evaluate.py --config configs/experiment/e3_sbi_df40.yaml --ckpt <best.pth>
```

### E4 — RECCE on c23 (conditional)
Run only if open decision (b) is "yes". RECCE is an external codebase; reproduce
on the c23 protocol and drop its numbers into the table.

### E5 — frequency-saliency faithfulness check
```bash
python scripts/evaluate.py --config configs/experiment/e5_faithfulness.yaml --ckpt <e1_best.pth>
```
Spectral-occlusion (or GT-mask where available) over the FFT magnitude:
perturb the spectral regions the `(ρ, θ)` saliency highlights and report the
**Δ** in detector output. This **quantifies and reports** behaviour — it does not
license calling the saliency "faithful" in the method's framing.

### Ablations (`scripts/run_ablation.py`, swap the `model/` config)
- `spatial_only` / `frequency_only` — branch-alone baselines.
- `fusion_input_concat` — FFT as a 4th input channel (SFIAD-style), vs the
  cross-attention default.
- `fusion_two_stage` — multi-stage fusion vs single-stage. (Note: ablation only;
  the headline model stays single-stage to preserve the pre-fusion saliency
  invariant.)
- phase-processing vs magnitude-only (magnitude-only is the default).

The value of the dual branch is judged **relatively** here — full model vs
`spatial_only` on the diffusion subsets — not against any fixed printed bar.

---

## Open decisions (settle from first results)

**(a) Frame- vs video-level granularity for the proposed model.** Both are
produced; which one anchors the headline comparison is decided after seeing E1.
Until then, never collapse frame and video numbers into a single column.

**(b) Re-run RECCE on c23?** Decides whether E4 runs. If yes, RECCE is reproduced
on the c23 protocol for a like-for-like row; if no, E4 is dropped and the table
notes RECCE is cited from its original compression setting.

## Weights ledger

Final checkpoints are **not** committed (gitignored). Record each canonical
checkpoint's cloud path, `sha256`, and producing commit hash in
`experiments/results/MODELS.md` (tracked) so a result is always traceable to the
exact weights and code that produced it.
