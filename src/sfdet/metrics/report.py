"""sfdet.metrics.report — paste-ready results-table emitter (for §2.9 / the paper).

Turns the evaluator's output into a labeled table. Non-negotiables baked in:
  * every number carries its COMPRESSION/protocol and GRANULARITY (frame|video);
  * every number is marked as THIS WORK's own measurement — the emitter only ever
    prints values measured here and never transcribes a literature/target number;
  * DF40 per-subset rows are explicitly this work's measurements under FF++ c23,
    NOT the DF40 paper's Xception baselines.

Pure string formatting over the eval dict (no torch), so it is deterministic and
unit-testable. Consumes the dict produced by sfdet.engine.evaluator.evaluate_all
(or an eval.json loaded from disk).
"""
from __future__ import annotations

# Compression / protocol label per dataset. These encode the commensurability-
# critical axis; override via emit_results_table(compression_map=...).
DEFAULT_COMPRESSION = {
    "faceforensics_c23": "c23",
    "celebdf_v2": "CDF-v2",
    "dfdc": "DFDC-full",          # full DFDC, NOT DFDCP (locked project decision)
    "wilddeepfake": "WildDeepfake",
}
DF40_COMPRESSION = "c23-pipeline"  # DeepfakeBench-processed crops, scored by our c23-trained model

AXIS_ORDER = ["faceforensics_c23", "celebdf_v2", "dfdc", "wilddeepfake"]
_PRETTY = {
    "faceforensics_c23": "FaceForensics++ (in-domain)",
    "celebdf_v2": "Celeb-DF v2",
    "dfdc": "DFDC",
    "wilddeepfake": "WildDeepfake",
}
_DF40_SUBSET = {
    "stable_diffusion_2_1": "Stable Diffusion 2.1",
    "ddpm": "DDPM",
    "pixart_alpha": "PixArt-\u03b1",
    "dit_xl_2": "DiT-XL/2",
}


def _pretty(name: str) -> str:
    if name in _PRETTY:
        return _PRETTY[name]
    if name.startswith("df40_"):
        subset, _, domain = name[len("df40_"):].rpartition("_")
        return f"DF40: {_DF40_SUBSET.get(subset, subset)} ({domain})"
    return name


def _compression_for(name: str, comp_map: dict) -> str:
    if name in comp_map:
        return comp_map[name]
    if name.startswith("df40_"):
        return DF40_COMPRESSION
    return "?"


def _ordered(names):
    head = [n for n in AXIS_ORDER if n in names]
    tail = sorted(n for n in names if n not in AXIS_ORDER)   # df40_* etc.
    return head + tail


def _fmt(x, percent: bool) -> str:
    if x is None or (isinstance(x, float) and x != x):       # None or NaN
        return "n/a"
    return f"{x * 100:.2f}" if percent else f"{x:.4f}"


def _row_flag(r: dict) -> str:
    """'‡' when the harness flagged this dataset (number not trustworthy)."""
    if not (r.get("both_classes_frame", True) and r.get("both_classes_video", True)):
        return "\u2021"
    if r.get("mixed_label_videos", 0):
        return "\u2021"
    if (r.get("leakage", {}) or {}).get("identity_overlap_with_train"):
        return "\u2021"
    return ""


def _infer_reduce(results: dict) -> str:
    for r in results.values():
        if "video_reduce" in r:
            return r["video_reduce"]
    return "mean"


def _build_rows(results: dict, comp_map: dict, percent: bool, source_label: str):
    rows = []
    any_df40 = any_flag = False
    for name in _ordered(results):
        r = results[name]
        comp = _compression_for(name, comp_map)
        pretty = _pretty(name)
        is_df40 = name.startswith("df40_")
        any_df40 = any_df40 or is_df40
        flag = _row_flag(r)
        any_flag = any_flag or bool(flag)
        src = f"{source_label} \u2020" if is_df40 else source_label
        for gran in ("frame", "video"):
            m = r[gran]
            rows.append([pretty + flag, comp, gran,
                         _fmt(m["auc"], percent), _fmt(m["acc"], percent), _fmt(m["eer"], percent), src])
    return rows, any_df40, any_flag


def _md_table(rows) -> str:
    header = ["Dataset", "Compression", "Granularity", "AUC-ROC", "Acc", "EER", "Source"]
    table = [header] + rows
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    out = ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(header)) + " |",
           "|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def _latex_table(rows) -> str:
    header = ["Dataset", "Compression", "Granularity", "AUC-ROC", "Acc", "EER", "Source"]
    esc = lambda s: s.replace("_", r"\_").replace("\u03b1", r"$\alpha$")  # noqa: E731
    lines = [r"\begin{tabular}{lll rrr l}", r"\toprule",
             " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def emit_results_table(eval_out: dict, *, fmt: str = "md", compression_map: dict = None,
                       percent: bool = False, title: str = "Spatial-only (E1) — cross-dataset results",
                       train_protocol: str = "FaceForensics++ c23, four-manipulation",
                       source_label: str = "this work") -> str:
    """Render the §2.9 results table. `eval_out` is evaluate_all's output (or its
    'results' dict). `fmt` is 'md' or 'latex'. Every number is labeled with its
    compression and granularity and attributed to this work."""
    results = eval_out.get("results", eval_out)
    sanity = eval_out.get("sanity", {}) if isinstance(eval_out, dict) else {}
    comp_map = {**DEFAULT_COMPRESSION, **(compression_map or {})}
    video_reduce = sanity.get("video_reduce") or _infer_reduce(results)
    scale = "x100, percent" if percent else "in [0,1]"

    rows, any_df40, any_flag = _build_rows(results, comp_map, percent, source_label)
    if fmt == "md":
        table = _md_table(rows)
    elif fmt == "latex":
        table = _latex_table(rows)
    else:
        raise ValueError(f"fmt must be 'md' or 'latex', got {fmt!r}")

    head = (f"## {title}\n\n"
            f"Trained on: {train_protocol}. Anchor metric: AUC-ROC ({scale}). "
            f"Video-level scores reduce each source video's frame scores by **{video_reduce}**. "
            f"All values below are measured in this work.\n")

    notes = ["", "Notes:",
             f"- All values are **{source_label}**'s own measurements (model trained on "
             f"{train_protocol}, evaluated on each dataset as distributed). No number is "
             "transcribed from another paper.",
             "- Do not compare across compressions or granularities: c23 \u2260 c40/raw; "
             "DFDC (full) \u2260 DFDCP; video-level (e.g. SBI) and frame-level (e.g. Qiao) "
             "numbers are not interchangeable."]
    if any_df40:
        notes.append("- \u2020 DF40 per-subset numbers are **this work's measurements under the "
                     "FF++ c23 protocol**, not the DF40 paper's Xception baselines.")
    if any_flag:
        notes.append("- \u2021 flagged by the eval harness (single-class / identity overlap / "
                     "mixed-label) \u2014 treat the number as not trustworthy; see the eval sanity report.")
    notes.append("- 'n/a' = metric undefined (a single-class split has no ROC).")

    return head + "\n" + table + "\n" + "\n".join(notes) + "\n"
