"""sfdet.engine.evaluator — the cross-dataset evaluation + metrics harness.

Every paper number flows through here, so it is deliberately conservative:

  * Per dataset, SEPARATELY: AUC-ROC (the anchor), accuracy, EER — at BOTH
    frame level and video level. The video reduction (mean/median over a source
    video's frame probabilities) is explicit and recorded in the output.
  * Leakage / sanity, per dataset and globally: real/fake balance (frame & video),
    both-classes-present, mixed-label-video detection, and — for the in-domain
    FF++ test row — an identity-overlap check against the training split (must be
    0). The cross-datasets are separate corpora, so "no test data seen in
    training" holds by construction; the harness states that and verifies the one
    case (FF++ in-domain) where an identity-disjoint split is the assumption.

Everything except the inference pass is torch-free and operates on numpy arrays +
metadata lists, so the metric math is unit-testable without a GPU. ``infer_loader``
is the only function that touches torch (imported lazily inside it).

Model contract: a callable returning one logit per sample for ``batch["spatial"]``
(the spatial-only SpatialClassifier; the dual model plugs in via ``predict``).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sfdet.metrics.classification import binary_metrics

REAL, FAKE = 0, 1
# canonical display order; df40_* rows are appended sorted after these
AXIS_ORDER = ["faceforensics_c23", "celebdf_v2", "dfdc", "wilddeepfake"]


def _origin_id(source_video_id: str) -> str:
    """Mirror of sfdet.data.dataset._ffpp_origin_id (kept local so this module
    stays torch-free): 'Deepfakes__033_097' -> '033', 'original__050' -> '050'."""
    s = source_video_id.split("__", 1)[1] if "__" in source_video_id else source_video_id
    return s.split("_")[0]


def balance(labels) -> dict:
    labels = np.asarray(labels).astype(int)
    n = int(labels.size)
    n_fake = int((labels == FAKE).sum())
    n_real = int((labels == REAL).sum())
    return {"n": n, "n_real": n_real, "n_fake": n_fake,
            "frac_fake": (n_fake / n) if n else 0.0}


def reduce_video(video_ids, probs, labels, reduce: str = "mean"):
    """Aggregate frame probabilities to one score per source video.

    Returns (video_labels, video_scores, video_ids, mixed_label_videos). A video
    is expected to be single-label; any video seen with >1 distinct label is
    reported in ``mixed`` (a data/id-collision bug) and uses its first label."""
    if reduce not in ("mean", "median"):
        raise ValueError(f"video_reduce must be 'mean' or 'median', got {reduce!r}")
    reducer = np.mean if reduce == "mean" else np.median
    agg: dict = {}
    for v, p, l in zip(video_ids, probs, labels):
        d = agg.setdefault(v, {"p": [], "l": int(l)})
        d["p"].append(float(p))
        if int(l) != d["l"]:
            d["mixed"] = True
    vids = list(agg)
    v_scores = np.array([reducer(agg[v]["p"]) for v in vids], dtype=float)
    v_labels = np.array([agg[v]["l"] for v in vids], dtype=int)
    mixed = [v for v in vids if agg[v].get("mixed")]
    return v_labels, v_scores, vids, mixed


def dataset_report(probs, labels, video_ids, *, video_reduce: str = "mean",
                   threshold: float = 0.5) -> dict:
    """Frame- and video-level AUC/acc/EER + balance + sanity flags for one dataset."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).astype(int)

    frame_m = binary_metrics(labels, probs, threshold)
    v_labels, v_scores, _, mixed = reduce_video(video_ids, probs, labels, video_reduce)
    video_m = binary_metrics(v_labels, v_scores, threshold)

    return {
        "frame": {**frame_m, **balance(labels)},
        "video": {**video_m, **balance(v_labels)},
        "video_reduce": video_reduce,
        "both_classes_frame": bool(np.unique(labels).size >= 2),
        "both_classes_video": bool(np.unique(v_labels).size >= 2),
        "mixed_label_videos": len(mixed),
    }


def leakage_report(name: str, video_ids, *, train_origin_ids=None,
                   is_indomain: bool = False) -> dict:
    """Identity-overlap check. For the in-domain FF++ test row, verify none of its
    source-video identities appear in the training split (must be 0). Cross-datasets
    are separate corpora -> no identity-overlap assumption to violate."""
    out = {"dataset": name, "is_indomain": is_indomain}
    if is_indomain and train_origin_ids is not None:
        test_ids = {_origin_id(v) for v in video_ids}
        overlap = sorted(test_ids & set(train_origin_ids))
        out["identity_overlap_with_train"] = len(overlap)
        out["overlap_sample"] = overlap[:10]
        out["ok"] = len(overlap) == 0
    else:
        out["identity_overlap_with_train"] = None
        out["ok"] = True
    return out


def infer_loader(model, loader, device, *, amp: bool = False, predict=None,
                 max_batches=None) -> dict:
    """Run the model over a loader. Returns probs/labels/video_ids/subsets/domains
    as numpy/lists. The only torch-touching function (torch imported lazily)."""
    import torch

    model.eval()
    if predict is None:
        def predict(m, b):
            return m(b["spatial"].to(device, non_blocking=True))

    prob_chunks, labels, vids, subsets, domains = [], [], [], [], []
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16) if amp \
        else torch.autocast(device_type=device.type, enabled=False)
    with torch.no_grad(), autocast:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            logits = predict(model, batch)
            if logits.ndim > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            prob_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.extend(int(x) for x in batch["label"].tolist())
            vids.extend(batch["source_video_id"])
            n = len(batch["source_video_id"])
            subsets.extend(batch.get("subset", [""] * n))
            domains.extend(batch.get("domain", [""] * n))

    probs = np.concatenate(prob_chunks) if prob_chunks else np.empty(0)
    return {"probs": probs, "labels": np.asarray(labels, dtype=int),
            "video_ids": vids, "subsets": subsets, "domains": domains}


def evaluate_all(model, test_loaders: dict, device, *, video_reduce: str = "mean",
                 threshold: float = 0.5, train_origin_ids=None,
                 train_dataset: str = "faceforensics_c23", amp: bool = False,
                 max_batches=None, verbose: bool = True) -> dict:
    """Evaluate every loader separately; attach per-dataset leakage/sanity. Returns
    {'results': {name: report}, 'sanity': {...}}."""
    results: dict = {}
    for name, loader in test_loaders.items():
        pred = infer_loader(model, loader, device, amp=amp, max_batches=max_batches)
        if pred["probs"].size == 0:
            if verbose:
                print(f"[evaluate] {name}: 0 samples — skipped")
            continue
        is_indomain = (name == train_dataset)
        rep = dataset_report(pred["probs"], pred["labels"], pred["video_ids"],
                             video_reduce=video_reduce, threshold=threshold)
        rep["leakage"] = leakage_report(name, pred["video_ids"],
                                        train_origin_ids=train_origin_ids,
                                        is_indomain=is_indomain)
        rep["is_indomain"] = is_indomain
        results[name] = rep
        if verbose:
            f, v = rep["frame"], rep["video"]
            print(f"[evaluate] {name:<28} frame AUC {f['auc']:.4f} | "
                  f"video AUC {v['auc']:.4f}  ({v['n']} videos, {f['frac_fake']*100:.0f}% fake)")

    sanity = summarize_sanity(results, train_dataset=train_dataset, video_reduce=video_reduce)
    return {"results": results, "sanity": sanity}


def summarize_sanity(results: dict, *, train_dataset: str = "faceforensics_c23",
                     video_reduce: str = "mean") -> dict:
    """Global leakage/sanity rollup (torch-free, so it is unit-testable)."""
    overlap = (results.get(train_dataset, {}).get("leakage", {})
               .get("identity_overlap_with_train"))
    single = [n for n, r in results.items()
              if not (r["both_classes_frame"] and r["both_classes_video"])]
    mixed = [n for n, r in results.items() if r["mixed_label_videos"] > 0]
    return {
        "train_dataset": train_dataset,
        "eval_datasets": list(results),
        "video_reduce": video_reduce,
        "ffpp_identity_overlap": overlap,
        "single_class_datasets": single,
        "mixed_label_datasets": mixed,
        "note": ("Cross-datasets are separate corpora (no training overlap by "
                 "construction); FF++ in-domain test is the identity-disjoint test split."),
        "ok": (overlap in (None, 0)) and not single and not mixed,
    }


def _ordered(names):
    head = [n for n in AXIS_ORDER if n in names]
    tail = sorted(n for n in names if n not in AXIS_ORDER)   # df40_* etc.
    return head + tail


def format_table(out: dict) -> str:
    results = out["results"]
    rows = [("dataset", "fAUC", "fACC", "fEER", "vAUC", "vACC", "vEER", "Nf", "Nv", "%fk", "flag")]
    for name in _ordered(results):
        r = results[name]
        f, v = r["frame"], r["video"]
        flags = ""
        if not (r["both_classes_frame"] and r["both_classes_video"]):
            flags += "1CLASS "
        if r["mixed_label_videos"]:
            flags += "MIXED "
        lk = r.get("leakage", {})
        if lk.get("identity_overlap_with_train"):
            flags += "OVERLAP "
        rows.append((name, f"{f['auc']:.3f}", f"{f['acc']:.3f}", f"{f['eer']:.3f}",
                     f"{v['auc']:.3f}", f"{v['acc']:.3f}", f"{v['eer']:.3f}",
                     str(f["n"]), str(v["n"]), f"{f['frac_fake']*100:.0f}", flags.strip() or "ok"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for j, row in enumerate(rows):
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        if j == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(widths))))
    s = out["sanity"]
    lines.append("")
    lines.append(f"video_reduce = {s['video_reduce']}  (anchor metric = AUC-ROC; fAUC=frame, vAUC=video)")
    lines.append(f"sanity: FF++ identity overlap = {s['ffpp_identity_overlap']} | "
                 f"single-class = {s['single_class_datasets'] or 'none'} | "
                 f"mixed-label = {s['mixed_label_datasets'] or 'none'} | OK = {s['ok']}")
    return "\n".join(lines)


def save_results(out: dict, out_dir, stem: str = "eval") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(out, indent=2))

    csv_path = out_dir / f"{stem}.csv"
    cols = ["dataset", "is_indomain", "video_reduce",
            "frame_auc", "frame_acc", "frame_eer", "frame_n", "frame_frac_fake",
            "video_auc", "video_acc", "video_eer", "video_n", "video_frac_fake",
            "both_classes_frame", "both_classes_video", "mixed_label_videos",
            "identity_overlap_with_train"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name in _ordered(out["results"]):
            r = out["results"][name]
            f, v, lk = r["frame"], r["video"], r.get("leakage", {})
            w.writerow([name, r["is_indomain"], r["video_reduce"],
                        f["auc"], f["acc"], f["eer"], f["n"], f["frac_fake"],
                        v["auc"], v["acc"], v["eer"], v["n"], v["frac_fake"],
                        r["both_classes_frame"], r["both_classes_video"], r["mixed_label_videos"],
                        lk.get("identity_overlap_with_train")])
    return {"json": str(json_path), "csv": str(csv_path)}
