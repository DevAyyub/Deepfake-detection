"""sfdet.engine.trainer — FF++ c23 training loop for the spatial-only model.

Minimal but correct: enough to fit EfficientNet-B4 to a real, trustworthy
checkpoint (this is E1's spatial-only arm), not a smoke test. Includes
BCE-with-logits, AMP, AdamW/Adam + linear-warmup→cosine schedule, gradient
clipping, checkpoint selection on validation AUC-ROC, and per-epoch train/val
logging (frame AND video AUC/acc/EER).

It is model-agnostic: it trains whatever `nn.Module` it is handed whose
`forward(x)` returns one logit per sample for `batch["spatial"]`. The dual model
trains through this same loop later by swapping the model/config.
"""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from sfdet.engine.losses import build_loss
from sfdet.metrics.classification import binary_metrics


# --------------------------------------------------------------------------- #
# setup helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster fine-tuning; train-time nondeterminism is fine


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    o = (cfg.get("train", {}) or {}).get("optimizer", {}) or {}
    name = str(o.get("name", "adam")).lower()
    lr = float(o.get("lr", 2e-4))
    wd = float(o.get("weight_decay", 1e-5))
    betas = tuple(o.get("betas", (0.9, 0.999)))
    # Discriminative LR. The spatial backbone is ImageNet-pretrained while the frequency
    # branch + fusion + head are from scratch; backbone_lr_mult (<1) fine-tunes the
    # pretrained conv weights gentler so they aren't washed out while the new modules
    # learn at full lr. Default 1.0 == single LR, so the spatial-only arm is unchanged;
    # the dual configs set e.g. 0.1. Backbone params are identified by name ("backbone"
    # appears only in the timm trunk: spatial_branch.backbone.* / branch.backbone.*).
    mult = float(o.get("backbone_lr_mult", 1.0))

    def _split(named):
        # No weight decay on biases / norm (1-D) params — standard for fine-tuning.
        decay, no_decay = [], []
        for n, p in named:
            (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append(p)
        return decay, no_decay

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if mult == 1.0:
        decay, no_decay = _split(named)
        groups = [{"params": decay, "lr": lr, "weight_decay": wd},
                  {"params": no_decay, "lr": lr, "weight_decay": 0.0}]
    else:
        rest = [(n, p) for n, p in named if "backbone" not in n]
        backbone = [(n, p) for n, p in named if "backbone" in n]
        rd, rnd = _split(rest)
        bd, bnd = _split(backbone)
        # rest groups FIRST so param_groups[0] (used by the scheduler base_lr and the
        # epoch lr log) reflects the MAIN lr, not the reduced backbone lr.
        groups = [{"params": rd,  "lr": lr,        "weight_decay": wd},
                  {"params": rnd, "lr": lr,        "weight_decay": 0.0},
                  {"params": bd,  "lr": lr * mult, "weight_decay": wd},
                  {"params": bnd, "lr": lr * mult, "weight_decay": 0.0}]
        groups = [g for g in groups if len(g["params"]) > 0]   # drop empties

    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(o.get("momentum", 0.9)), nesterov=True)
    raise ValueError(f"unknown optimizer '{name}'")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int, epochs: int):
    s = (cfg.get("train", {}) or {}).get("scheduler", {}) or {}
    name = str(s.get("name", "cosine")).lower()
    total = max(steps_per_epoch * epochs, 1)
    if name in ("none", "constant", ""):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name != "cosine":
        raise ValueError(f"unknown scheduler '{name}' (only 'cosine'/'none' wired)")

    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = float(s.get("min_lr", base_lr * 0.01))
    warmup = int(round(float(s.get("warmup_epochs", 1)) * steps_per_epoch))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total - warmup, 1), eta_min=min_lr)
    if warmup <= 0:
        return cosine
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cosine], milestones=[warmup])


# --------------------------------------------------------------------------- #
# train / eval steps
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, *,
                    amp_enabled: bool, grad_clip=None, log_interval: int = 50,
                    max_batches=None, epoch: int = 0, forward_fn=None) -> float:
    # forward_fn(model, batch) -> logits [B]; defaults to spatial-only so existing
    # single-input callers are unaffected. The dual model passes a (spatial, frequency)
    # adapter (see sfdet.models.factory.build_model).
    forward_fn = forward_fn or (lambda m, b: m(b["spatial"]))
    model.train()
    dev_type = device.type
    running, seen = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        y = batch["label"].float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_enabled):
            logits = forward_fn(model, batch)     # [B] — adapter pulls this variant's inputs
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()                      # per-step (warmup + cosine over total steps)

        bs = y.size(0)
        running += loss.item() * bs
        seen += bs
        if log_interval and (i % log_interval == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch} step {i:>5}/{len(loader)}  loss {loss.item():.4f}  lr {lr:.2e}")
    return running / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, amp_enabled: bool, max_batches=None,
             forward_fn=None) -> dict:
    """Validation pass: BCE loss + frame-level AND video-level AUC/acc/EER.
    Video score = mean of a source video's per-frame probabilities."""
    forward_fn = forward_fn or (lambda m, b: m(b["spatial"]))
    model.eval()
    dev_type = device.type
    logit_chunks, label_chunks, vids = [], [], []
    loss_sum, seen = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        y = batch["label"].float()
        with autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_enabled):
            logits = forward_fn(model, batch)
            loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        seen += y.size(0)
        logit_chunks.append(logits.float().cpu())
        label_chunks.append(y.cpu())
        vids.extend(batch["source_video_id"])

    logits = torch.cat(logit_chunks).numpy()
    labels = torch.cat(label_chunks).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))                 # sigmoid

    frame = binary_metrics(labels, probs)

    # video-level: average frame probs per source video (video_reduce: mean)
    agg: dict = {}
    for v, p, l in zip(vids, probs, labels):
        d = agg.setdefault(v, {"p": [], "l": l})
        d["p"].append(p)
    v_scores = np.array([np.mean(d["p"]) for d in agg.values()])
    v_labels = np.array([d["l"] for d in agg.values()])
    video = binary_metrics(v_labels, v_scores)

    return {"loss": loss_sum / max(seen, 1), "frame": frame, "video": video,
            "n_frames": int(seen), "n_videos": int(len(agg))}


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, cfg) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best": best, "cfg": cfg}, path)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def fit(model, train_loader, val_loader, cfg: dict, device, *, out_dir=None,
        run_name=None, resume=None, max_train_batches=None, max_val_batches=None,
        forward_fn=None) -> dict:
    import yaml

    train_cfg = cfg.get("train", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}
    epochs = int(train_cfg.get("epochs", 30))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    grad_clip = train_cfg.get("grad_clip_norm", None)
    patience = train_cfg.get("early_stop_patience", None)

    out_dir = Path(out_dir or log_cfg.get("out_dir", "experiments/results"))
    run_name = run_name or f"spatial_only_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    criterion = build_loss(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    scaler = GradScaler("cuda", enabled=amp_enabled)

    start_epoch, best, since_best = 0, -float("inf"), 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best", best)
        print(f"resumed from {resume} at epoch {start_epoch} (best val AUC {best:.4f})")

    metrics_path = run_dir / "metrics.csv"
    if not resume or not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "train_loss", "val_loss",
                 "val_auc_frame", "val_acc_frame", "val_eer_frame",
                 "val_auc_video", "val_acc_video", "val_eer_video"])

    print(f"run: {run_dir}  | device={device} amp={amp_enabled} epochs={epochs} "
          f"steps/epoch={steps_per_epoch}")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            amp_enabled=amp_enabled, grad_clip=grad_clip,
            max_batches=max_train_batches, epoch=epoch, forward_fn=forward_fn)
        val = evaluate(model, val_loader, criterion, device,
                       amp_enabled=amp_enabled, max_batches=max_val_batches,
                       forward_fn=forward_fn)
        lr = optimizer.param_groups[0]["lr"]
        sel = val["frame"]["auc"]              # checkpoint-selection metric (save_best_on: auc_roc)

        print(f"[epoch {epoch:>3}/{epochs}] {time.time()-t0:5.0f}s  "
              f"train_loss {train_loss:.4f}  val_loss {val['loss']:.4f}  | "
              f"frame AUC {val['frame']['auc']:.4f} acc {val['frame']['acc']:.4f} "
              f"EER {val['frame']['eer']:.4f}  | "
              f"video AUC {val['video']['auc']:.4f} acc {val['video']['acc']:.4f} "
              f"EER {val['video']['eer']:.4f}  ({val['n_videos']} vids)")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{lr:.3e}", f"{train_loss:.6f}", f"{val['loss']:.6f}",
                 f"{val['frame']['auc']:.6f}", f"{val['frame']['acc']:.6f}", f"{val['frame']['eer']:.6f}",
                 f"{val['video']['auc']:.6f}", f"{val['video']['acc']:.6f}", f"{val['video']['eer']:.6f}"])

        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best, cfg)
        improved = np.isfinite(sel) and sel > best
        if improved:
            best, since_best = sel, 0
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best, cfg)
            print(f"  ** new best val frame AUC {best:.4f} -> best.pt")
        else:
            since_best += 1
            if patience and since_best >= int(patience):
                print(f"early stop: no val-AUC improvement in {patience} epochs")
                break

    print(f"done. best val frame AUC {best:.4f}  | {run_dir/'best.pt'}")
    return {"best_auc": float(best), "best_path": str(run_dir / "best.pt"), "run_dir": str(run_dir)}
