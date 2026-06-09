# SETUP — from a fresh clone to the first E1 (spatial-only) results

Ordered, practical checklist for **this** project. Train on FaceForensics++ c23
(four-manipulation) **only**; evaluate on Celeb-DF v2, DFDC, WildDeepfake, and the
DF40 diffusion subsets. `docs/DATASETS.md` has per-dataset layout/quirks — this
file is the run order. Everything runs at **256×256** end-to-end.

---

## Where things run (recommended topology)

**Colab Pro + Google Drive**, split by what each step actually needs:

- **Face extraction is CPU-only** (dlib HOG detect + cv2 align — no GPU). Run it
  wherever the raw videos sit, or in a Colab **CPU** runtime. Don't spend GPU
  credits on it. It is slow (FF++ ~1k videos + Celeb-DF ~6k videos × 32 frames) —
  do it once, overnight.
- **Training/eval need the GPU → Colab Pro.** One rule that matters most:
  **never train off crops sitting in mounted Drive.** Millions of tiny PNG reads
  starve the GPU and burn Pro hours. Keep crops as a single tarball on Drive; at
  the start of each GPU session copy it to `/content` (local SSD) and untar there;
  train against `/content`. Checkpoint `best.pt` back to Drive; `--resume` (same
  `--run-name`) if a session drops.
- **Local VS Code** = editing + cheap verification (CPU torch): run `pytest`,
  `verify_env.py`, `verify_dataloaders.py` locally to catch problems before Colab.

> Upgrade path (optional): if the 30-epoch run keeps hitting Colab session limits,
> or you run many ablations, a cheap on-demand cloud GPU (RunPod / Vast / Lambda —
> persistent NVMe, no session cap) removes the Drive-I/O + disconnect pain for a
> few $/hr. Not needed to start.

> **Absolute-path gotcha:** the extraction log and manifests store **absolute**
> `crop_path`s and are gitignored (machine-local, derived). So keep crops at a
> **stable path** and rebuild the manifest on the machine you train on. The clean
> Colab pattern: always extract/stage crops to `/content/crops`, build manifests
> there, persist `/content/crops` + the manifests to Drive as a tarball, and
> always re-stage to the **same** `/content/crops` path next session.

---

## 0. Local dev env (no data needed)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e . && pip install -r requirements.txt
#   No local GPU? install CPU torch instead of the default CUDA build:
#   pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
python verify_env.py        # imports + EfficientNet-B4 forward pass at 256
pytest -q                   # with torch present, the numeric FFT/shape tests RUN (not skip)
#   Per-module smoke checks — first real-torch run of each component (prints shapes + the PRE-FUSION order):
python -m sfdet.models.frequency_branch   # polar grid_sample + pre-fusion tap  -> freq_feat [B,256,16,16]
python -m sfdet.models.fusion             # single-stage cross-attention        -> pooled    [B,512]
python -m sfdet.models.model              # assembled forward + explain(); re-checks the tap fires PRE-fusion
python -m sfdet.explain.explainability    # Grad-CAM backward to each pre-fusion target
```

For preprocessing only, a separate light env with dlib:
```bash
pip install -r requirements-preprocess.txt   # dlib-bin
```

## 1. Get the three things the dataset downloads do NOT include

### 1a. FaceForensics++ official split JSONs  (HARD prereq — loaders error without)
Public (not behind the FF++ EULA). Place at `<ff_root>/splits/` (or set
`ffpp_splits:` in `paths.yaml`).
```bash
mkdir -p <ff_root>/splits
for f in train val test; do
  wget -O <ff_root>/splits/$f.json \
    https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/$f.json
done
```
Format = lists of identity pairs (`[["071","054"], ...]`); the loader flattens them
into the per-split id set and assigns each crop by its source identity. 720/140/140
identity-disjoint videos for train/val/test.

### 1b. dlib 81-landmark predictor  (extraction only)
~19 MB. This is the **81**-point model (the extractor needs 81, not dlib's standard
68-point `shape_predictor_68_face_landmarks.dat`).
```bash
mkdir -p dlib_tools
wget -O dlib_tools/shape_predictor_81_face_landmarks.dat \
  https://raw.githubusercontent.com/codeniko/shape_predictor_81_face_landmarks/master/shape_predictor_81_face_landmarks.dat
# pass --predictor-path dlib_tools/shape_predictor_81_face_landmarks.dat to extract_faces.py
```

### 1c. DF40 real-image packs  (DF40 ships FAKES only)
Without the reals, every DF40 subset is single-class → AUC reported as `n/a`
(the harness flags it). Requires DF40 access (Google form on the DF40 repo).
```bash
pip install gdown
# Open scripts/download_df40.sh, paste your 4 post-approval Drive folder links
# (the two REAL-pack file IDs are already filled in), then:
DF40_DEST=<DF40 root> bash scripts/download_df40.sh
# Reals land in <DF40 root>/real/{ff,cdf}. ⚠ gdown caps folders at ~50 files —
# if a fake subset looks short, re-fetch it with the rclone block in that script.
```
Use DF40's **own** real packs (not your extracted FF++/Celeb-DF reals) so real-vs-
fake inside the DF40 test shares one resampling history (see `docs/DATASETS.md`). If
you must fall back to your own reals, document that resampling confound.

## 2. paths.yaml

```bash
cp paths.example.yaml paths.yaml      # then edit it
```
Fill the roots you have. Leave `dfdc:` as-is (its manifest just won't exist — fine).
Add `ffpp_splits:` if the split JSONs aren't at `<faceforensics_c23>/splits/`.

## 3. Extract faces — FF++ + Celeb-DF only  (videos → 256² aligned crops; CPU)

```bash
# smoke first (seconds) to confirm it runs, then drop --limit for the full pass:
python -m sfdet.preprocess.extract_faces --dataset faceforensics_c23 \
  --data-root <FF++ root> --crops-root <crops_root> \
  --predictor-path dlib_tools/shape_predictor_81_face_landmarks.dat --comp c23 \
  --manipulations Deepfakes Face2Face FaceSwap NeuralTextures --limit 5

python -m sfdet.preprocess.extract_faces --dataset celebdf_v2 \
  --data-root <CelebDF root> --crops-root <crops_root> \
  --predictor-path dlib_tools/shape_predictor_81_face_landmarks.dat \
  --celebdf-test-list <CelebDF root>/List_of_testing_videos.txt --limit 5
```
DFDC: skip (no data). WildDeepfake + DF40: pre-cropped — **no** extraction.

## 4. Build manifests  (everything you have)

```bash
python -m sfdet.preprocess.manifest --dataset faceforensics_c23 --crops-root <crops_root>
python -m sfdet.preprocess.manifest --dataset celebdf_v2        --crops-root <crops_root>
python -m sfdet.preprocess.manifest --dataset wilddeepfake   --data-root <WildDeepfake root>
python -m sfdet.preprocess.manifest --dataset df40_diffusion --data-root <DF40 root>
# DFDC manifest: skip.
```

## 5. Stage crops + train on Colab GPU  (the Drive pattern)

```bash
# once, after extraction (on the extraction machine):
tar -C <crops_root> -czf crops.tar.gz .
# upload crops.tar.gz + the manifests + WildDeepfake/DF40 crops to Drive.

# in a Colab GPU session:
#   from google.colab import drive; drive.mount('/content/drive')
#   !mkdir -p /content/crops && tar -C /content/crops -xzf /content/drive/MyDrive/<...>/crops.tar.gz
#   point paths.yaml: crops_root -> /content/crops  (rebuild manifests here if paths differ)
```

## 6. Verify → smoke → E1

```bash
python scripts/verify_dataloaders.py          # exit 0 = batch contract + coverage OK (the gate)

python scripts/train.py --model-config configs/model/spatial_only.yaml \
  --max-train-batches 5 --max-val-batches 5 --run-name smoke      # wiring check
python scripts/train.py --model-config configs/model/spatial_only.yaml --run-name e1_spatial

python scripts/evaluate.py --checkpoint experiments/results/e1_spatial/best.pt \
  --model-config configs/model/spatial_only.yaml                  # -> eval.json/csv/md (first §2.9)
```
First train run downloads B4 ImageNet weights; the normalization guard validates
against the real `pretrained_cfg` then.

## 7. Git + Colab clone

```bash
git init && git add -A && git commit -m "Phase A: spatial-only pipeline"
git remote add origin <private-repo-url> && git push -u origin main
# Colab: !git clone <repo> (PAT for private) → %pip install -e . → mount Drive so paths.yaml resolves
```

---

## Quick reference — what's evaluation-only

| Dataset | Train? | Ships as | Extraction? | Manifest source |
|---|---|---|---|---|
| FaceForensics++ c23 | **yes (only)** | video | yes (256²) | extraction log |
| Celeb-DF v2 | no (test) | video | yes (256²) | extraction log |
| DFDC (full) | no (test) | video | yes (256²) | extraction log — *deferred (no data yet)* |
| WildDeepfake | no (test) | pre-cropped | no | walk folders |
| DF40 (4 diffusion subsets) | no (test) | pre-cropped | no | walk folders (+ separate reals) |
