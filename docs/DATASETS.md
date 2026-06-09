# Datasets — acquisition, layout, and protocol

Reference for the five datasets in this project. **Train on FaceForensics++ (c23,
standard four-manipulation) only; everything else is evaluation-only.** Anchor
metric is AUC-ROC (accuracy + EER alongside), reported per dataset at frame and
video granularity. Filesystem roots come from `paths.yaml` (gitignored) — nothing
here is committed.

## Reference table

| Dataset | Role | Axis | Ships as | Source & access | Approx. size |
|---|---|---|---|---|---|
| **FaceForensics++** (c23) | **Train** (only) | training source | raw mp4 video → extract | TUM Google form → emailed download script (`download-FaceForensics.py`) | tens of GB (c23 video)* |
| **Celeb-DF v2** | Test | conventional lab | raw mp4 video → extract | `yuezunli/celeb-deepfakeforensics` Google form → emailed link | ~10 GB* |
| **DFDC** | Test | conventional lab | raw mp4 video → extract | Kaggle competition `deepfake-detection-challenge` (accept rules); **test set only** | test set a few–tens of GB; **full train ~470 GB** — don't pull |
| **WildDeepfake** | Test | real-world / internet-collected | **pre-cropped** face frames → no extraction | `OpenTAI/wild-deepfake`, agreement form; mirrored on Hugging Face | ~6 GB* |
| **DF40** (4 diffusion subsets) | Test | diffusion | **pre-cropped** face images → no extraction | DF40 Google form → Google Drive / Baidu; pull 4 folders only (see `scripts/download_df40.sh`) | 4 subsets ≈ a few GB; **full test set ~93 GB** |

\*Sizes marked `*` are approximate — confirm on download. Verified from source:
DFDC (~470 GB full train) and DF40 (~93 GB full test set, all 40 methods).

## Per-dataset quirks (the things that bite)

- **FaceForensics++** — the download also offers DeepFakeDetection, FaceShifter,
  and `actors/`; pull **only the four** (Deepfakes, Face2Face, FaceSwap,
  NeuralTextures) plus the originals. Use the identity-disjoint train/val/test
  split (no video leakage). Real class = `original_sequences`.
- **Celeb-DF v2** — evaluate on the **518-video `List_of_testing_videos.txt`**
  only, not the full set. Make sure it is **v2** (a newer *Celeb-DF++* now exists —
  not that). Real = Celeb-real + YouTube-real; fake = Celeb-synthesis.
- **DFDC** — test-only here, so skip the 470 GB train corpus and use the DFDC
  **test set**; labels come from a CSV, not folder names. Not to be confused with
  the **DFDCP** preview (the notes want full DFDC).
- **WildDeepfake** — ships as face-sequence folders (`<seq>/<frame>.png`), so
  **no MTCNN extraction**; the sequence folders give video-level grouping for
  free. Email replies are slow — the Hugging Face mirror is the practical route.
  This is the real-world / unknown-provenance axis; do not group it by generation
  family.
- **DF40** — the **DDPM folder is named `ddim/`** on disk (DF40's own paper table
  calls method #29 "DDPM"); there is **no** `ddpm/` folder. DF40 ships **fake
  images only** — the real class is a separate download. Each subset folder has
  `ff/` and `cdf/` sub-domains. See `scripts/download_df40.sh`.

## Unified local layout

Keep each dataset's **native layout** under one root. The extractor writes the
three video datasets into one canonical `crops/` layout; the two pre-cropped sets
are consumed in place; a per-dataset **manifest** is the actual unifying layer.
The directory names below map to `paths.yaml` keys (in brackets); the whole
`/data` tree is gitignored.

```
/data/                                    # dataset root (gitignored; lives on the GPU box)
│
├── FaceForensics++/c23/                  # RAW mp4 — native FF++ layout  → extract     [faceforensics_c23]
│   ├── original_sequences/youtube/c23/videos/
│   └── manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/
│
├── Celeb-DF-v2/                          # RAW mp4 — native Celeb-DF layout  → extract  [celebdf_v2]
│   ├── Celeb-real/   YouTube-real/   Celeb-synthesis/
│   └── List_of_testing_videos.txt        # 518-video official test split
│
├── DFDC/                                 # RAW mp4 — TEST set only  → extract           [dfdc]
│   ├── test/                             # ~5k test videos
│   └── labels.csv                        # filename → REAL/FAKE
│
├── WildDeepfake/                         # PRE-CROPPED face frames — consumed in place   [wilddeepfake]
│   ├── real_test/<seq_id>/<frame>.png
│   └── fake_test/<seq_id>/<frame>.png
│
├── DF40/                                 # PRE-CROPPED — diffusion subsets only          [df40_diffusion]
│   ├── sd2.1/   {ff,cdf}/                #   SD-2.1    (fake images only)
│   ├── ddim/    {ff,cdf}/                #   = DDPM    (fake images only)  ← folder literally `ddim`
│   ├── PixArt/  {ff,cdf}/                #   PixArt-α  (fake images only)
│   ├── DiT/     {ff,cdf}/                #   DiT-XL/2  (fake images only)
│   └── real/                             #   real class — downloaded SEPARATELY
│       ├── ff/                           #     FF++-real crop pack
│       └── cdf/                          #     Celeb-DF-real crop pack
│
└── crops/                                # extractor OUTPUT (video datasets only)        [crops_root]
    ├── FaceForensics++/c23/<label>/<video_id>/<frame>.png
    ├── Celeb-DF-v2/<label>/<video_id>/<frame>.png
    └── DFDC/<label>/<video_id>/<frame>.png
```

The unifier is the **manifest**, not the folder shape. `manifest.py` emits one
CSV/JSON per dataset with identical columns:

```
crop_path, label, dataset, subset, domain, source_video_id
```

built from `crops/` for FF++/Celeb-DF/DFDC, and by walking the provided folders
for WildDeepfake and DF40 (`subset` ∈ {sd2.1, ddim→ddpm, PixArt, DiT}; `domain` ∈
{ff, cdf}). `datasets.py` then treats all five identically; per-subset DF40
reporting and video-level grouping both fall out of the `subset` /
`source_video_id` columns. Only FF++, Celeb-DF v2, DFDC go through the isolated
MTCNN env; WildDeepfake and DF40 skip it.

## Crop size / resampling (a frequency-branch concern)

Everything runs at **256×256** end-to-end. 256 was chosen deliberately to **match
DF40 / DeepfakeBench's native 256² crops** (`base.image_size: 256`), so DF40's
pre-cropped images need no resampling at all. The extractor writes FF++/Celeb-DF/
DFDC at 256² (`crop_size=256`), and WildDeepfake (whatever its native size) is
resized to 256 at load with one fixed interpolation. Per the §1.8 caveat,
resize/alignment resampling injects interpolation artifacts into the FFT magnitude
spectrum, so keeping a single resize/interpolation history across datasets matters;
fixing one target size (256) for everyone is what avoids a per-dataset resize
confound. EfficientNet-B4's nominal 380 input is **not** used — B4 runs fine at 256
via adaptive pooling, and cross-dataset resampling consistency wins over B4's native
resolution. (Alternative, if ever revisited: a different global `base.image_size`,
not a per-dataset override.)

For the DF40 test specifically: use **DF40's own `real/` crops** as the real
class, not your MTCNN-extracted FF++/Celeb-DF crops, so real-vs-fake inside the
DF40 test shares one extraction/resampling history.

## Getting the data

- FF++, Celeb-DF v2, DFDC, WildDeepfake: via the access columns in the table
  above (forms / Kaggle / Hugging Face).
- DF40 diffusion subsets: `scripts/download_df40.sh` (fill in your post-approval
  Drive links first).

## Framing reminders (carry into prose and the paper)

- Three separate test axes: conventional lab (Celeb-DF v2, DFDC), real-world /
  internet-collected (WildDeepfake), diffusion (DF40). Do not collapse them or
  call the lab/real-world sets "GAN-based."
- DF40 subset differences = a mixture of shared and distinct spectral signatures
  whose balance tracks how much convolutional upsampling sits in the generation
  path — **not** a latent-vs-pixel split (pixel-space DDPM out-scores latent DiT
  in the reference numbers) and not "qualitatively different mechanisms."
- DDPM (DF40 method #29, folder `ddim/`) is pixel-space, not the DDIM sampler.
- DF40's own method table (its Tab. 2) **mislabels** the diffusion subsets — it lists
  DDPM as "latent diffusion" and SD-2.1 as "GAN-based." Both are wrong (DDPM is
  pixel-space; SD-2.1 is latent diffusion). The framing above is the accurate one;
  do **not** cite DF40's sub-type column, as a reviewer who checks it will find it
  inconsistent.
- The DF40 paper's Xception (and SRM/SPSL/RECCE/RFM) baselines are trained on DF40's
  **own** FF-domain fakes, DeepfakeBench-aligned — **not** the classic FF++ c23
  four-manipulation set we train on. Same test data, different training source. So a
  DF40-paper "Xception" AUC is **not** a like-for-like baseline for this model; our
  matched comparison is the spatial-only vs full-model E1 gap (and E3 for SBI on the
  subsets). Treat any external DF40 reference numbers as provenance-pending until
  sourced to an exact table/protocol.
