#!/usr/bin/env bash
# =============================================================================
# scripts/download_df40.sh
# Download ONLY the four DF40 diffusion subsets (+ the two shared real-image
# packs) for this project's test-only diffusion axis.
#
# Pulls (FAKE images only):
#   SD-2.1, DDPM (folder named `ddim`), PixArt-alpha, DiT-XL/2
# plus the shared real packs (FF++-real, Celeb-DF-real) that serve as the REAL
# class for ALL four subsets and BOTH ff/cdf domains.
#
# Does NOT download: the other 36 DF40 methods, the ~50 GB DF40 *training* data,
# or any original videos. DF40 is TEST-ONLY in this project.
#
# ---------------------------------------------------------------------------
# PREREQUISITE: request DF40 access (Google form on the YZY-stack/DF40 repo).
# After approval, open the Drive "testing data" folder, go into EACH of the four
# subset subfolders, Share -> copy link, and paste into FAKE_*_URL below. Until
# you do, this script refuses to run (loud, not silent).
#
# ⚠ gdown FOLDER LIMIT: `gdown --folder` downloads at most ~50 files per folder.
#   The DF40 subset folders contain thousands of crops, so gdown will TRUNCATE
#   them. Use the rclone alternative at the bottom for the fake subsets (it has
#   no such cap). gdown is fine for the two real packs (single-file archives).
#
# Tooling: `pip install gdown` (and optionally rclone). Lightweight; install in
#   any env — these are plain image files with no torch coupling.
# =============================================================================
set -euo pipefail

# --- Destination (match `df40_diffusion` in paths.yaml) ----------------------
DEST_ROOT="${DF40_DEST:-/data/DF40}"

# --- FILL IN: Drive links for the four diffusion SUBSET FOLDERS ---------------
FAKE_SD21_URL="PASTE_DRIVE_FOLDER_LINK_FOR_sd2.1"
FAKE_DDPM_URL="PASTE_DRIVE_FOLDER_LINK_FOR_ddim"     # DF40's "DDPM" == folder `ddim`
FAKE_PIXART_URL="PASTE_DRIVE_FOLDER_LINK_FOR_PixArt"
FAKE_DIT_URL="PASTE_DRIVE_FOLDER_LINK_FOR_DiT"

# --- Real packs: file IDs published in the DF40 README (verify; may change) --
REAL_FFPP_ID="1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb"     # FF++-real crops (archive)
REAL_CDF_ID="1FGZ3aYsF-Yru50rPLoT5ef8-2Nkt4uBw"      # Celeb-DF-real crops (archive)

# -----------------------------------------------------------------------------
command -v gdown >/dev/null 2>&1 || { echo "ERROR: gdown not found. Run: pip install gdown"; exit 1; }

# Refuse to run while the fake-subset links are still placeholders.
for v in FAKE_SD21_URL FAKE_DDPM_URL FAKE_PIXART_URL FAKE_DIT_URL; do
  case "${!v}" in
    PASTE_*) echo "ERROR: $v is still a placeholder. Fill in your DF40 Drive links first (see header)."; exit 1;;
  esac
done

echo ">> DF40 diffusion subsets -> $DEST_ROOT"
mkdir -p "$DEST_ROOT"/{sd2.1,ddim,PixArt,DiT,real/_archives}

dl_fake_folder () {  # <url> <dest_dir> <label>
  echo ">> [$3] gdown --folder -> $2   (⚠ caps at ~50 files/folder; see rclone block if truncated)"
  gdown --folder --remaining-ok -O "$2" "$1"
}

dl_fake_folder "$FAKE_SD21_URL"   "$DEST_ROOT/sd2.1"  "SD-2.1"
dl_fake_folder "$FAKE_DDPM_URL"   "$DEST_ROOT/ddim"   "DDPM (ddim)"
dl_fake_folder "$FAKE_PIXART_URL" "$DEST_ROOT/PixArt" "PixArt-alpha"
dl_fake_folder "$FAKE_DIT_URL"    "$DEST_ROOT/DiT"    "DiT-XL/2"

echo ">> real packs (single-file archives) -> $DEST_ROOT/real/_archives"
gdown "$REAL_FFPP_ID" -O "$DEST_ROOT/real/_archives/" || echo "WARN: FF++-real fetch failed; check access / ID."
gdown "$REAL_CDF_ID"  -O "$DEST_ROOT/real/_archives/" || echo "WARN: Celeb-DF-real fetch failed; check access / ID."

cat <<EOF

>> Download step complete. NEXT STEPS:
   1) Extract the archives in $DEST_ROOT/real/_archives/ into:
        $DEST_ROOT/real/ff/    (FF++-real crops)
        $DEST_ROOT/real/cdf/   (Celeb-DF-real crops)
      (.zip -> unzip ; .tar/.tar.gz -> tar xf. Inspect first — verify the
       internal layout before assuming it unpacks straight into ff/ and cdf/.)
   2) Sanity-check fake image counts under sd2.1/ ddim/ PixArt/ DiT/ (ff & cdf).
      If any look short (~50), gdown truncated the folder — re-fetch with rclone.
   3) Point df40_diffusion in paths.yaml at $DEST_ROOT.

   rclone alternative for the fake subsets (no 50-file cap):
     # one-time: `rclone config`  -> add a Google Drive remote called e.g. gdrive
     # then, per subset (path is relative to that remote / shared folder):
     rclone copy "gdrive:DF40/testing/sd2.1"  "$DEST_ROOT/sd2.1"  -P
     rclone copy "gdrive:DF40/testing/ddim"   "$DEST_ROOT/ddim"   -P
     rclone copy "gdrive:DF40/testing/PixArt" "$DEST_ROOT/PixArt" -P
     rclone copy "gdrive:DF40/testing/DiT"    "$DEST_ROOT/DiT"    -P
EOF
