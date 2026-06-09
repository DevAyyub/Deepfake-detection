"""sfdet.preprocess — one-time face detection and cropping over raw videos.

  extract_faces.py   Detect + align + crop faces from the raw videos and write
                     image crops to crops_root. Applies ONLY to the three datasets
                     that ship as video: FF++, Celeb-DF v2, and DFDC. WildDeepfake
                     and DF40 are skipped here — both ship pre-extracted face crops
                     and are consumed in place by the data layer.
  manifest.py        Build/read the crop manifest (paths, labels, source-video
                     ids) the data layer consumes. For FF++/Celeb-DF/DFDC it reads
                     the crops written by extract_faces.py; for WildDeepfake and
                     DF40 it walks their provided crop folders directly. Same
                     manifest schema for all five so datasets.py treats them alike.

Runs in the ISOLATED preprocessing environment (requirements-preprocess.txt),
not the training env, because the MTCNN detector (facenet-pytorch) pins an old
torch/numpy. The crops it writes are plain image files, so there is no
torch-version coupling back to training.

Resampling caveat (notes 1.8): alignment/resize resampling injects interpolation
artifacts into the FFT spectrum the frequency branch later consumes. The
detector, target size, and interpolation must be FIXED and documented here and
applied identically across all datasets so the spectrum stays comparable —
including the pre-cropped sets (WildDeepfake, DF40), whose crops are resized to
the same target size at load (DF40/WildDeepfake are upscaled to base.image_size).

Implementations land in a later data/pipeline chat.
"""
