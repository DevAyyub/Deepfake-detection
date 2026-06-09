"""Tests for sfdet.preprocess.manifest — stdlib-only, so they run in any
environment (no torch needed)."""
import json
from pathlib import Path

from sfdet.preprocess import manifest as M


def test_fields_schema_is_stable():
    assert M.FIELDS == ["crop_path", "label", "dataset", "subset", "domain",
                        "source_video_id", "frame", "mask_path", "landmark_path"]


def test_write_read_roundtrip(tmp_path):
    rows = [
        M._row("/crops/a.png", 0, "faceforensics_c23", subset="real",
               source_video_id="original__033", frame="0"),
        M._row("/crops/b.png", 1, "faceforensics_c23", subset="Deepfakes",
               source_video_id="Deepfakes__033_097", frame="0",
               mask_path="/crops/b_mask.png", landmark_path="/crops/b.npy"),
    ]
    out = M.write_manifest(rows, tmp_path / "m.csv")
    assert Path(out).is_file()

    back = M.read_manifest(out)
    assert len(back) == 2
    assert set(back[0].keys()) == set(M.FIELDS)        # exact header
    assert back[0]["crop_path"] == "/crops/a.png"
    assert back[0]["label"] == "0" and back[1]["label"] == "1"   # CSV stringifies
    assert back[1]["mask_path"] == "/crops/b_mask.png"


def test_from_extraction_log_flattens_and_drops_zero_face(tmp_path):
    d = tmp_path / "faceforensics_c23"
    d.mkdir()
    lines = [
        {"dataset": "faceforensics_c23", "source_video_id": "Deepfakes__033_097",
         "label": 1, "subset": "Deepfakes", "domain": "", "excluded_zero_faces": False,
         "crops": [
             {"frame": 0, "crop_path": "/c/0.png", "mask_path": "/c/0_mask.png", "landmark_path": "/c/0.npy"},
             {"frame": 7, "crop_path": "/c/7.png", "mask_path": "", "landmark_path": ""},
         ]},
        {"dataset": "faceforensics_c23", "source_video_id": "original__050",
         "label": 0, "subset": "real", "domain": "", "excluded_zero_faces": True, "crops": []},
        {"dataset": "faceforensics_c23", "source_video_id": "original__051",
         "label": 0, "subset": "real", "domain": "", "crops": []},   # empty crops -> skipped too
    ]
    (d / "_extraction_log.jsonl").write_text("\n".join(json.dumps(x) for x in lines))

    rows = M.from_extraction_log(tmp_path, "faceforensics_c23")
    assert len(rows) == 2                                  # only the 2 crops of the valid video
    assert {r["frame"] for r in rows} == {0, 7}
    assert all(r["label"] == 1 for r in rows)
    assert all(r["source_video_id"] == "Deepfakes__033_097" for r in rows)
    by_frame = {r["frame"]: r for r in rows}
    assert by_frame[0]["crop_path"] == "/c/0.png"
    assert by_frame[0]["mask_path"] == "/c/0_mask.png"


def test_from_wilddeepfake_labels_and_grouping(tmp_path):
    (tmp_path / "real_test" / "seqA").mkdir(parents=True)
    (tmp_path / "real_test" / "seqA" / "0.png").touch()
    (tmp_path / "fake_test" / "seqB").mkdir(parents=True)
    (tmp_path / "fake_test" / "seqB" / "3.png").touch()

    rows = M.from_wilddeepfake(tmp_path)
    assert len(rows) == 2
    by_label = {r["label"]: r for r in rows}
    assert by_label[0]["source_video_id"] == "seqA" and by_label[0]["frame"] == "0"
    assert by_label[0]["dataset"] == "wilddeepfake"
    assert by_label[1]["source_video_id"] == "seqB"


def test_from_df40_folder_to_name_and_domain(tmp_path):
    # sd2.1 folder -> reporting name stable_diffusion_2_1; reals under real/<domain>
    (tmp_path / "sd2.1" / "ff" / "vid1").mkdir(parents=True)
    (tmp_path / "sd2.1" / "ff" / "vid1" / "0.png").touch()
    (tmp_path / "real" / "ff" / "vidR").mkdir(parents=True)
    (tmp_path / "real" / "ff" / "vidR" / "0.png").touch()

    rows = M.from_df40(tmp_path)
    assert len(rows) == 2
    fake = next(r for r in rows if r["label"] == 1)
    real = next(r for r in rows if r["label"] == 0)
    assert fake["subset"] == "stable_diffusion_2_1" and fake["domain"] == "ff"
    assert real["subset"] == "real" and real["domain"] == "ff"
    assert fake["dataset"] == "df40_diffusion"
