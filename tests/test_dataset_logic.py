"""Tests for the torch-free logic in sfdet.data.dataset: FF++ splits, class
balancing, DF40 grouping, default paths, and the get_dataloaders wiring. These
run everywhere — under the conftest torch stub (no torch installed) or with real
torch — because none of them touch tensor math."""
import json

import pytest

from sfdet.data import dataset as D
from sfdet.preprocess.manifest import _row, write_manifest


# --------------------------------------------------------------------------- #
# FaceForensics++ split
# --------------------------------------------------------------------------- #
def test_ffpp_origin_id():
    assert D._ffpp_origin_id("original__033") == "033"
    assert D._ffpp_origin_id("Deepfakes__033_097") == "033"        # target id
    assert D._ffpp_origin_id("Face2Face__097_033") == "097"


def _write_splits(d):
    (d / "train.json").write_text(json.dumps([["033", "097"], ["010", "011"]]))
    (d / "val.json").write_text(json.dumps([["140", "200"]]))
    (d / "test.json").write_text(json.dumps([["300", "301"]]))


def test_apply_ffpp_split_identity_disjoint(tmp_path):
    _write_splits(tmp_path)
    rows = [
        {"dataset": "faceforensics_c23", "source_video_id": "original__033", "label": "0", "subset": "real"},
        {"dataset": "faceforensics_c23", "source_video_id": "Deepfakes__033_097", "label": "1", "subset": "Deepfakes"},
        {"dataset": "faceforensics_c23", "source_video_id": "Face2Face__010_011", "label": "1", "subset": "Face2Face"},
        {"dataset": "faceforensics_c23", "source_video_id": "original__140", "label": "0", "subset": "real"},
        {"dataset": "celebdf_v2", "source_video_id": "id0_0000", "label": "1", "subset": ""},  # non-ffpp -> dropped
    ]
    train = D.apply_ffpp_split(rows, tmp_path, "train")
    val = D.apply_ffpp_split(rows, tmp_path, "val")
    assert len(train) == 3                                   # 033 real + both 033/010-origin fakes
    assert all(r["dataset"] == "faceforensics_c23" for r in train)
    assert {r["source_video_id"] for r in val} == {"original__140"}
    # leakage check: no row object lands in two splits
    assert set(id(r) for r in train).isdisjoint(id(r) for r in val)


def test_load_ffpp_split_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.load_ffpp_split_ids(tmp_path, "train")


# --------------------------------------------------------------------------- #
# Class imbalance
# --------------------------------------------------------------------------- #
def _balance_rows():
    return ([{"label": "0", "subset": "real"}] * 2 +
            [{"label": "1", "subset": "Deepfakes"}] * 4 +
            [{"label": "1", "subset": "Face2Face"}] * 2 +
            [{"label": "1", "subset": "FaceSwap"}] * 2 +
            [{"label": "1", "subset": "NeuralTextures"}] * 2)


def test_balance_method_5050_and_even_methods():
    rows = _balance_rows()
    w = D.balance_weights(rows, "method")
    real = sum(w[i] for i, r in enumerate(rows) if r["label"] == "0")
    fake = sum(w[i] for i, r in enumerate(rows) if r["label"] == "1")
    assert abs(sum(w) - 1.0) < 1e-9
    assert abs(real - 0.5) < 1e-9 and abs(fake - 0.5) < 1e-9
    for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        mass = sum(w[i] for i, r in enumerate(rows) if r["subset"] == m)
        assert abs(mass - 0.125) < 1e-9                      # each method equal despite 4 vs 2 counts
    dw = [w[i] for i, r in enumerate(rows) if r["subset"] == "Deepfakes"]
    assert max(dw) - min(dw) < 1e-12                         # within a method, equal per-sample


def test_balance_binary_none_and_bad():
    rows = _balance_rows()
    wb = D.balance_weights(rows, "binary")
    assert abs(sum(wb[i] for i, r in enumerate(rows) if r["label"] == "0") - 0.5) < 1e-9
    assert abs(sum(wb[i] for i, r in enumerate(rows) if r["label"] == "1") - 0.5) < 1e-9
    assert D.balance_weights(rows, None) is None
    assert D.balance_weights([], "method") is None
    with pytest.raises(ValueError):
        D.balance_weights(rows, "bogus")


def test_make_balanced_sampler():
    rows = _balance_rows()
    sampler = D.make_balanced_sampler(rows, "method")
    assert len(sampler) == len(rows)
    assert D.make_balanced_sampler(rows, None) is None


# --------------------------------------------------------------------------- #
# DF40 grouping
# --------------------------------------------------------------------------- #
def test_group_df40_pairs_domain_reals():
    rows = ([{"subset": "real", "domain": "ff", "label": "0"}] * 2 +
            [{"subset": "real", "domain": "cdf", "label": "0"}] * 1 +
            [{"subset": "stable_diffusion_2_1", "domain": "ff", "label": "1"}] * 2 +
            [{"subset": "ddpm", "domain": "ff", "label": "1"}] * 1 +
            [{"subset": "stable_diffusion_2_1", "domain": "cdf", "label": "1"}] * 1)
    g = D.group_df40(rows)
    assert len(g[("stable_diffusion_2_1", "ff")]) == 4      # 2 fake + 2 ff reals
    assert len(g[("ddpm", "ff")]) == 3                      # 1 fake + 2 ff reals
    assert len(g[("stable_diffusion_2_1", "cdf")]) == 2     # 1 fake + 1 cdf real
    assert all(k[0] != "real" for k in g)                   # 'real' is never its own group


# --------------------------------------------------------------------------- #
# Default path conventions (must mirror manifest.py outputs)
# --------------------------------------------------------------------------- #
def test_default_paths():
    paths = {"crops_root": "/c", "wilddeepfake": "/wd", "df40_diffusion": "/df", "faceforensics_c23": "/ff"}
    m = D._default_manifests(paths)
    np = lambda p: str(p).replace("\\", "/")   # OS-agnostic: Windows joins with backslashes
    assert np(m["faceforensics_c23"]).endswith("/c/faceforensics_c23_manifest.csv")
    assert np(m["wilddeepfake"]).endswith("/wd/wilddeepfake_manifest.csv")
    assert np(D._default_splits_dir(paths)).endswith("/ff/splits")
    assert np(D._default_splits_dir({**paths, "ffpp_splits": "/explicit"})) == "/explicit"


# --------------------------------------------------------------------------- #
# get_dataloaders wiring (uses the stub DataLoader to inspect what was built)
# --------------------------------------------------------------------------- #
def _cfg():
    return {"data": {"image_size": 16, "frequency": {}, "augmentation": {}},
            "train": {"batch_size": 4}}


def _ff_manifest(path):
    write_manifest([
        _row("/x.png", 0, "faceforensics_c23", subset="real", source_video_id="original__033", frame="0"),
        _row("/y.png", 1, "faceforensics_c23", subset="Deepfakes", source_video_id="Deepfakes__033_097", frame="0"),
        _row("/z.png", 0, "faceforensics_c23", subset="real", source_video_id="original__140", frame="0"),
    ], path)


def _df40_manifest(path):
    write_manifest([
        _row("/a.png", 0, "df40_diffusion", subset="real", domain="ff", source_video_id="r1", frame="0"),
        _row("/b.png", 1, "df40_diffusion", subset="stable_diffusion_2_1", domain="ff", source_video_id="f1", frame="0"),
    ], path)


def test_get_dataloaders_wiring(tmp_path):
    sdir = tmp_path / "splits"
    sdir.mkdir()
    _write_splits(sdir)
    ff = tmp_path / "ff_manifest.csv"
    _ff_manifest(ff)
    df = tmp_path / "df40_manifest.csv"
    _df40_manifest(df)
    # celebdf/dfdc/wilddeepfake omitted from manifests -> they should be skipped
    manifests = {"faceforensics_c23": str(ff), "df40_diffusion": str(df)}

    train, val, tests = D.get_dataloaders(
        _cfg(), {}, splits_dir=str(sdir), manifests=manifests, num_workers=0, verbose=False)

    assert len(train.dataset.rows) == 2          # 033 real + 033/097 fake (origin 033 in train)
    assert len(val.dataset.rows) == 1            # 140
    # balanced train (WeightedRandomSampler) vs deterministic val. Real torch always
    # assigns val a SequentialSampler (never None); the conftest stub leaves it None.
    # So assert on the *weighted* sampler type (D.WeightedRandomSampler resolves to the
    # real or stubbed class per environment), not on `val.sampler is None`.
    assert isinstance(train.sampler, D.WeightedRandomSampler)        # balanced train
    assert not isinstance(val.sampler, D.WeightedRandomSampler)      # deterministic val
    assert train.drop_last is True
    assert "df40_stable_diffusion_2_1_ff" in tests
    assert len(tests["df40_stable_diffusion_2_1_ff"].dataset.rows) == 2   # fake + ff real
    assert not any(k in tests for k in ("celebdf_v2", "dfdc", "wilddeepfake"))


def test_get_dataloaders_missing_ffpp_manifest_raises(tmp_path):
    sdir = tmp_path / "splits"
    sdir.mkdir()
    _write_splits(sdir)
    with pytest.raises(FileNotFoundError):
        D.get_dataloaders(_cfg(), {}, splits_dir=str(sdir),
                          manifests={"faceforensics_c23": str(tmp_path / "nope.csv")},
                          num_workers=0, verbose=False)


def test_get_dataloaders_include_indomain_test(tmp_path):
    sdir = tmp_path / "splits"
    sdir.mkdir()
    _write_splits(sdir)                      # test.json = [["300","301"]]
    ff = tmp_path / "ff_manifest.csv"
    write_manifest([
        _row("/a.png", 0, "faceforensics_c23", subset="real", source_video_id="original__033", frame="0"),
        _row("/b.png", 1, "faceforensics_c23", subset="Deepfakes", source_video_id="Deepfakes__033_097", frame="0"),
        _row("/c.png", 0, "faceforensics_c23", subset="real", source_video_id="original__140", frame="0"),
        _row("/d.png", 1, "faceforensics_c23", subset="Deepfakes", source_video_id="Deepfakes__300_301", frame="0"),
        _row("/e.png", 0, "faceforensics_c23", subset="real", source_video_id="original__300", frame="0"),
    ], ff)
    manifests = {"faceforensics_c23": str(ff)}
    _, _, tests = D.get_dataloaders(_cfg(), {}, splits_dir=str(sdir), manifests=manifests,
                                    num_workers=0, include_indomain_test=True, verbose=False)
    assert "faceforensics_c23" in tests
    assert len(tests["faceforensics_c23"].dataset.rows) == 2   # the two 300-origin (test-split) rows
    _, _, tests_off = D.get_dataloaders(_cfg(), {}, splits_dir=str(sdir), manifests=manifests,
                                        num_workers=0, verbose=False)
    assert "faceforensics_c23" not in tests_off               # default off -> backward compatible
