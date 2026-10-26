"""Checks on the real built dataset. Skipped until `python -m scripts.build_dataset` has run."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path

CFG = load_config()
PROCESSED = project_path(CFG, "processed_data")
FILES = ["modelling_train.parquet", "portfolio_test.parquet", "splits.parquet", "feature_list.json"]

pytestmark = pytest.mark.skipif(
    not all((PROCESSED / f).exists() for f in FILES),
    reason="processed dataset not built yet",
)


@pytest.fixture(scope="module")
def data():
    with (PROCESSED / "feature_list.json").open(encoding="utf-8") as f:
        features = json.load(f)
    return {
        "train": pd.read_parquet(PROCESSED / "modelling_train.parquet"),
        "test": pd.read_parquet(PROCESSED / "portfolio_test.parquet"),
        "splits": pd.read_parquet(PROCESSED / "splits.parquet"),
        "features": features,
    }


def test_row_counts_and_unique_ids(data):
    assert len(data["train"]) == 307_511 and data["train"]["SK_ID_CURR"].is_unique
    assert len(data["test"]) == 48_744 and data["test"]["SK_ID_CURR"].is_unique


def test_no_leakage_columns_in_feature_list(data):
    feats = data["features"]["numeric"] + data["features"]["categorical"]
    assert "TARGET" not in feats and "SK_ID_CURR" not in feats
    assert len(feats) == len(set(feats))


def test_feature_columns_exist_in_both_tables(data):
    feats = data["features"]["numeric"] + data["features"]["categorical"]
    assert set(feats) <= set(data["train"].columns)
    assert set(feats) <= set(data["test"].columns)
    assert "TARGET" not in data["test"].columns


def test_target_is_binary(data):
    assert set(data["train"]["TARGET"].unique()) == {0, 1}


def test_no_infinite_values(data):
    num = data["train"][data["features"]["numeric"]].to_numpy(dtype="float64")
    assert not np.isinf(num).any()


def test_splits_cover_train_exactly_once(data):
    s = data["splits"]
    assert s["SK_ID_CURR"].is_unique
    assert set(s["SK_ID_CURR"]) == set(data["train"]["SK_ID_CURR"])
    assert set(s["split"]) == {"train", "validation", "calibration", "test"}


def test_split_proportions_and_target_rate(data):
    s = data["splits"].merge(data["train"][["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR")
    shares = s["split"].value_counts(normalize=True)
    for name, share in CFG["splits"].items():
        if name in shares:
            assert shares[name] == pytest.approx(share, abs=0.001)
    rates = s.groupby("split")["TARGET"].mean()
    assert (rates - s["TARGET"].mean()).abs().max() < 0.001


def test_cleaning_was_applied(data):
    t = data["train"]
    assert "DAYS_EMPLOYED" not in t and "DAYS_BIRTH" not in t
    assert t["EMPLOYED_YEARS"].min() >= 0
    assert "XNA" not in set(t["CODE_GENDER"].dropna())
    assert (t["HAS_BUREAU"] == 0).sum() == (t["BUREAU_COUNT"] == 0).sum()
