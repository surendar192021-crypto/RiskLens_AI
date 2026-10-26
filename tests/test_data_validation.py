import numpy as np
import pandas as pd
import pytest

from src.data.load_data import optimise_dtypes, raw_path
from src.data.validate_data import (
    DAYS_PLACEHOLDER,
    SchemaError,
    categorical_summary,
    key_relationship,
    key_uniqueness,
    missingness_table,
    numeric_profile,
    require_columns,
    target_distribution,
    temporal_summary,
)


def test_require_columns_names_missing_column():
    with pytest.raises(SchemaError, match="TARGET"):
        require_columns(pd.DataFrame({"a": [1]}), ["a", "TARGET"], "train")


def test_missingness_table_counts_and_sorts():
    df = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, None], "c": [1, 2, 3, 4]})
    out = missingness_table(df)
    assert list(out["column"]) == ["a", "b", "c"]
    assert list(out["n_missing"]) == [2, 1, 0]
    assert out.loc[0, "pct_missing"] == pytest.approx(50.0)


def test_key_uniqueness_detects_duplicates_and_nulls():
    df = pd.DataFrame({"id": [1, 2, 2, None]})
    r = key_uniqueness(df, "id")
    assert (r["n_null"], r["n_unique"], r["n_duplicates"]) == (1, 2, 1)
    assert not r["is_primary_key"]
    assert key_uniqueness(pd.DataFrame({"id": [1, 2, 3]}), "id")["is_primary_key"]


def test_key_relationship_orphans_and_coverage():
    parent = pd.Series([1, 2, 3, 4], name="id")
    child = pd.Series([1, 1, 1, 2, 9], name="id")  # 9 is an orphan
    r = key_relationship(child, parent, "child", "parent")
    assert r["orphan_child_keys"] == 1
    assert r["pct_child_keys_in_parent"] == pytest.approx(200 / 3)
    assert r["pct_parent_with_child"] == pytest.approx(50.0)
    assert r["rows_per_key_max"] == 3


def test_temporal_summary_excludes_placeholder_and_flags_future():
    df = pd.DataFrame({"DAYS_X": [-10, -5, 3, DAYS_PLACEHOLDER, None], "AMT": [1, 2, 3, 4, 5]})
    out = temporal_summary(df, "t")
    assert list(out["column"]) == ["DAYS_X"]
    row = out.iloc[0]
    assert row["n_placeholder_365243"] == 1
    assert row["max"] == 3 and row["min"] == -10
    assert row["n_positive"] == 1
    assert row["n_missing"] == 1


def test_target_distribution_counts():
    r = target_distribution(pd.Series([0, 0, 0, 1]))
    assert r["n_positive"] == 1 and r["positive_rate"] == pytest.approx(0.25)
    assert r["negatives_per_positive"] == pytest.approx(3.0)


@pytest.mark.parametrize("bad", [pd.Series([0, 1, 2]), pd.Series([0, 1, None])])
def test_target_distribution_rejects_non_binary_or_missing(bad):
    with pytest.raises(SchemaError):
        target_distribution(bad)


def test_categorical_summary_only_text_columns():
    df = pd.DataFrame({"g": ["F", "M", "F", None], "x": [1.0, 2.0, 3.0, 4.0]})
    out = categorical_summary(df)
    assert list(out["column"]) == ["g"]
    assert out.loc[0, "n_unique"] == 2 and out.loc[0, "n_missing"] == 1


def test_numeric_profile_quantiles():
    df = pd.DataFrame({"x": np.arange(101, dtype=float)})
    row = numeric_profile(df, ["x"]).iloc[0]
    assert (row["min"], row["median"], row["max"]) == (0, 50, 100)


def test_optimise_dtypes_is_lossless():
    df = pd.DataFrame({"i": [1, 2, 3, 4], "f": [0.1, 0.2, 0.3, 0.4], "s": ["a", "b", "a", "a"]})
    original = df.copy()
    out = optimise_dtypes(df)
    assert out["i"].dtype == np.int8
    assert out["f"].dtype == np.float64
    assert isinstance(out["s"].dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(out.astype(original.dtypes.to_dict()), original)


def test_raw_path_rejects_unknown_table():
    with pytest.raises(KeyError, match="Unknown table"):
        raw_path("not_a_table")
