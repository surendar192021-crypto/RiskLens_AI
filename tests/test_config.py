import copy
from itertools import pairwise

import pytest

from src.config import (
    ConfigError,
    load_config,
    project_path,
    validate_config,
    validate_risk_bands,
    validate_splits,
)


@pytest.fixture
def cfg():
    return load_config()


def test_project_config_loads_and_is_valid(cfg):
    assert cfg["project"]["random_seed"] == 42
    assert cfg["data"]["target_column"] == "TARGET"


def test_splits_sum_to_one(cfg):
    validate_splits(cfg["splits"])


def test_splits_reject_bad_sum():
    with pytest.raises(ConfigError, match="sum to 1"):
        validate_splits({"train": 0.6, "validation": 0.2, "calibration": 0.1, "test": 0.2})


def test_splits_reject_missing_name():
    with pytest.raises(ConfigError, match="missing"):
        validate_splits({"train": 0.7, "validation": 0.15, "test": 0.15})


def test_risk_bands_cover_unit_interval(cfg):
    edges = cfg["risk_bands"]["edges"]
    assert edges[0] == 0 and edges[-1] == 1
    assert all(b > a for a, b in pairwise(edges))


@pytest.mark.parametrize(
    "bands, message",
    [
        ({"edges": [0.0, 0.5, 0.9], "labels": ["A", "B"]}, "end at 1"),
        ({"edges": [0.1, 0.5, 1.0], "labels": ["A", "B"]}, "start at 0"),
        ({"edges": [0.0, 0.5, 0.5, 1.0], "labels": ["A", "B", "C"]}, "strictly increasing"),
        ({"edges": [0.0, 0.5, 1.0], "labels": ["A"]}, "len\\(labels\\)"),
        ({"edges": [0.0, 0.5, 1.0], "labels": ["A", "A"]}, "unique"),
    ],
)
def test_risk_bands_reject_invalid(bands, message):
    with pytest.raises(ConfigError, match=message):
        validate_risk_bands(bands)


def test_missing_section_rejected(cfg):
    broken = copy.deepcopy(cfg)
    del broken["splits"]
    with pytest.raises(ConfigError, match="missing sections"):
        validate_config(broken)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_project_path_resolves(cfg):
    assert project_path(cfg, "raw_data").name == "raw"
    with pytest.raises(ConfigError):
        project_path(cfg, "does_not_exist")
