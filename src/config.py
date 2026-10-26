"""Load and validate config/config.yaml.

Every module reads settings through `load_config()` so that thresholds and
assumptions are defined in exactly one place and checked before use.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

REQUIRED_SECTIONS = ("project", "paths", "data", "splits", "risk_bands")
SPLIT_NAMES = ("train", "validation", "calibration", "test")


class ConfigError(ValueError):
    """Raised when the configuration is missing values or is inconsistent."""


def validate_splits(splits: dict[str, Any]) -> None:
    """Split fractions must all be in (0, 1) and sum to 1."""
    missing = [name for name in SPLIT_NAMES if name not in splits]
    if missing:
        raise ConfigError(f"splits is missing: {missing}")
    fractions = [splits[name] for name in SPLIT_NAMES]
    for name, value in zip(SPLIT_NAMES, fractions, strict=True):
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ConfigError(f"splits.{name} must be a number in (0, 1), got {value!r}")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ConfigError(f"split fractions must sum to 1, got {sum(fractions)}")


def validate_risk_bands(bands: dict[str, Any]) -> None:
    """Band edges must start at 0, end at 1 and strictly increase.

    This guarantees the bands cover every probability in [0, 1] with no gaps
    and no overlaps (each interval is [lower, upper), the last includes 1).
    """
    edges = bands.get("edges")
    labels = bands.get("labels")
    if not edges or not labels:
        raise ConfigError("risk_bands needs both 'edges' and 'labels'")
    if len(labels) != len(edges) - 1:
        raise ConfigError(
            f"risk_bands needs len(labels) == len(edges) - 1, got {len(labels)} labels "
            f"and {len(edges)} edges"
        )
    if edges[0] != 0 or edges[-1] != 1:
        raise ConfigError(f"risk_bands edges must start at 0 and end at 1, got {edges}")
    if any(b <= a for a, b in pairwise(edges)):
        raise ConfigError(f"risk_bands edges must be strictly increasing, got {edges}")
    if len(set(labels)) != len(labels):
        raise ConfigError(f"risk_bands labels must be unique, got {labels}")


def validate_config(cfg: dict[str, Any]) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config is missing sections: {missing}")
    if not isinstance(cfg["project"].get("random_seed"), int):
        raise ConfigError("project.random_seed must be an integer")
    validate_splits(cfg["splits"])
    validate_risk_bands(cfg["risk_bands"])


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config, validate it and return it as a dict."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config file is empty or not a mapping: {path}")
    validate_config(cfg)
    return cfg


def project_path(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path for an entry in the `paths` section."""
    try:
        return PROJECT_ROOT / cfg["paths"][key]
    except KeyError as exc:
        raise ConfigError(f"paths.{key} is not defined in config") from exc
