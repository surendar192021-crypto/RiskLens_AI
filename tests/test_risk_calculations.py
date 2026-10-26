import numpy as np
import pytest

from src.config import ConfigError, load_config
from src.risk.borrower_risk import assign_risk_band, band_distribution

CFG = load_config()
EDGES = CFG["risk_bands"]["edges"]
LABELS = CFG["risk_bands"]["labels"]


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, "Very Low"), (0.1999999, "Very Low"), (0.2, "Low"), (0.3999, "Low"),
        (0.4, "Moderate"), (0.6, "High"), (0.7999, "High"), (0.8, "Critical"), (1.0, "Critical"),
    ],
)
def test_band_boundaries(p, expected):
    assert assign_risk_band([p], EDGES, LABELS)[0] == expected


def test_every_probability_gets_exactly_one_band():
    p = np.linspace(0, 1, 100_001)
    bands = assign_risk_band(p, EDGES, LABELS)
    assert not bands.isna().any()
    assert list(bands.categories) == LABELS


def test_invalid_probabilities_and_bands_rejected():
    with pytest.raises(ValueError):
        assign_risk_band([1.01], EDGES, LABELS)
    with pytest.raises(ConfigError):
        assign_risk_band([0.5], [0.0, 0.6, 0.5, 1.0], ["a", "b", "c"])


def test_band_distribution_counts_and_rates():
    p = [0.05, 0.1, 0.25, 0.9]
    y = [0, 1, 0, 1]
    d = band_distribution(p, EDGES, LABELS, y).set_index("band")
    assert d["borrowers"].sum() == 4 and d.loc["Very Low", "borrowers"] == 2
    assert d.loc["Very Low", "observed_rate"] == 0.5
    assert d.loc["Moderate", "borrowers"] == 0 and np.isnan(d.loc["Moderate", "observed_rate"])
    assert d["share"].sum() == pytest.approx(1.0)
