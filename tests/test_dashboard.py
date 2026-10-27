"""Headless dashboard tests with Streamlit's AppTest.

Each page must run without exceptions, and key numbers on the overview must equal an
independent calculation with the same src/ functions. Skipped until the pipeline outputs
exist (see dashboard/common.py REQUIRED).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config, project_path
from src.models.predict import load_model, predict_proba

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
CFG = load_config()
NEEDED = ["models/current_model.json", "data/processed/portfolio_test.parquet",
          "data/processed/alerts/current_alerts.parquet",
          "reports/experiment_results/phase10_monitoring_summary.json"]

pytestmark = pytest.mark.skipif(not all((ROOT / f).exists() for f in NEEDED),
                                reason="pipeline outputs not built")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
PAGES = sorted(p.name for p in (DASH / "pages").glob("*.py"))
OVERVIEW = "1_Portfolio_Overview.py"


def run_page(name: str):
    at = AppTest.from_file(str(DASH / "pages" / name), default_timeout=300)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_eight_pages_exist_and_are_all_in_navigation():
    assert len(PAGES) == 8
    nav = (DASH / "app.py").read_text(encoding="utf-8")
    assert all(p in nav for p in PAGES)


def test_entry_point_opens_overview():
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Executive / Portfolio Overview"


def test_overview_numbers_match_independent_calculation():
    models = project_path(CFG, "models")
    current = json.loads((models / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models / current["model_file"])
    port = pd.read_parquet(project_path(CFG, "processed_data") / "portfolio_test.parquet")
    p = predict_proba(model, port)
    metrics = {m.label: m.value for m in run_page(OVERVIEW).metric}
    assert metrics["Borrowers"] == f"{len(port):,}"
    assert metrics["Mean PD"] == f"{p.mean():.2%}"
    assert metrics["Expected cases (sum of PD)"] == f"{p.sum():,.0f}"
    label = f"Above threshold ({current['threshold']:.1%})"
    assert metrics[label] == f"{int((p >= current['threshold']).sum()):,}"


def test_simulation_page_runs_a_small_simulation():
    at = AppTest.from_file(str(DASH / "pages" / "5_Simulation_and_Stress.py"),
                           default_timeout=300)
    at.run()
    at.number_input[0].set_value(500)
    at.button[0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    assert list(table.iloc[:, 0]) == ["Baseline", "Stressed"]


def test_borrower_page_counterfactual_button():
    at = run_page("2_Borrower_Analysis.py")
    at.selectbox[0].set_value("Just above review threshold")
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("hypothetical" in c.value for c in at.caption)
