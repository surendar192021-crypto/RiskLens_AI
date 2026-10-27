"""Monte Carlo simulation and stress testing, run live with user-chosen assumptions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from common import CFG, model_and_meta, money, page_setup, portfolio, scored  # noqa: E402

from src.models.predict import predict_proba  # noqa: E402
from src.simulation.monte_carlo import loss_statistics, simulate_losses  # noqa: E402
from src.simulation.stress_testing import apply_scenario, check_scenario_output  # noqa: E402

page_setup("Monte Carlo Simulation and Stress Testing")
model, current = model_and_meta()
s, port = scored(), portfolio()
threshold = float(current["threshold"])
mc = CFG["monte_carlo"]
# Defaults = the Phase 9 combined adverse scenario, read from config (not copied by hand).
stress_cfg = CFG["stress_testing"]
adverse = stress_cfg["scenarios"]["combined_adverse"]
st.warning("Everything on this page is ASSUMPTION-BASED and HYPOTHETICAL: correlation and LGD "
           "are assumed, stress scenarios are feature perturbations (not calibrated to "
           "economic data), and results are not regulatory capital or forecasts of real losses.")

with st.form("settings"):
    c = st.columns(4)
    n_sims = c[0].number_input("Simulations", 500, 20000, int(stress_cfg["n_simulations"]), 500)
    seed = c[1].number_input("Random seed", 0, 10**6, int(mc["seed"]))
    rho_mode = c[2].selectbox("Correlation (rho)", ["fixed value", "Basel other-retail formula"])
    rho_val = c[3].slider("rho (if fixed)", 0.0, 0.5, float(mc["rho"]), 0.01)
    c = st.columns(3)
    lgd = c[0].slider("Baseline LGD", 0.0, 1.0, float(CFG["expected_loss"]["lgd"]), 0.05)
    lgd_s = c[1].slider("Stressed LGD", 0.0, 1.0,
                        float(adverse.get("lgd", CFG["expected_loss"]["lgd"])), 0.05)
    st.markdown("**Stress scenario** (changes borrower inputs, then rescoring by the model)")
    c = st.columns(5)
    income_cut = c[0].slider("Income cut", 0.0, 0.6, float(adverse.get("income_cut", 0.0)), 0.05)
    annuity_rise = c[1].slider("Annuity rise", 0.0, 0.6,
                               float(adverse.get("annuity_rise", 0.0)), 0.05)
    ext_cut = c[2].slider("External score cut", 0.0, 0.5,
                          float(adverse.get("ext_source_cut", 0.0)), 0.05)
    job_share = c[3].slider("Job-loss share of employed", 0.0, 0.5,
                            float(adverse.get("job_loss_share", 0.0)), 0.05)
    job_income = c[4].slider("Income after job loss", 0.1, 1.0,
                             float(adverse.get("income_after_job_loss", 0.5)), 0.1)
    run = st.form_submit_button("Run simulation")

rho = "basel_other_retail" if rho_mode.startswith("Basel") else rho_val
spec = {k: v for k, v in {"income_cut": income_cut, "annuity_rise": annuity_rise,
                          "ext_source_cut": ext_cut}.items() if v > 0}
if job_share > 0:
    spec.update({"job_loss_share": job_share, "income_after_job_loss": job_income})


@st.cache_data(show_spinner="Running Monte Carlo simulation...")
def simulate(p: np.ndarray, ead: np.ndarray, lgd: float, rho, n: int, seed: int) -> dict:
    sim = simulate_losses(p, ead, lgd, rho, n, seed, mc["chunk_size"])
    stats = loss_statistics(sim["losses"], float((p * lgd * ead).sum()),
                            mc["confidence_levels"])
    return {"losses": sim["losses"], "stats": stats}


@st.cache_data(show_spinner="Applying scenario and rescoring...")
def stressed_pd(spec_items: tuple, seed: int) -> np.ndarray:
    stressed = apply_scenario(port, dict(spec_items), seed=seed)
    check_scenario_output(port, stressed)
    return predict_proba(model, stressed.drop(columns=["_job_loss"], errors="ignore"))


if not run and "sim_ran" not in st.session_state:
    st.info("Choose assumptions and press **Run simulation**. Defaults reproduce the Phase 9 "
            "combined adverse scenario.")
    st.stop()
st.session_state["sim_ran"] = True

p0, ead = s["probability"].to_numpy(), s["ead"].to_numpy()
p1 = stressed_pd(tuple(sorted(spec.items())), int(seed))
base = simulate(p0, ead, lgd, rho, int(n_sims), int(seed))
stress = simulate(p1, ead, lgd_s, rho, int(n_sims), int(seed))


def row(name: str, p: np.ndarray, res: dict, lgd_used: float) -> dict:
    st_ = res["stats"]
    return {"": name, "Mean PD": f"{p.mean():.2%}", "Expected cases": f"{p.sum():,.0f}",
            "Above threshold": f"{int((p >= threshold).sum()):,}", "LGD": f"{lgd_used:.0%}",
            "Analytic EL": money(st_["analytic_el"]), "Mean sim. loss": money(st_["mean_loss"]),
            "VaR 95%": money(st_["var_95"]), "VaR 99%": money(st_["var_99"]),
            "VaR 99.9%": money(st_["var_99.9"]), "ES 99%": money(st_["es_99"])}


st.dataframe(pd.DataFrame([row("Baseline", p0, base, lgd), row("Stressed", p1, stress, lgd_s)]),
             hide_index=True, width="stretch")
b = base["stats"]
gap_se = (b["mean_loss"] - b["analytic_el"]) / b["mc_standard_error"]
st.caption(f"rho = {rho}; {int(n_sims):,} simulations, seed {int(seed)}. Mean simulated loss "
           f"differs from analytic EL by {gap_se:.1f} Monte Carlo standard errors (baseline "
           "check). Both runs share random draws, so their difference is reliable.")

delta = p1 - p0
down = float((delta < 0).mean())
if spec and down > 0.10:
    st.error(f"Model response check: {down:.0%} of borrowers get a LOWER PD under this adverse "
             "scenario. The model's learned associations are not causal and not monotone "
             "(e.g. for income and instalment size), so this part of the stress result is not "
             "economically credible. See docs/simulation_and_stress.md.")

fig = go.Figure()
fig.add_histogram(x=base["losses"] / 1e6, name="Baseline", opacity=0.6, nbinsx=80,
                  marker_color="#1F77B4")
fig.add_histogram(x=stress["losses"] / 1e6, name="Stressed", opacity=0.6, nbinsx=80,
                  marker_color="#D62728")
for res, name, colour in [(base, "baseline", "#1F77B4"), (stress, "stressed", "#D62728")]:
    fig.add_vline(x=res["stats"]["var_99.9"] / 1e6, line_dash="dot", line_color=colour,
                  annotation_text=f"VaR 99.9% {name}")
fig.update_layout(barmode="overlay", height=420,
                  xaxis_title="Simulated portfolio loss (millions, dataset units)",
                  yaxis_title="Simulations")
st.plotly_chart(fig, width="stretch")

with st.expander("Model definitions and assumptions"):
    st.markdown(
        "- **Vasicek one-factor model:** borrower *i* defaults if "
        "sqrt(rho) Z + sqrt(1 - rho) e_i < Phi^-1(PD_i); Z is shared by all borrowers.\n"
        "- **VaR a** = a-quantile of simulated loss. **ES a** = mean loss beyond VaR a.\n"
        "- **EAD** = AMT_CREDIT, assumed fully outstanding. **LGD** assumed (no recovery data).\n"
        "- **Horizon** = that of TARGET (early payment difficulty), not one year.\n"
        "- **Stress** = borrower inputs changed, derived features recomputed, everyone "
        "rescored by the deployed model. PDs are never multiplied directly.")
