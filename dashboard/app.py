"""RiskLens AI dashboard: entry point and navigation.

Run from the project root:  streamlit run dashboard/app.py
"""

from pathlib import Path

import streamlit as st

PAGES = Path(__file__).resolve().parent / "pages"

st.set_page_config(page_title="RiskLens AI", layout="wide")
nav = st.navigation({
    "Portfolio": [
        st.Page(PAGES / "1_Portfolio_Overview.py", title="Portfolio Overview", default=True),
        st.Page(PAGES / "4_Portfolio_Analytics.py", title="Portfolio Analytics"),
        st.Page(PAGES / "5_Simulation_and_Stress.py", title="Simulation & Stress Testing"),
    ],
    "Borrowers": [
        st.Page(PAGES / "2_Borrower_Analysis.py", title="Borrower Analysis"),
        st.Page(PAGES / "3_Early_Warning_Center.py", title="Early Warning Center"),
    ],
    "Model": [
        st.Page(PAGES / "6_Model_Explainability.py", title="Model Explainability"),
        st.Page(PAGES / "7_Monitoring_and_Fairness.py", title="Monitoring & Fairness"),
        st.Page(PAGES / "8_Methodology.py", title="Methodology & Model Info"),
    ],
})
st.sidebar.caption("RiskLens AI: decision support for academic demonstration. "
                   "Not a lending authority.")
nav.run()
