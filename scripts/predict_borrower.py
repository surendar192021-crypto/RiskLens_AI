"""Score one portfolio borrower with the deployed model and explain the result.

Read-only: loads the saved model and data, prints the answer, writes nothing.

Run from the project root:
    python -m scripts.predict_borrower            (borrower 215910, the highest PD)
    python -m scripts.predict_borrower 100001     (any SK_ID_CURR from application_test)
"""

import json
import sys

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.explainability.shap_explainer import ShapExplainer, format_value
from src.models.predict import load_model, predict_proba
from src.risk.borrower_risk import assign_risk_band
from src.risk.early_warning import LEVELS, pd_level, resolve_levels


def main() -> int:
    cfg = load_config()
    borrower_id = int(sys.argv[1]) if len(sys.argv) > 1 else 215910

    # 1. Load the deployed model (no training happens here)
    models = project_path(cfg, "models")
    current = json.loads((models / "current_model.json").read_text(encoding="utf-8"))
    model, _ = load_model(models / current["model_file"])
    threshold = float(current["threshold"])

    # 2. Find the borrower in the current portfolio
    port = pd.read_parquet(project_path(cfg, "processed_data") / "portfolio_test.parquet")
    row = port[port["SK_ID_CURR"] == borrower_id]
    if row.empty:
        print(f"Borrower {borrower_id} is not in the portfolio (application_test).")
        return 1

    # 3. Predict: the same single scoring function the dashboard uses
    pd_value = float(predict_proba(model, row)[0])
    bands = cfg["risk_bands"]
    band = assign_risk_band(np.array([pd_value]), bands["edges"], bands["labels"])[0]
    levels = resolve_levels(cfg["early_warning"]["levels"], threshold)
    level = LEVELS[int(pd_level(np.array([pd_value]), levels)[0])]

    print(f"Model:        {current['model_version']}")
    print(f"Borrower:     {borrower_id}")
    print(f"PD:           {pd_value:.1%}  (chance of early payment difficulty)")
    print(f"Threshold:    {threshold:.1%}  -> "
          f"{'FLAG FOR REVIEW' if pd_value >= threshold else 'no review needed'}")
    print(f"Alert level:  {level.upper()}")
    print(f"Risk band:    {band}")

    # 4. Explain: top SHAP contributions (log-odds; + raises risk, - lowers it)
    result = ShapExplainer(model).explain_borrower(row)
    print(f"\nWhy (top 8 SHAP contributions; base value {result['base_value']:.3f}, "
          f"raw score {result['raw_score']:.3f}):")
    for r in result["contributions"].head(8).itertuples():
        direction = "raises" if r.shap > 0 else "lowers"
        print(f"  {r.shap:+.3f}  {direction} risk   {r.feature} = {format_value(r.value)}")
    print("\nContributions describe the model, not causes. Decision support only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
