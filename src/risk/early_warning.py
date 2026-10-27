"""Early-warning engine: turns scored borrowers into transparent, reviewable alerts.

Alert level = PD level (from configurable thresholds), optionally raised one step
by factual red flags from the borrower's pre-application history.

Levels: green < yellow < orange < red. Only non-green borrowers become alerts.
Alerts recommend a REVIEW; they are never a lending decision.

Deterioration alerts (a borrower's level worsening between two scoring runs) need
repeated snapshots of the same borrowers. Home Credit has none, so
`compare_snapshots` is only run on explicitly synthetic snapshots in this project.
"""

from __future__ import annotations

import hashlib
import operator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEVELS = ["green", "yellow", "orange", "red"]
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
       "==": operator.eq}


def resolve_levels(levels_cfg: dict[str, Any], operating_threshold: float) -> dict[str, float]:
    """Numeric thresholds for yellow/orange/red; 'operating_threshold' is looked up."""
    out = {}
    for name in ("yellow", "orange", "red"):
        v = levels_cfg[name]
        out[name] = float(operating_threshold if v == "operating_threshold" else v)
        if not 0 < out[name] < 1:
            raise ValueError(f"early_warning level {name} must be in (0, 1), got {out[name]}")
    if not out["yellow"] < out["orange"] < out["red"]:
        raise ValueError(f"levels must increase yellow < orange < red, got {out}")
    return out


def pd_level(p: Any, thresholds: dict[str, float]) -> np.ndarray:
    """Level index per borrower from PD alone (0 green ... 3 red)."""
    p = np.asarray(p, dtype="float64")
    edges = [thresholds["yellow"], thresholds["orange"], thresholds["red"]]
    return np.searchsorted(edges, p, side="right")


def evaluate_flags(df: pd.DataFrame, flags_cfg: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Boolean column per flag. A missing feature value never raises a flag."""
    out = {}
    for name, f in flags_cfg.items():
        if f["feature"] not in df.columns:
            raise KeyError(f"flag {name}: feature {f['feature']} not in data")
        if f["op"] not in OPS:
            raise ValueError(f"flag {name}: unsupported operator {f['op']}")
        v = df[f["feature"]]
        out[name] = (OPS[f["op"]](v, f["value"]) & v.notna()).to_numpy()
    return pd.DataFrame(out, index=df.index)


def final_level(pd_idx: np.ndarray, any_flag: np.ndarray, escalate: bool) -> np.ndarray:
    """Flags raise the level by one step, but never above orange (flags alone are not 'red')."""
    if not escalate:
        return pd_idx
    raised = np.where(any_flag, np.minimum(pd_idx + 1, LEVELS.index("orange")), pd_idx)
    return np.maximum(pd_idx, raised)


def alert_key(borrower_id: Any, level: str, triggers: str, model_version: str) -> str:
    text = f"{borrower_id}|{level}|{triggers}|{model_version}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def generate_alerts(
    scored: pd.DataFrame, flags: pd.DataFrame, thresholds: dict[str, float],
    ew_cfg: dict[str, Any], model_version: str, as_of: str,
    drivers: pd.Series | None = None, id_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """One alert row per non-green borrower.

    `scored` needs columns id_col, probability, risk_band. `drivers` (optional) is a
    Series of driver text aligned with `scored`.
    """
    for c in (id_col, "probability", "risk_band"):
        if c not in scored.columns:
            raise KeyError(f"scored data missing column {c}")
    texts = {name: f["text"] for name, f in ew_cfg["flags"].items()}
    p_idx = pd_level(scored["probability"], thresholds)
    any_flag = flags.any(axis=1).to_numpy()
    lvl = final_level(p_idx, any_flag, ew_cfg["flag_escalation"])

    rows = []
    for i in np.flatnonzero(lvl > 0):  # flagged-but-green borrowers get no alert unless escalated
        p = float(scored["probability"].iloc[i])
        level = LEVELS[lvl[i]]
        triggers = []
        if p_idx[i] > 0:
            name = LEVELS[p_idx[i]]
            triggers.append(f"PD {p:.1%} >= {name} threshold {thresholds[name]:.1%}")
        fired = [texts[f] for f in flags.columns[flags.iloc[i].to_numpy()]]
        triggers += [f"flag: {t}" for t in fired]
        if lvl[i] > p_idx[i]:
            triggers.append("level raised one step by red flag(s)")
        trig = "; ".join(triggers)
        bid = scored[id_col].iloc[i]
        rows.append({
            "alert_key": alert_key(bid, level, trig, model_version),
            id_col: bid,
            "level": level,
            "severity": int(lvl[i]),
            "probability": p,
            "risk_band": str(scored["risk_band"].iloc[i]),
            "pd_level": LEVELS[p_idx[i]],
            "n_flags": len(fired),
            "triggers": trig,
            "main_drivers": "" if drivers is None else str(drivers.iloc[i]),
            "observation_time": as_of,
            "model_version": model_version,
            "recommended_action": ew_cfg["actions"][level],
        })
    cols = ["alert_key", id_col, "level", "severity", "probability", "risk_band", "pd_level",
            "n_flags", "triggers", "main_drivers", "observation_time", "model_version",
            "recommended_action"]
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["severity", "probability"], ascending=False).reset_index(drop=True)


def top_drivers_text(values: np.ndarray, names: list[str], n: int = 3) -> list[str]:
    """Top-n features pushing each borrower's estimate UP (largest positive SHAP)."""
    order = np.argsort(-values, axis=1)[:, :n]
    out = []
    for r, idx in enumerate(order):
        parts = [f"{names[j]} (+{values[r, j]:.2f})" for j in idx if values[r, j] > 0]
        out.append(", ".join(parts))
    return out


class AlertStore:
    """Alert history in a parquet file, with de-duplication.

    - An alert whose key is already active is not added again (last_seen is updated).
    - An active alert for a borrower that no longer receives that alert is closed.
    """

    COLUMNS = ["status", "first_seen", "last_seen", "closed_at"]

    def __init__(self, path: Path, id_col: str = "SK_ID_CURR"):
        self.path = Path(path)
        self.id_col = id_col

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame()

    def upsert(self, alerts: pd.DataFrame, run_time: str) -> dict[str, int]:
        hist = self.load()
        if hist.empty:
            hist = pd.DataFrame(columns=[*alerts.columns, *self.COLUMNS])
        active = hist["status"] == "active"
        active_keys = set(hist.loc[active, "alert_key"])
        new_keys = set(alerts["alert_key"])

        seen = active & hist["alert_key"].isin(new_keys)
        hist.loc[seen, "last_seen"] = run_time
        closing = active & ~hist["alert_key"].isin(new_keys)
        hist.loc[closing, "status"] = "closed"
        hist.loc[closing, "closed_at"] = run_time

        fresh = alerts[~alerts["alert_key"].isin(active_keys)].copy()
        fresh["status"] = "active"
        fresh["first_seen"] = run_time
        fresh["last_seen"] = run_time
        fresh["closed_at"] = None
        hist = pd.concat([hist, fresh], ignore_index=True) if len(hist) else fresh
        self.path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(self.path, index=False)
        return {"new": int(len(fresh)), "unchanged": int(seen.sum()),
                "closed": int(closing.sum()),
                "active_total": int((hist["status"] == "active").sum())}


def compare_snapshots(
    previous: pd.DataFrame, current: pd.DataFrame, id_col: str = "SK_ID_CURR"
) -> pd.DataFrame:
    """Borrowers whose alert level WORSENED between two snapshots of the same borrowers.

    Both frames need id_col, level (green/yellow/orange/red) and probability.
    """
    m = previous[[id_col, "level", "probability"]].merge(
        current[[id_col, "level", "probability"]], on=id_col, suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    rank = {lv: i for i, lv in enumerate(LEVELS)}
    worse = m["level_after"].map(rank) > m["level_before"].map(rank)
    out = m[worse].copy()
    out["change"] = out["level_before"] + " -> " + out["level_after"]
    out["probability_change"] = out["probability_after"] - out["probability_before"]
    return out.reset_index(drop=True)
