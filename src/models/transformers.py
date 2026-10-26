"""Custom sklearn transformers and the model pipelines.

Kept in their own module because saved models (joblib) refer to these classes by
import path: moving them would break loading of older artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


class FeatureFrame(BaseEstimator, TransformerMixin):
    """First step of every pipeline: the single schema guard.

    - selects the training features in the training order;
    - fails with a clear message if any feature is missing;
    - casts numerics to float64 and categoricals to the categories seen in training
      (a category never seen in training becomes missing instead of crashing).
    """

    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureFrame:
        self._check_columns(X)
        self.columns_ = list(self.numeric) + list(self.categorical)
        self.categories_ = {
            c: [v for v in pd.Series(X[c]).astype("category").cat.categories]
            for c in self.categorical
        }
        return self

    def _check_columns(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureFrame expects a pandas DataFrame")
        missing = [c for c in list(self.numeric) + list(self.categorical) if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing {len(missing)} feature(s): {missing[:10]}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        out = X[self.columns_].copy()
        for c in self.numeric:
            out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
        for c, cats in self.categories_.items():
            values = out[c].astype("object")
            values = values.where(values.isin(cats))  # unseen category -> missing
            out[c] = pd.Categorical(values, categories=cats)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data (missing stays missing)."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> QuantileClipper:
        A = np.asarray(X, dtype="float64")
        with np.errstate(all="ignore"), _quiet_nan_warnings():
            low = np.nanquantile(A, self.lower, axis=0)
            high = np.nanquantile(A, self.upper, axis=0)
        self.low_ = np.where(np.isnan(low), -np.inf, low)
        self.high_ = np.where(np.isnan(high), np.inf, high)
        self.n_features_in_ = A.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        A = np.asarray(X, dtype="float64")
        return np.clip(A, self.low_, self.high_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class _quiet_nan_warnings:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def to_object(X: Any) -> Any:
    """Categorical -> object so sklearn imputers/encoders accept new fill values."""
    return X.astype("object") if hasattr(X, "astype") else X


def _categorical_to_object() -> FunctionTransformer:
    return FunctionTransformer(to_object, feature_names_out="one-to-one")


def logistic_regression_pipeline(
    numeric: list[str], categorical: list[str], params: dict[str, Any],
    class_weight: Any, seed: int,
) -> Pipeline:
    """Clip outliers -> median impute (+ missing indicators) -> standardise; one-hot categories."""
    numeric_steps = Pipeline(
        [
            ("clip", QuantileClipper(params["clip_lower"], params["clip_upper"])),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("to_object", _categorical_to_object()),
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=params["min_category_frequency"])),
        ]
    )
    pre = ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)]
    )
    model = LogisticRegression(
        C=params["C"], max_iter=params["max_iter"], class_weight=class_weight, random_state=seed
    )
    return _pipeline(numeric, categorical, pre, model)


def _pipeline(numeric: list[str], categorical: list[str], pre: Any, model: Any) -> Pipeline:
    return Pipeline(
        [("features", FeatureFrame(numeric, categorical)), ("pre", pre), ("model", model)]
    )


def random_forest_pipeline(
    numeric: list[str], categorical: list[str], params: dict[str, Any],
    class_weight: Any, seed: int,
) -> Pipeline:
    """Trees need no scaling; sklearn forests accept missing numeric values natively."""
    categorical_steps = Pipeline(
        [
            ("to_object", _categorical_to_object()),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-1)),
        ]
    )
    pre = ColumnTransformer(
        [("num", "passthrough", numeric), ("cat", categorical_steps, categorical)]
    )
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"], min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"], class_weight=class_weight, n_jobs=-1,
        random_state=seed,
    )
    return _pipeline(numeric, categorical, pre, model)
