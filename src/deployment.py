"""
Deployment utilities.

- load_model: lazy-load pipeline
- _get_feature_columns_from_pipeline: infer feature list or load from models/feature_columns.json
- predict_single: align single input dict to expected features, coerce numeric types, predict
- predict_batch: align DataFrame and predict
"""

from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, FEATURES_PATH

_model = None
_feature_cols_cache: Optional[List[str]] = None


def load_model():
    """Lazy-load the best trained model pipeline."""
    global _model
    if _model is None:
        if not Path(BEST_MODEL_PATH).exists():
            raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}. Train models first.")
        _model = joblib.load(BEST_MODEL_PATH)
    return _model


def _get_feature_columns_from_pipeline(model) -> Optional[List[str]]:
    """Try to infer feature columns from pipeline or fallback to feature_columns.json."""
    # 1) pipeline.feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # 2) preprocessor.feature_names_in_
    pre = model.named_steps.get("preprocess", None)
    if pre is not None and hasattr(pre, "feature_names_in_"):
        return list(pre.feature_names_in_)

    # 3) pre.transformers_ columns list
    if pre is not None and hasattr(pre, "transformers_"):
        cols = []
        for _, _, cols_spec in pre.transformers_:
            if isinstance(cols_spec, (list, tuple)):
                cols.extend([c for c in cols_spec if isinstance(c, str)])
        if cols:
            return cols

    # 4) fallback to feature_columns.json
    if Path(FEATURES_PATH).exists():
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data

    return None


def _ensure_input_frame(input_obj: Any, feature_cols: List[str]) -> pd.DataFrame:
    """Ensure input is a DataFrame aligned to feature columns."""
    if isinstance(input_obj, dict):
        row = {c: input_obj.get(c, np.nan) for c in feature_cols}
        return pd.DataFrame([row])
    elif isinstance(input_obj, pd.DataFrame):
        return input_obj.reindex(columns=feature_cols)
    else:
        raise TypeError("Input must be dict or pandas DataFrame")


def _coerce_numeric_columns(df: pd.DataFrame, model) -> pd.DataFrame:
    """Convert numeric columns to proper dtype, coercing errors to NaN."""
    pre = model.named_steps.get("preprocess", None)
    if pre is not None and hasattr(pre, "transformers_"):
        for name, _, cols in pre.transformers_:
            if name == "num" and isinstance(cols, (list, tuple)):
                for c in cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict churn for a single customer record (dict)."""
    model = load_model()
    global _feature_cols_cache
    if _feature_cols_cache is None:
        _feature_cols_cache = _get_feature_columns_from_pipeline(model)

    if _feature_cols_cache is None:
        raise ValueError("Could not determine feature columns. Ensure feature_columns.json exists.")

    df_input = _ensure_input_frame(input_data, _feature_cols_cache)
    df_input = _coerce_numeric_columns(df_input, model)

    try:
        proba = model.predict_proba(df_input)[0, 1]
        pred = model.predict(df_input)[0]
    except Exception as e:
        raise ValueError("Model transform/predict failed: " + str(e)) from e

    return {
        "prediction": int(pred),
        "churn_probability": float(proba)
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict churn for a batch of customer records (DataFrame)."""
    model = load_model()
    global _feature_cols_cache
    if _feature_cols_cache is None:
        _feature_cols_cache = _get_feature_columns_from_pipeline(model)

    if _feature_cols_cache is None:
        raise ValueError("Could not determine feature columns. Ensure feature_columns.json exists.")

    df_input = df.reindex(columns=_feature_cols_cache)
    df_input = _coerce_numeric_columns(df_input, model)

    try:
        probas = model.predict_proba(df_input)[:, 1]
        preds = model.predict(df_input)
    except Exception as e:
        raise ValueError("Model transform/predict failed: " + str(e)) from e

    out = df.copy()
    out["churn_prediction"] = preds.astype(int)
    out["churn_probability"] = probas.astype(float)
    return out