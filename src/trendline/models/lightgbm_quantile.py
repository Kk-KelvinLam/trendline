"""LightGBM quantile regression for next-day high / low / close returns."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from trendline.config import (
    FEATURE_COLS,
    LGB_EARLY_STOPPING,
    LGB_N_ESTIMATORS,
    LGB_PARAMS,
    MODEL_DIR,
    QUANTILES,
    TARGETS,
)


def _target_col(target: str) -> str:
    return f"y_{target}"


class QuantileLGBM:
    """Nine models: {high,low,close} × {q10,q50,q90}."""

    name = "lgbm_quantile"

    def __init__(
        self,
        params: dict | None = None,
        n_estimators: int | None = None,
        early_stopping: int | None = None,
    ) -> None:
        self.params = {**LGB_PARAMS, **(params or {})}
        self.n_estimators = int(n_estimators if n_estimators is not None else LGB_N_ESTIMATORS)
        self.early_stopping = int(early_stopping if early_stopping is not None else LGB_EARLY_STOPPING)
        self.models: dict[tuple[str, float], lgb.Booster] = {}

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
    ) -> "QuantileLGBM":
        X_tr = train[FEATURE_COLS]
        X_va = valid[FEATURE_COLS] if valid is not None and len(valid) else None
        self.models = {}
        for target in TARGETS:
            y_tr = train[_target_col(target)]
            y_va = valid[_target_col(target)] if X_va is not None else None
            for q in QUANTILES:
                params = {**self.params, "alpha": q}
                dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
                valid_sets = [dtrain]
                valid_names = ["train"]
                callbacks = [lgb.log_evaluation(period=0)]
                if X_va is not None:
                    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)
                    valid_sets.append(dvalid)
                    valid_names.append("valid")
                    callbacks.append(lgb.early_stopping(self.early_stopping, verbose=False))
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.n_estimators,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks,
                )
                self.models[(target, q)] = booster
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[FEATURE_COLS]
        out = pd.DataFrame(index=frame.index)
        for target in TARGETS:
            for q in QUANTILES:
                key = (target, q)
                col = f"pred_{target}_q{int(q * 100):02d}"
                if key not in self.models:
                    out[col] = np.nan
                    continue
                out[col] = self.models[key].predict(X)
        # Enforce high >= close >= low on the median path (isotonic-lite).
        if {"pred_high_q50", "pred_low_q50", "pred_close_q50"} <= set(out.columns):
            out["pred_high_q50"] = np.maximum(out["pred_high_q50"], out["pred_close_q50"])
            out["pred_low_q50"] = np.minimum(out["pred_low_q50"], out["pred_close_q50"])
        return out

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or MODEL_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        for (target, q), booster in self.models.items():
            path = directory / f"{target}_q{int(q * 100):02d}.txt"
            booster.save_model(str(path))
        return directory

    def load(self, directory: Path | None = None) -> "QuantileLGBM":
        directory = Path(directory or MODEL_DIR)
        self.models = {}
        for target in TARGETS:
            for q in QUANTILES:
                path = directory / f"{target}_q{int(q * 100):02d}.txt"
                if path.exists():
                    self.models[(target, q)] = lgb.Booster(model_file=str(path))
        if not self.models:
            raise FileNotFoundError(f"no LightGBM models in {directory}")
        return self
