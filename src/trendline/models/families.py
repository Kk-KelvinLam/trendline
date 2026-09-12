"""Shared / sector / per-stock LightGBM bundles with a common predict interface."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trendline.config import (
    MODEL_SECTOR_DIR,
    MODEL_SHARED_DIR,
    MODEL_STOCK_DIR,
    QUANTILES,
    STOCK_EARLY_STOPPING,
    STOCK_LGB_PARAMS,
    STOCK_MIN_TRAIN_ROWS,
    STOCK_N_ESTIMATORS,
    TARGETS,
)
from trendline.models.lightgbm_quantile import QuantileLGBM
from trendline.universe import model_sector_for, model_sector_map


def pred_col(target: str, q: float, family: str | None = None) -> str:
    base = f"pred_{target}_q{int(q * 100):02d}"
    return f"{base}_{family}" if family else base


def rename_preds(preds: pd.DataFrame, family: str) -> pd.DataFrame:
    out = preds.copy()
    mapping = {}
    for target in TARGETS:
        for q in QUANTILES:
            src = pred_col(target, q)
            if src in out.columns:
                mapping[src] = pred_col(target, q, family)
    return out.rename(columns=mapping)


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "Other"


def _split_train_valid(train: pd.DataFrame, q: float = 0.88) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if train.empty:
        return train, None
    cut = train["date"].quantile(q)
    tr = train[train["date"] <= cut]
    va = train[train["date"] > cut]
    if len(va) < 30:
        return train, None
    return tr, va


class SharedBundle:
    family = "shared"

    def __init__(self) -> None:
        self.model = QuantileLGBM()

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "SharedBundle":
        if valid is None:
            train, valid = _split_train_valid(train)
        self.model.fit(train, valid)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(frame)

    def save(self, directory: Path | None = None) -> Path:
        return self.model.save(Path(directory or MODEL_SHARED_DIR))

    def load(self, directory: Path | None = None) -> "SharedBundle":
        self.model.load(Path(directory or MODEL_SHARED_DIR))
        return self


class SectorBundle:
    family = "sector"

    def __init__(self) -> None:
        self.models: dict[str, QuantileLGBM] = {}
        self.sector_of: dict[str, str] = model_sector_map()

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "SectorBundle":
        work = train.copy()
        work["model_sector"] = work["ticker"].map(lambda t: self.sector_of.get(t, model_sector_for(t)))
        self.models = {}
        for sector, g in work.groupby("model_sector", sort=False):
            if len(g) < 200:
                continue
            if valid is not None and not valid.empty:
                vv = valid.copy()
                vv["model_sector"] = vv["ticker"].map(lambda t: self.sector_of.get(t, model_sector_for(t)))
                va = vv[vv["model_sector"] == sector]
                tr, va_use = g, (va if len(va) > 30 else None)
            else:
                tr, va_use = _split_train_valid(g)
            self.models[str(sector)] = QuantileLGBM().fit(tr, va_use)
        return self

    def predict(self, frame: pd.DataFrame, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predict with sector models; rows without a sector model use fallback (shared)."""
        out = pd.DataFrame(index=frame.index)
        for target in TARGETS:
            for q in QUANTILES:
                out[pred_col(target, q)] = np.nan

        sectors = frame["ticker"].map(lambda t: self.sector_of.get(t, model_sector_for(t)))
        for sector, idx in sectors.groupby(sectors).groups.items():
            model = self.models.get(str(sector))
            if model is None:
                continue
            part = model.predict(frame.loc[idx])
            for c in part.columns:
                out.loc[idx, c] = part[c].to_numpy()

        if fallback is not None:
            for target in TARGETS:
                for q in QUANTILES:
                    col = pred_col(target, q)
                    miss = out[col].isna()
                    if miss.any() and col in fallback.columns:
                        out.loc[miss, col] = fallback.loc[miss, col].to_numpy()
        return out

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or MODEL_SECTOR_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"sectors": sorted(self.models), "sector_of": self.sector_of}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for sector, model in self.models.items():
            model.save(directory / _sanitize(sector))
        return directory

    def load(self, directory: Path | None = None) -> "SectorBundle":
        directory = Path(directory or MODEL_SECTOR_DIR)
        meta_path = directory / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.sector_of = meta.get("sector_of") or model_sector_map()
            sectors = meta.get("sectors") or []
        else:
            self.sector_of = model_sector_map()
            sectors = [p.name for p in directory.iterdir() if p.is_dir()]
        self.models = {}
        for sector in sectors:
            path = directory / _sanitize(sector)
            if path.is_dir() and any(path.glob("*.txt")):
                self.models[sector] = QuantileLGBM().load(path)
        if not self.models:
            raise FileNotFoundError(f"no sector models in {directory}")
        return self


class StockBundle:
    family = "stock"

    def __init__(self, min_train_rows: int = STOCK_MIN_TRAIN_ROWS) -> None:
        self.min_train_rows = int(min_train_rows)
        self.models: dict[str, QuantileLGBM] = {}
        self.trained_tickers: list[str] = []

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "StockBundle":
        self.models = {}
        self.trained_tickers = []
        groups = list(train.groupby("ticker", sort=False))
        n_eligible = sum(1 for _, g in groups if len(g) >= self.min_train_rows)
        done = 0
        for ticker, g in groups:
            if len(g) < self.min_train_rows:
                continue
            if valid is not None and not valid.empty:
                va = valid[valid["ticker"] == ticker]
                tr, va_use = g, (va if len(va) > 20 else None)
            else:
                tr, va_use = _split_train_valid(g, q=0.90)
            model = QuantileLGBM(
                params=STOCK_LGB_PARAMS,
                n_estimators=STOCK_N_ESTIMATORS,
                early_stopping=STOCK_EARLY_STOPPING,
            ).fit(tr, va_use)
            self.models[str(ticker)] = model
            self.trained_tickers.append(str(ticker))
            done += 1
            if done == 1 or done % 50 == 0 or done == n_eligible:
                print(f"    stock fit {done}/{n_eligible}", flush=True)
        return self

    def predict(self, frame: pd.DataFrame, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-ticker preds; thin / missing tickers fall back to shared preds."""
        out = pd.DataFrame(index=frame.index)
        for target in TARGETS:
            for q in QUANTILES:
                out[pred_col(target, q)] = np.nan

        for ticker, idx in frame.groupby("ticker").groups.items():
            model = self.models.get(str(ticker))
            if model is None:
                continue
            part = model.predict(frame.loc[idx])
            for c in part.columns:
                out.loc[idx, c] = part[c].to_numpy()

        if fallback is not None:
            for target in TARGETS:
                for q in QUANTILES:
                    col = pred_col(target, q)
                    miss = out[col].isna()
                    if miss.any() and col in fallback.columns:
                        out.loc[miss, col] = fallback.loc[miss, col].to_numpy()
        return out

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or MODEL_STOCK_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"tickers": sorted(self.models), "min_train_rows": self.min_train_rows}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for ticker, model in self.models.items():
            model.save(directory / _sanitize(ticker))
        return directory

    def load(self, directory: Path | None = None) -> "StockBundle":
        directory = Path(directory or MODEL_STOCK_DIR)
        meta_path = directory / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tickers = meta.get("tickers") or []
            self.min_train_rows = int(meta.get("min_train_rows", STOCK_MIN_TRAIN_ROWS))
        else:
            tickers = [p.name for p in directory.iterdir() if p.is_dir()]
        self.models = {}
        for ticker in tickers:
            path = directory / _sanitize(ticker)
            if path.is_dir() and any(path.glob("*.txt")):
                self.models[ticker] = QuantileLGBM(
                    params=STOCK_LGB_PARAMS,
                    n_estimators=STOCK_N_ESTIMATORS,
                    early_stopping=STOCK_EARLY_STOPPING,
                ).load(path)
        self.trained_tickers = sorted(self.models)
        return self


def empty_pred_frame(index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for target in TARGETS:
        for q in QUANTILES:
            out[pred_col(target, q)] = np.nan
    return out
