"""
oct 14
- coarse + zoom search preliminarily working but needs more fine tuning
-------

1. need to be more thoughtful about the occurrence of "infrequent" prices
2. inelasticity is not being scaled enough (case of unscented 16lb)
3. treat promo vs BAU differently? promo data is more uncommon. so skip aggregation?
"""

# --------- built_in_logic.py  ---------
# (RMSE-focused; Top-N only; adds annualized opps & data range)
from __future__ import annotations
from typing import Callable, Optional, List, Dict, Tuple, Iterable
import itertools
import math
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# viz
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

# ML
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from pygam import ExpectileGAM, s, f

# local import
from helpers import DataEng

# ignore warnings
import warnings


warnings.filterwarnings("ignore")


EXPECTED_COLS = [
    "price",
    "deal_discount_percent",
    "event_encoded",
    "product_encoded",
    "year",
    "month",
    "week",
]

# dtpes
CAT_COLS = ["event_encoded", "product_encoded"]
NUM_COLS = ["price", "deal_discount_percent"]

# features
FEAT_COLS = [
    "price",
    "deal_discount_percent",
    "event_encoded",
    "product_encoded",
    "year",
    "month",
    "week",
]
TARGET_COL = "shipped_units"
WEIGHT_COL = "w"


class ElasticityAnalyzer:
    @staticmethod
    def compute(topsellers: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-9
        elasticity = (
            topsellers.groupby("product")
            .agg(
                asp_max=("price", "max"),
                asp_min=("price", "min"),
                shipped_units_max=("shipped_units", "max"),
                shipped_units_min=("shipped_units", "min"),
                product_count=("product", "count"),
            )
            .reset_index()
        )

        # Calculate elasticity metrics
        elasticity["pct_change_price"] = 100.0 * (
            np.log(np.maximum(elasticity["asp_max"], eps))
            - np.log(np.maximum(elasticity["asp_min"], eps))
        )
        elasticity["pct_change_qty"] = 100.0 * (
            np.log(np.maximum(elasticity["shipped_units_max"], eps))
            - np.log(np.maximum(elasticity["shipped_units_min"], eps))
        )

        # Calculate price elasticity ratio
        elasticity["ratio"] = elasticity["pct_change_qty"] / np.where(
            elasticity["pct_change_price"] == 0, np.nan, elasticity["pct_change_price"]
        )

        # Add normalized elasticity score (0-100)
        elasticity["elasticity_score"] = 100 * (1 - elasticity["ratio"].rank(pct=True))

        # Add confidence metric based on data points
        elasticity["confidence"] = np.minimum(100, elasticity["product_count"] / 5)

        return elasticity.sort_values("ratio", ascending=False).reset_index(drop=True)


class DataEngineer:
    def __init__(self, pricing_df, product_df, top_n=10, granularity="daily"):
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n
        self.granularity = granularity  # "daily", "weekly", "monthly", "run"

    @staticmethod
    def _nonneg(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        return s.clip(lower=0)

    def _days_at_price(self, df) -> pd.DataFrame:
        days_at_asp = (
            df[["asin", "order_date", "price"]]
            .groupby(["asin", "price"])
            .nunique()
            .reset_index()
        )
        days_at_asp.rename(columns={"order_date": "days_sold"}, inplace=True)
        return df.merge(days_at_asp, on=["asin", "price"])

    # synthesize order_date when real dates are missing
    def _synthesize_order_dates(
        self,
        df: pd.DataFrame,
        start="2023-09-17",
        end="2025-09-17",
        seed=42,
        col="order_date",
    ) -> pd.DataFrame:
        out = df.copy()
        start_ts = pd.to_datetime(start).value // 10**9
        end_ts = pd.to_datetime(end).value // 10**9

        rng = np.random.default_rng(seed)
        # generate for all rows to fully override
        rand_ts = rng.integers(start_ts, end_ts, size=len(out), endpoint=False)
        out[col] = pd.to_datetime(rand_ts, unit="s")
        return out

    def _label_encoder(self, df) -> pd.DataFrame:
        """label encoding categorical variable"""
        le = LabelEncoder()
        res = df.copy()

        res["event_encoded"] = le.fit_transform(res["event_name"])
        res["product_encoded"] = le.fit_transform(res["product"])

        return res

    # --- STEP HELPERS ---

    def _normalize_inputs(self):
        # Always normalize columns to lowercase, safe merge
        self.pricing_df = DataEng.clean_cols(self.pricing_df)
        self.product_df = DataEng.clean_cols(self.product_df)

        df = self.pricing_df.merge(self.product_df, how="left", on="asin")
        df["product"] = DataEng.compute_product_series(df)
        for c in ("tag", "variation"):
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
        return df

    def _cast_types(self, df):
        df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")
        if "shipped_units" in df:
            df["shipped_units"] = self._nonneg(df["shipped_units"])
        if "price" in df:
            df["price"] = self._nonneg(df["price"])
        return df

    def _aggregate_daily(self, df):
        df_days = self._days_at_price(df)
        df_days["revenue"] = (df_days["shipped_units"] * df_days["price"]).clip(lower=0)

        # Example: if we collapse to weeks
        if self.granularity == "weekly":
            df_days["year"] = df_days["order_date"].dt.year
            df_days["week"] = df_days["order_date"].dt.isocalendar().week
            group_cols = ["asin", "product", "event_name", "year", "week"]
        else:
            group_cols = ["asin", "product", "event_name", "order_date"]

        df_agg = (
            df_days[
                group_cols
                + ["shipped_units", "revenue", "deal_discount_percent", "current_price"]
            ]
            .groupby(group_cols)
            .agg(
                {
                    "shipped_units": "sum",
                    "revenue": "sum",
                    "deal_discount_percent": "first",
                    "current_price": "first",
                }
            )
            .reset_index()
        )

        df_agg["price"] = (df_agg["revenue"] / df_agg["shipped_units"]).replace(
            [np.inf, -np.inf], np.nan
        )
        df_agg["price"] = (
            pd.to_numeric(df_agg["price"], errors="coerce").fillna(0).round(2)
        )
        return df_days, df_agg

    def _add_temporal_features(self, df):
        df["year"] = df["order_date"].dt.year
        df["month"] = df["order_date"].dt.month
        df["week"] = df["order_date"].dt.isocalendar().week
        return df

    def _filter_top_n(self, df):
        top_n = (
            df.groupby("product")["revenue"]
            .sum()
            .reset_index()
            .sort_values("revenue", ascending=False)["product"]
            .head(self.top_n)
            .tolist()
        )
        return df[df["product"].isin(top_n)].copy()

    def prepare(self):
        df = self._normalize_inputs()
        df = self._cast_types(df)
        df_days, df_agg = self._aggregate_daily(df)
        df_agg = self._add_temporal_features(df_agg)
        df_filtered = self._filter_top_n(df_agg)

        df_filtered["asin"] = df_filtered["asin"].astype(str)
        df_filtered.rename(columns={"revenue": "revenue_share_amt"}, inplace=True)
        df_filtered["revenue_share_amt"] = self._nonneg(
            df_filtered["revenue_share_amt"]
        )

        return self._label_encoder(df_filtered)


class Weighting:
    """
    Lightweight, safe weighting:
      - optional time decay (order_date)
      - rarity as inverse local frequency over `rarity_col` (e.g., price/SKU),
        gently blended & tightly bounded so it cannot dominate
      - single normalization path (median -> 1) keeps scale stable
      - quantile clip + final sanitization (finite, >0) avoids NaN/Inf issues
    """

    def __init__(
        self,
        *,
        time_col: str = "order_date",
        # rarity_col: str = "price",
        half_life_days: int | None = 90,  # None to disable time decay
        # rarity_gamma: float = 0,          # 0=off, 1=full strength
        rarity_bounds: tuple[float, float] = (0.90, 1.12),
        normalize: bool = True,
        clip_quantiles: tuple[float, float] = (0.05, 0.95),
        nan_fill: float = 1.0,
        posinf_fill: float = 3.0,
        neginf_fill: float = 0.0,
    ):
        self.time_col = time_col
        # self.rarity_col = rarity_col
        self.half_life_days = half_life_days
        # self.rarity_gamma = float(rarity_gamma)
        self.rarity_bounds = rarity_bounds
        self.normalize = bool(normalize)
        self.clip_quantiles = clip_quantiles
        self.nan_fill = float(nan_fill)
        self.posinf_fill = float(posinf_fill)
        self.neginf_fill = float(neginf_fill)

    def build(self, df: pd.DataFrame) -> np.ndarray:
        """Return 1-D numpy array of final sample weights aligned to df.index (finite, >0)."""
        w_time = self._time_weights(df, col=self.time_col)
        # w_rar = self._rarity_multiplier(df, col=self.rarity_col)

        # combine
        # w = w_time * w_rar
        w = w_time

        # single normalization (median -> 1)
        if self.normalize:
            med = np.nanmedian(w)
            if np.isfinite(med) and med > 0:
                w = w / med

        # light clipping
        lo_q, hi_q = self.clip_quantiles
        finite_mask = np.isfinite(w)
        if finite_mask.any():
            qlo, qhi = np.nanquantile(w[finite_mask], [lo_q, hi_q])
            if np.isfinite(qlo) and np.isfinite(qhi) and qhi > qlo:
                w = np.clip(w, qlo, qhi)

        # final sanitization
        w = np.nan_to_num(
            w, nan=self.nan_fill, posinf=self.posinf_fill, neginf=self.neginf_fill
        )
        w[w <= 0] = 1e-6
        return w.astype(float, copy=False)

    def _time_weights(self, df: pd.DataFrame, *, col: str) -> np.ndarray:
        if self.half_life_days is None or col not in df.columns:
            return np.ones(len(df), dtype=float)

        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().any():
            ref = s.max()
            dt_days = (ref - s).dt.days.to_numpy()
            # fill NaT deltas with median
            if np.isnan(dt_days).any():
                med_days = np.nanmedian(dt_days)
                if not np.isfinite(med_days):
                    med_days = 0.0
                dt_days = np.where(np.isfinite(dt_days), dt_days, med_days)
            hl = max(1.0, float(self.half_life_days))
            w = 0.5 ** (np.clip(dt_days, 0.0, None) / hl)
            return w.astype(float, copy=False)
        else:
            return np.ones(len(df), dtype=float)


class ParamSearchCV:
    """
    Fixed-grid search over (n_splines, λ) with a single scalar objective:
        objective(ns: int, lam: float) -> float   # lower is better
    Change the defaults below to adjust behavior globally.
    """
    def __init__(
        self,
        *,
        n_splines_grid: Iterable[int] = (16, 24, 32),  # single place to edit
        lam_min: float = 1e-3,                         # single place to edit
        lam_max: float = 50.0,                         # single place to edit
        n_grid: int = 9,                               # single place to edit
        objective: "Callable[[int, float], float]",
        logger_print: "Optional[Callable[[str], None]]" = None,
    ):
        self.n_splines_grid = tuple(int(x) for x in n_splines_grid)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        if not (self.lam_max > self.lam_min > 0):
            raise ValueError("lam_max must be > lam_min > 0")
        self.n_grid = int(n_grid)
        if self.n_grid < 2:
            raise ValueError("n_grid must be >= 2")
        self.objective = objective     # the only function needed
        self._log = logger_print if logger_print is not None else (lambda s: None)

        # public results
        self.best_ns_: "Optional[int]" = None
        self.best_lam_: "Optional[float]" = None
        self.best_val_loss_: "Optional[float]" = None
        self.search_history_: "list[dict]" = []

    def _log(self, msg: str):
        """Timestamped console log for debugging."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    @staticmethod
    def _zoom_bounds_around(
        best: float,
        *,
        factor: float = 5.0,
        lo_floor: float = 1e-8,
        hi_ceiling: float = 1e8,
    ) -> Tuple[float, float]:
        best = float(max(best, lo_floor))
        lo = max(best / float(factor), lo_floor)
        hi = min(best * float(factor), hi_ceiling)
        if not (lo < hi):
            eps = 1e-6
            lo = max(best * (1 - eps), lo_floor)
            hi = min(best * (1 + eps), hi_ceiling)
        return lo, hi

    def _search_lambda_zoom(
        self,
        *,
        evaluate: Callable[[float], float],
        lam_min: float,
        lam_max: float,
        rounds: int,
        n_grid: int,
        zoom_factor: float,
    ) -> Tuple[float, float, List[Dict]]:
        history: List[Dict] = []
        lo, hi = float(lam_min), float(lam_max)
        best_lam: Optional[float] = None
        best_loss: Optional[float] = None

        for r in range(1, int(rounds) + 1):
            grid = self._logspace(lo, hi, int(n_grid))
            losses = []
            for lam in grid:
                lam_f = float(lam)
                loss = float(evaluate(lam_f))
                losses.append(loss)
                history.append({"round": int(r), "lam": lam_f, "loss": loss})

            idx = int(np.argmin(losses))
            cand, cand_loss = float(grid[idx]), float(losses[idx])
            if best_loss is None or cand_loss < best_loss:
                best_lam, best_loss = cand, cand_loss

            print("\n")
            self._log(
                f"[CZ] r{r}: window=({self._fmt(lo)}, {self._fmt(hi)}) → "
                f"best λ={self._fmt(best_lam)} | loss={self._fmt(best_loss)}"
            )
            lo, hi = self._zoom_bounds_around(best_lam, factor=float(zoom_factor))

        return float(best_lam), float(best_loss), history

    def fit(self):

        lam_grid = np.geomspace(self.lam_min, self.lam_max, num=self.n_grid)
        best = {"n_splines": None, "lam": None, "loss": float("inf")}
        self.search_history_.clear()

        for ns in self.n_splines_grid:
            self._log(f"[CZ] ns={ns} | λ-grid=({lam_grid[0]:g} … {lam_grid[-1]:g}) | n_grid={self.n_grid}")
            for lam in lam_grid:
                loss = float(self.objective(int(ns), float(lam)))
                self.search_history_.append({"ns": int(ns), "lam": float(lam), "loss": loss})
                self._log(f"[CZ]   eval ns={ns} λ={lam:g} → loss={loss:.3f}")
                if loss < best["loss"]:
                    best = {"n_splines": int(ns), "lam": float(lam), "loss": loss}

        self.best_ns_ = best["n_splines"]
        self.best_lam_ = best["lam"]
        self.best_val_loss_ = best["loss"]
        self._log(f"[CZ] best ns={self.best_ns_} | best λ={self.best_lam_:g} | val_loss={self.best_val_loss_:.3f}")
        return best

    # ---------- internals ----------
    @staticmethod
    def _fmt(x, nd=4):
        try:
            if x is None:
                return "NA"
            if isinstance(x, (int, float)):
                if math.isnan(x) or math.isinf(x):
                    return str(x)
                return f"{x:.{nd}g}"
            return str(x)
        except Exception:
            return str(x)

    @staticmethod
    def _logspace(lo: float, hi: float, n: int) -> np.ndarray:
        lo = max(lo, 1e-12)
        hi = max(hi, lo * (1 + 1e-12))
        return np.logspace(math.log10(lo), math.log10(hi), int(max(2, n)))


class GAMModeler:
    """
    Minimal GAM wrapper for univariate price → units with expectile bands.
    Keep it tiny: fit(), predict_units(), add_predictions_to_df(), sanitize_results_for_downstream().
    """

    def __init__(self, feature_cols, base_gam_kwargs=None, expectiles=(0.025, 0.5, 0.975)):
        from pygam import ExpectileGAM  # local import to avoid global dependency at import time
        self.feature_cols = list(feature_cols)
        self.expectiles = tuple(expectiles)
        self.base_gam_kwargs = {} if base_gam_kwargs is None else dict(base_gam_kwargs)
        self.models = {}  # q -> fitted ExpectileGAM
        self._ExpectileGAM = ExpectileGAM

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray, weights=None, verbose=False):
        X = train_df[self.feature_cols].to_numpy(dtype=float)
        y = np.asarray(y_train, float)
        w = None if weights is None else np.asarray(weights, float)

        self.models.clear()
        for q in self.expectiles:
            kw = dict(self.base_gam_kwargs)
            kw["expectile"] = float(q)
            gam = self._ExpectileGAM(**kw)
            gam.fit(X, y, weights=w)
            self.models[q] = gam
            if verbose:
                print(f"[GAMModeler] Fitted ExpectileGAM(q={q}) with kwargs={kw}")
        return self

    def predict_units(self, df: pd.DataFrame, expectiles=None) -> dict[float, np.ndarray]:
        if not self.models:
            raise RuntimeError("GAMModeler not fitted")
        qs = self.expectiles if expectiles is None else tuple(expectiles)
        X = df[self.feature_cols].to_numpy(dtype=float)
        out = {}
        for q in qs:
            m = self.models.get(q)
            if m is None:
                raise KeyError(f"Expectile {q} not fitted")
            out[q] = m.predict(X).astype(float)
        return out

    def add_predictions_to_df(self, df: pd.DataFrame, write_units=True, write_revenue=True, price_col="price", inplace=False):
        base = df if inplace else df.copy()
        units = self.predict_units(base)
        price = base[price_col].to_numpy(dtype=float)

        for q, uhat in units.items():
            if write_units:
                base[f"units_pred_{q}"] = uhat
            if write_revenue:
                base[f"revenue_pred_{q}"] = price * uhat
        return base

    @staticmethod
    def sanitize_results_for_downstream(df: pd.DataFrame) -> pd.DataFrame:
        # Keep this as a no-op or add small dtype/ordering fixes if your downstream expects them
        return df



class Optimizer:
    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        df = all_gam_results.copy()

        # choose prediction basis
        if "revenue_pred_0.5" in df.columns:
            base_col = "revenue_pred_0.5"
            avg_col = "revenue_pred_avg" if "revenue_pred_avg" in df.columns else None
        else:
            base_col = "units_pred_0.5" if "units_pred_0.5" in df.columns else None
            avg_col = "units_pred_avg" if "units_pred_avg" in df.columns else None

        if base_col is None:
            raise ValueError(
                "No prediction columns (units or revenue) found in all_gam_results."
            )

        # compute weighted_pred (same logic; just make it numeric/finitized)
        if avg_col is not None:
            if "ratio" in df.columns:
                max_ratio = (
                    float(df["ratio"].max()) if pd.notna(df["ratio"].max()) else 0.0
                )
                if max_ratio > 0:
                    confidence_weight = 1.0 - (df["ratio"] / max_ratio)
                else:
                    confidence_weight = 0.5
            else:
                confidence_weight = 0.5
            # clip & fill for safety
            if not np.isscalar(confidence_weight):
                confidence_weight = confidence_weight.clip(0.0, 1.0).fillna(0.5)
            df["weighted_pred"] = df[base_col] * confidence_weight + df[avg_col] * (
                1.0 - confidence_weight
            )
        else:
            df["weighted_pred"] = df[base_col]

        # ---- minimal sanitization: drop non-finite metrics per selection ----
        base_metric = pd.to_numeric(df[base_col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        df_base = df.loc[base_metric.notna()]

        weighted_metric = pd.to_numeric(df["weighted_pred"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        df_weighted = df.loc[weighted_metric.notna()]

        return {
            "best_avg": DataEng.pick_best_by_group(df_base, "product", base_col),
            "best_weighted": DataEng.pick_best_by_group(
                df_weighted, "product", "weighted_pred"
            ),
        }


class PricingPipeline:

    def __init__(self, pricing_df, product_df, top_n=10, param_search_kwargs=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # expose a simple print-like callable
        # self._log = logger.info
        self._log = print

        self.engineer = DataEngineer(pricing_df, product_df, top_n)
        self.pricing_df = pricing_df
        self.product_df = product_df

    @classmethod
    def from_csv_folder(
        cls,
        base_dir,
        data_folder="data",
        pricing_file="pricing.csv",
        product_file="products.csv",
        top_n=10,
        param_search_kwargs=None,
    ):
        pricing_df = pd.read_csv(os.path.join(base_dir, data_folder, pricing_file))
        product_df = pd.read_csv(os.path.join(base_dir, data_folder, product_file))
        return cls(
            pricing_df, product_df, top_n, param_search_kwargs
        ).assemble_dashboard_frames()

    def _build_curr_price_df(self):
        product = self.engineer.product_df.copy()

        # Ensure asin exists
        if "asin" not in product.columns:
            raise KeyError("product_df must contain an 'asin' column for mapping")

        # Compute product label safely
        if "tag" not in product.columns and "variation" not in product.columns:
            product["product"] = product["asin"].astype(str)
        else:
            product["product"] = DataEng.compute_product_series(product)

        return product

    # ---------------------------- Prep & weights ----------------------------

    def _pp_prepare_topsellers(self) -> pd.DataFrame:
        return self.engineer.prepare()

    def _pp_ensure_columns(self, topsellers: pd.DataFrame) -> None:
        if "asp" not in topsellers.columns and "price" in topsellers.columns:
            topsellers["asp"] = pd.to_numeric(topsellers["price"], errors="coerce")
        if "__intercept__" not in topsellers.columns:
            topsellers["__intercept__"] = 1.0

    def _pp_compute_weights(self, topsellers: pd.DataFrame) -> np.ndarray:
        W = Weighting()
        w = W.build(topsellers)
        return w

    # ---------------------------- Features / numeric ----------------------------

    def _pp_assemble_numeric(self, topsellers: pd.DataFrame):
        need_cols = FEAT_COLS + [TARGET_COL, WEIGHT_COL]
        ts = topsellers[need_cols].copy()

        # numerics
        for c in FEAT_COLS + [TARGET_COL, WEIGHT_COL]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

        ts = ts.dropna(subset=need_cols).reset_index(drop=True)

        X = ts[FEAT_COLS].to_numpy(dtype=float)
        y = ts[TARGET_COL].to_numpy(dtype=float)
        w = ts[WEIGHT_COL].to_numpy(dtype=float)
        w = np.nan_to_num(w, nan=1.0, posinf=3.0, neginf=1e-6)
        w[w <= 0] = 1e-6
        return ts, X, y, w

    def _pp_elasticity_best_effort(self, topsellers: pd.DataFrame) -> pd.DataFrame:
        try:
            return ElasticityAnalyzer.compute(topsellers)
        except Exception:
            return pd.DataFrame(columns=["product", "ratio", "elasticity_score"])

    def _pp_standardize_continuous_inplace(self, X: np.ndarray) -> np.ndarray:
        _cont = [c for c in ["price", "deal_discount_percent", "year", "month", "week"] if c in FEAT_COLS]
        idx = [FEAT_COLS.index(c) for c in _cont]
        if idx:
            Z = X[:, idx]
            mu = np.nanmean(Z, axis=0)
            sd = np.nanstd(Z, axis=0)
            sd[sd == 0] = 1.0
            X[:, idx] = (Z - mu) / sd
        return X

    # ---------------------------- Split & objective ----------------------------

    def _pp_train_val_split(self, ts: pd.DataFrame, y_all: np.ndarray, w_all: np.ndarray):
        X_price = ts[["price"]].to_numpy(dtype=float)
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X_price, y_all.copy(), w_all.copy(), test_size=0.20, random_state=42
        )
        w_tr = np.clip(np.nan_to_num(w_tr, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None)
        w_val = np.clip(np.nan_to_num(w_val, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None)
        return {
            "X_tr": X_tr, "X_val": X_val,
            "y_tr": y_tr, "y_val": y_val,
            "w_tr": w_tr, "w_val": w_val,
            "price_val": X_val[:, 0],
            "w_all": w_all, "y_all": y_all
        }

    def _pp_make_objective(self, split: dict):
        from pygam import ExpectileGAM
        X_tr, X_val = split["X_tr"], split["X_val"]
        y_tr, y_val = split["y_tr"], split["y_val"]
        w_tr, w_val = split["w_tr"], split["w_val"]
        price_val   = split["price_val"]

        # sparsity-aware validation weighting
        q = np.linspace(0, 1, 21)
        edges = np.quantile(price_val, q)
        idx_bin = np.clip(np.searchsorted(edges, price_val, side="right") - 1, 0, len(edges) - 2)
        counts = np.bincount(idx_bin, minlength=len(edges) - 1)
        mask = counts[idx_bin] >= 10
        w_val_eff = np.where(mask, w_val, 0.0)
        w_val_eff = np.clip(np.nan_to_num(w_val_eff, nan=1.0), 0.1, 5.0)

        def objective(ns: int, lam: float) -> float:
            gam = ExpectileGAM(expectile=0.5, n_splines=int(ns), lam=float(lam), max_iter=5000)
            gam.fit(X_tr, y_tr, weights=w_tr)

            units_val = gam.predict(X_val).astype(float)
            rev_true = price_val * y_val
            rev_hat  = price_val * units_val

            se = (rev_true - rev_hat) ** 2
            num = float(np.sum(w_val_eff * se))
            den = float(np.sum(w_val_eff)) if np.sum(w_val_eff) > 0 else len(se)
            base = np.median(np.abs(rev_true)) or 1.0
            scale = base if np.isfinite(base) and base > 1.0 else 1.0
            return (num / max(den, 1.0)) / (scale ** 2)

        return objective

    # ---------------------------- Tuning & fit ----------------------------

    def _pp_tune(self, objective) -> dict:
        ps = ParamSearchCV(objective=objective, logger_print=self._log).fit()
        # ps is a dict per the simplified implementation
        return ps

    def _pp_fit_full(self, ts: pd.DataFrame, y_all: np.ndarray, w_all: np.ndarray, ns_star: int, lam_star: float):
        modeler = GAMModeler(
            feature_cols=["price"],
            base_gam_kwargs={"lam": float(lam_star), "n_splines": int(ns_star)},
        )
        modeler.fit(
            train_df=ts[modeler.feature_cols],
            y_train=y_all,
            weights=w_all,
            verbose=True,
        )
        return modeler

    # ---------------------------- Results assembly ----------------------------

    def _pp_assemble_results(self, topsellers: pd.DataFrame, modeler) -> pd.DataFrame:
        return (
            topsellers[["product", "price", "asin", "asp"]]
            .copy()
            .reset_index(drop=True)
        )

    def _pp_add_support_counts(self, topsellers: pd.DataFrame, all_gam_results: pd.DataFrame) -> pd.DataFrame:
        base_df = getattr(self, "pricing_df", None)
        if base_df is None and hasattr(self, "engineer") and hasattr(self.engineer, "pricing_df"):
            base_df = self.engineer.pricing_df

        all_gam_results["support_count"] = 0
        for prod_key in all_gam_results["product"].dropna().unique():
            g_pred = all_gam_results.loc[all_gam_results["product"] == prod_key]
            if base_df is not None and {"product", "price"}.issubset(base_df.columns):
                obs_prices = base_df.loc[base_df["product"].astype(str) == str(prod_key), "price"].to_numpy()
                if obs_prices.size == 0:
                    obs_prices = g_pred["price"].to_numpy()
            else:
                obs_prices = g_pred["price"].to_numpy()

            u = np.unique(np.sort(obs_prices))
            step = (np.nanmedian(np.diff(u)) if u.size >= 2 else max(0.25, np.nanstd(obs_prices) / 25.0))
            win = 1.25 * step
            P_pred = g_pred["price"].to_numpy()
            supp = np.array([(np.abs(obs_prices - p) <= win).sum() for p in P_pred], dtype=int)
            all_gam_results.loc[g_pred.index, "support_count"] = supp

        all_gam_results["support_count"] = all_gam_results["support_count"].fillna(0).astype(int)
        return all_gam_results

    def _pp_add_predictions(self, modeler, df: pd.DataFrame) -> pd.DataFrame:
        return modeler.add_predictions_to_df(
            df, write_units=True, write_revenue=True, price_col="price", inplace=False
        )

    def _pp_postprocess_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        for q_ in (0.025, 0.5, 0.975):
            ucol = f"units_pred_{q_}"
            rcol = f"revenue_pred_{q_}"
            if ucol in df.columns:
                df[ucol] = np.maximum(df[ucol].to_numpy(), 0.0)
                if rcol in df.columns:
                    df[rcol] = df["price"].to_numpy() * df[ucol].to_numpy()
        pred_cols = [c for c in df.columns if c.startswith(("units_pred_", "revenue_pred_"))]
        if not pred_cols:
            raise ValueError(f"[_pp_postprocess_predictions] Prediction assembly failed. DataFrame cols={list(df.columns)}")
        return df

    def _pp_passthrough_actuals(self, topsellers: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        df["deal_discount_percent"] = (
            topsellers["deal_discount_percent"].fillna(0).clip(lower=0).reset_index(drop=True)
        )
        df["revenue_actual"] = topsellers["price"].to_numpy() * np.asarray(topsellers["shipped_units"], dtype=float)
        df["daily_rev"] = df["revenue_actual"]
        df["actual_revenue_scaled"] = (
            topsellers["price"].to_numpy()
            * (1 - df["deal_discount_percent"].to_numpy() / 100.0)
            * np.asarray(topsellers["shipped_units"], dtype=float)
        )
        return df

    def _build_core_frames(self):
        """
        Glue: prepare data, weights, features, tune, refit, assemble outputs.
        Returns: topsellers, elasticity_df, all_gam_results
        """
        print("\n─────────────────────────────── 🧮 Starting Data Engineering ───────────────────────────────")
        topsellers = self._pp_prepare_topsellers()
        if topsellers.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        self._pp_ensure_columns(topsellers)
        print("✅ Data loaded & preprocessed. Proceeding to weight computation...")

        w_stable = self._pp_compute_weights(topsellers)
        topsellers[WEIGHT_COL] = w_stable
        print(f"⚖️  Weights computed | median={np.nanmedian(w_stable):.3f} | p95={np.nanpercentile(w_stable,95):.3f}")

        ts, X, y, w = self._pp_assemble_numeric(topsellers)
        print(f"⚖️  Weights ready | median={float(np.nanmedian(w)):.3f} | p95={float(np.nanpercentile(w,95)):.3f} | n={len(w):,}")

        elasticity_df = self._pp_elasticity_best_effort(topsellers)

        X = self._pp_standardize_continuous_inplace(X)

        print("\n\n" + 35 * "- " + " 🤖 Modeling, Tuning & Prediction " + "- " * 35)
        if "price" not in ts.columns:
            raise ValueError("Expected 'price' column to exist for univariate GAM tuning.")

        split = self._pp_train_val_split(ts, y, w)  # dict with X_tr, X_val, ...
        objective = self._pp_make_objective(split)

        best = self._pp_tune(objective)  # dict with n_splines, lam, loss
        ns_star, lam_star = int(best["n_splines"]), float(best["lam"])

        modeler = self._pp_fit_full(ts, y, w, ns_star, lam_star)

        all_gam_results = self._pp_assemble_results(topsellers, modeler)
        all_gam_results = self._pp_add_support_counts(topsellers, all_gam_results)
        all_gam_results = self._pp_add_predictions(modeler, all_gam_results)
        all_gam_results = self._pp_postprocess_predictions(all_gam_results)
        all_gam_results = self._pp_passthrough_actuals(topsellers, all_gam_results)
        all_gam_results = modeler.sanitize_results_for_downstream(all_gam_results)

        print("\n" + 32 * "- " + " 🎯 Pipeline Complete at " + datetime.now().strftime("%H:%M:%S") + " " + 32 * "- " + "\n")
        return topsellers, elasticity_df, all_gam_results


    def _build_best50(self, all_gam_results):
        """Pick best P50 revenue row per product, but only if support is sufficient."""
        # require these cols
        need = {"revenue_pred_0.5", "units_pred_0.5", "support_count"}
        if not need.issubset(all_gam_results.columns):
            # graceful fallback to old behavior
            idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
            ...
            return best50

        MIN_SUPPORT = 6  # tune: 5–8 works well
        eligible = all_gam_results[all_gam_results["support_count"] >= MIN_SUPPORT]
        # if a product has no eligible rows, fall back to the global argmax for that product
        if eligible.empty:
            idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
        else:
            idx = eligible.groupby("product")["revenue_pred_0.5"].idxmax()

        best50 = (
            all_gam_results.loc[
                idx,
                [
                    "product",
                    "asin",
                    "price",
                    "asp",
                    "units_pred_0.5",
                    "revenue_pred_0.5",
                    "support_count",
                ],
            ]
            .drop_duplicates(subset=["product"])
            .reset_index(drop=True)
        )
        return best50

    def _normalize_key_types(self, *dfs):
        """Ensure asin is str across frames."""
        for df in dfs:
            if isinstance(df, pd.DataFrame) and "asin" in df.columns:
                df["asin"] = df["asin"].astype(str)

    def _compute_date_meta(self):
        """Compute data range information"""
        df_dates = self.engineer.pricing_df.copy()
        df_dates["order_date"] = pd.to_datetime(df_dates["order_date"], errors="coerce")
        data_start = (
            pd.to_datetime(df_dates["order_date"].min()) if len(df_dates) else pd.NaT
        )
        data_end = (
            pd.to_datetime(df_dates["order_date"].max()) if len(df_dates) else pd.NaT
        )
        days_covered = (
            int((data_end - data_start).days) + 1
            if pd.notna(data_start) and pd.notna(data_end)
            else 0
        )
        return {
            "data_start": data_start,
            "data_end": data_end,
            "days_covered": days_covered,
        }

    def _build_opportunity_summary(self, best50, all_gam_results, curr_price_df):
        """Compute per-product upside at recommended vs current"""
        rows = []
        for _, r in best50.iterrows():
            p = r["product"]
            pk = str(r["asin"])
            price_best = float(r["price"])
            units_best = float(r.get("units_pred_0.5", np.nan))
            rev_best = float(r.get("revenue_pred_0.5", np.nan))

            prod_curve = all_gam_results[(all_gam_results["product"] == p)]
            cp_ser = curr_price_df.loc[curr_price_df["asin"] == pk, "current_price"]
            curr_price = float(cp_ser.iloc[0]) if len(cp_ser) else np.nan
            curr_row = (
                DataEng.nearest_row_at_price(prod_curve, curr_price)
                if pd.notna(curr_price)
                else None
            )

            if curr_row is not None:
                units_curr = float(curr_row.get("units_pred_0.5", np.nan))
                rev_curr = float(curr_row.get("revenue_pred_0.5", np.nan))
            else:
                units_curr, rev_curr = np.nan, np.nan

            du = (
                (units_best - units_curr)
                if (pd.notna(units_best) and pd.notna(units_curr))
                else np.nan
            )
            dr = (
                (rev_best - rev_curr)
                if (pd.notna(rev_best) and pd.notna(rev_curr))
                else np.nan
            )

            rows.append(
                {
                    "product": p,
                    "asin": pk,
                    "current_price": curr_price,
                    "best_price": price_best,
                    "units_pred_best": units_best,
                    "units_pred_curr": units_curr,
                    "revenue_pred_best": rev_best,
                    "revenue_pred_curr": rev_curr,
                    "delta_units": du,
                    "delta_revenue": dr,
                    "delta_units_annual": (du * 365 if pd.notna(du) else np.nan),
                    "delta_revenue_annual": (dr * 365 if pd.notna(dr) else np.nan),
                    "revenue_best_annual": (
                        rev_best * 365 if pd.notna(rev_best) else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _compute_model_fit_kpi(self, all_gam_results: pd.DataFrame) -> dict:
        """Compute model fit metrics"""
        df = all_gam_results.copy()
        # Backfill revenue_actual if not present
        if "revenue_actual" not in df.columns and {"price", "shipped_units"}.issubset(
            df.columns
        ):
            df["revenue_actual"] = pd.to_numeric(
                df["price"], errors="coerce"
            ) * pd.to_numeric(df["shipped_units"], errors="coerce")
        # Derive revenue_pred_* from units_pred_* if needed
        if "price" in df.columns:
            for q in ("0.025", "0.5", "0.975"):
                up = f"units_pred_{q}"
                rp = f"revenue_pred_{q}"
                if rp not in df.columns and up in df.columns:
                    df[rp] = pd.to_numeric(df[up], errors="coerce") * pd.to_numeric(
                        df["price"], errors="coerce"
                    )

        # Actual daily revenue
        # Ensure denominator is a Series; missing or zero -> NaN
        if "days_sold" in df.columns:
            den = pd.to_numeric(df["days_sold"], errors="coerce")
        else:
            den = pd.Series(np.nan, index=df.index)
        den = den.mask(den == 0)
        act_rev = (
            pd.to_numeric(df["revenue_actual"], errors="coerce")
            if "revenue_actual" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        daily_act = (act_rev / den).where(den.notna(), act_rev)

        out = {}

        # P50 metrics
        pred50_rev = (
            pd.to_numeric(df["revenue_pred_0.5"], errors="coerce")
            if "revenue_pred_0.5" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        daily_pred50 = (pred50_rev / den).where(den.notna(), pred50_rev)

        mask50 = (daily_act > 0) & daily_pred50.notna()
        w50 = act_rev.where(mask50, np.nan)
        pct50 = (daily_pred50 - daily_act) / daily_act
        out["pct_diff_p50"] = (
            float(np.nansum(pct50 * w50) / np.nansum(w50))
            if np.nansum(w50) > 0
            else np.nan
        )

        # Weighted metrics
        if "weighted_pred" in df.columns:
            rev_w = pd.to_numeric(df["weighted_pred"] * df["price"], errors="coerce")
            daily_pred_w = (rev_w / den).where(den.notna(), rev_w)
            maskw = (daily_act > 0) & daily_pred_w.notna()
            ww = act_rev.where(maskw, np.nan)
            pctw = (daily_pred_w - daily_act) / daily_act
            out["pct_diff_weighted"] = (
                float(np.nansum(pctw * ww) / np.nansum(ww))
                if np.nansum(ww) > 0
                else np.nan
            )
        else:
            out["pct_diff_weighted"] = np.nan

        return out

    def assemble_dashboard_frames(self) -> dict:
        # 1) Core frames with elasticity
        topsellers, elasticity_df, all_gam_results = self._build_core_frames()

        # 2) Optimizer tables
        optimizer_results = Optimizer.run(all_gam_results)  # Get all optimizer results
        best_avg = optimizer_results["best_avg"]
        best_weighted = optimizer_results["best_weighted"]  # Get best_weighted result

        # 3) Current prices
        curr_price_df = self._build_curr_price_df()

        # 4) Best P50
        best50 = self._build_best50(all_gam_results)

        # 5) Meta information
        meta = self._compute_date_meta()

        # 6) Opportunities summary
        opps_summary = self._build_opportunity_summary(
            best50, all_gam_results, curr_price_df
        )

        # 7) Normalize key types
        self._normalize_key_types(
            best_avg,
            all_gam_results,
            curr_price_df,
            topsellers,
            best50,
            opps_summary,
            best_weighted,
        )

        # 8) Pack results with all components
        return {
            "price_quant_df": topsellers.groupby(["price", "product"])["shipped_units"]
            .sum()
            .reset_index(),
            "best_avg": best_avg,
            "best_weighted": best_weighted,  # Add this line
            "all_gam_results": all_gam_results,
            "best_optimal_pricing_df": best50.copy(),
            "elasticity_df": elasticity_df,
            "curr_price_df": curr_price_df,
            "opps_summary": opps_summary,
            "meta": meta,
            "curr_opt_df": best_avg,
            "model_fit_kpi": self._compute_model_fit_kpi(
                all_gam_results
            ),  # Add model fit KPIs
        }
 

    # ——— simple weighted MSE we’ll reuse ———
    def _weighted_mse(self, y_true, y_pred, w=None) -> float:
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        if w is None:
            return float(np.mean((y_true - y_pred) ** 2))
        w = np.asarray(w, float)
        w = np.clip(np.nan_to_num(w, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None)
        return float(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))



    def _predict_units(self, model, df_val) -> np.ndarray:
        """
        Return a 1D float array of unit predictions for df_val.
        Tries, in order:
        - model.predict(df_val)
        - model.predict_units(df_val)
        - model.model.predict(df_val.values)
        - fallback via add_predictions_to_df(...), reading a units_pred_* column
        """

        # 1) direct predict on modeler
        if hasattr(model, "predict"):
            yhat = model.predict(df_val)
            return np.asarray(yhat, dtype=float).ravel()

        # 2) dedicated units predictor
        if hasattr(model, "predict_units"):
            yhat = model.predict_units(df_val)
            return np.asarray(yhat, dtype=float).ravel()

        # 3) predict on inner model (common in wrappers)
        base = getattr(model, "model", None)
        if base is not None and hasattr(base, "predict"):
            # many estimators expect ndarray, not DataFrame
            yhat = base.predict(getattr(df_val, "values", df_val))
            return np.asarray(yhat, dtype=float).ravel()

        # 4) robust fallback through add_predictions_to_df
        if hasattr(model, "add_predictions_to_df"):
            tmp = df_val.copy()
            # ensure the required price column exists; your df_val already has "price"
            out = model.add_predictions_to_df(
                tmp,
                write_units=True,
                write_revenue=False,
                price_col="price",
                inplace=False,
            )
            # pick a sensible central tendency column
            for col in ("units_pred_0.5", "units_pred", "units_pred_mean"):
                if col in out.columns:
                    return np.asarray(out[col], dtype=float).ravel()
            # last resort: first units_pred_* column
            unit_cols = [c for c in out.columns if c.startswith("units_pred_")]
            if unit_cols:
                return np.asarray(out[unit_cols[0]], dtype=float).ravel()

        raise AttributeError(
            "GAMModeler does not expose a prediction path. "
            "Expected one of: predict, predict_units, model.predict, or add_predictions_to_df."
        )


class viz:
    """
    Refactored Viz class kept in built_in_logic.py for predictive graph.
    Ensures:
        - ASP sorted before line plotting (no zig-zags)
        - Proper uncertainty band (P2.5 - P97.5)
        - Recommended/Conservative/Optimistic ASP markers
        - Optional per-series faint P50 overlays
    """

    def __init__(self, template="lux"):
        templates = [
            "bootstrap",
            "minty",
            "pulse",
            "flatly",
            "quartz",
            "cyborg",
            "darkly",
            "vapor",
            "lux",
        ]
        load_figure_template(templates)
        self.template = template

    def revenue_axis_label(self, granularity="weekly"):
        """granualrity = weekly/monthly/daily"""
        if granularity == "weekly":
            return "Expected Weekly Revenue ($)"
        elif granularity == "monthly":
            return "Expected Monthly Revenue ($)"
        else:
            return "Expected Avg Daily Revenue ($)"

    def gam_results(self, all_gam_results: pd.DataFrame):
        need_cols = [
            "product",
            "asp",
            "revenue_actual",
            "revenue_pred_0.025",
            "revenue_pred_0.5",
            "revenue_pred_0.975",
        ]
        missing = [c for c in need_cols if c not in all_gam_results.columns]
        if missing:
            return self.empty_fig(f"Missing column(s): {', '.join(missing)}")

        fig = go.Figure()

        if "order_date" in all_gam_results.columns:
            dates = pd.to_datetime(all_gam_results["order_date"])
            date_nums = (dates - dates.min()) / (dates.max() - dates.min())
            opacities = 0.05 + (0.75 * date_nums)
        else:
            opacities = pd.Series(0.55, index=all_gam_results.index)

        for group_name, g in all_gam_results.groupby("product"):
            g = g.dropna(subset=["asp"]).copy()
            if g.empty:
                continue
            g = g.sort_values("asp")
            group_opacities = (
                opacities[g.index] if "order_date" in all_gam_results.columns else 0.55
            )

            # ------- pred bands -------
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_pred_0.975"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_pred_0.025"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(232, 233, 235, 0.7)",  # bright gray, opc = 0.7
                    opacity=0.25,
                    name=f"{group_name} • Predicted Rev Band",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_pred_0.5"],
                    mode="lines",
                    name=f"{group_name} • Predicted Rev (P50)",
                    line=dict(color="rgba(184, 33, 50, 1)"),
                )
            )

            # ----------------------- actual ---------------------------------
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_actual"],
                    mode="markers",
                    marker_symbol="x",
                    name=f"{group_name} • Actual Revenue",
                    marker=dict(size=8, color="#808992", opacity=group_opacities),
                )
            )
            # ------------------- Diamonds for pred prices -------------------
            best_rows = {
                "Recommended (P50)": (
                    "revenue_pred_0.5",
                    g.loc[g["revenue_pred_0.5"].idxmax()],
                ),
                "Conservative (P2.5)": (
                    "revenue_pred_0.025",
                    g.loc[g["revenue_pred_0.025"].idxmax()],
                ),
                "Optimistic (P97.5)": (
                    "revenue_pred_0.975",
                    g.loc[g["revenue_pred_0.975"].idxmax()],
                ),
            }
            marker_colors = {
                "Conservative (P2.5)": "#1F6FEB",
                "Optimistic (P97.5)": "#238636",
                "Recommended (P50)": "#B82132",
            }
            for label, (pred_col, row) in best_rows.items():
                fig.add_trace(
                    go.Scatter(
                        x=[row["asp"]],
                        y=[row[pred_col]],
                        mode="markers",
                        marker=dict(
                            color=marker_colors[label], size=12, symbol="diamond"
                        ),
                        name=f"{group_name} • {label}",
                        # legendgroup=group_name,
                        hovertemplate=f"{label}<br>Price=%{{x:$,.2f}}<br>Rev=%{{y:$,.0f}}<extra></extra>",
                    )
                )

        # 3) Layout (avoid crashing if self.template is not set)
        template = getattr(self, "template", None)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            template=template if template else None,
            yaxis_title=self.revenue_axis_label(granularity="weekly"),
            xaxis_title="Average Selling Price (ASP)",
        )

        return fig

    def elast_dist(self, elast_df: pd.DataFrame):
        fig = (
            px.histogram(
                elast_df, x="ratio", width=1200, height=600, template=self.template
            )
            .update_xaxes(title_text="Elasticity")
            .update_yaxes(title_text="Product Count")
        )
        return fig

    def opportunity_chart(self, elast_df, best50_df, curr_df, all_gam):
        """
        Build the Top-N upside bar chart (Expected Revenue Δ at recommended vs current price).
        Aligns by product, uses P50 revenue, and annotates with elasticity.
        """
        # basic guards
        for df in (elast_df, best50_df, curr_df, all_gam):
            if df is None or getattr(df, "empty", True):
                return self.empty_fig("No data for opportunity chart")

        # required columns
        need_cols = [
            (all_gam, ["product", "asp", "revenue_pred_0.5"]),
            (best50_df, ["product", "asp", "revenue_pred_0.5"]),
            (curr_df, ["product", "current_price"]),
            (elast_df, ["product", "ratio"]),
        ]
        for frame, cols in need_cols:
            for c in cols:
                if c not in frame.columns:
                    return self.empty_fig(f"Missing column: {c}")

        # numeric coercions
        to_num = [
            (all_gam, ["asp", "revenue_pred_0.5"]),
            (best50_df, ["asp", "revenue_pred_0.5"]),
            (curr_df, ["current_price"]),
            (elast_df, ["ratio"]),
        ]
        for frame, cols in to_num:
            for c in cols:
                frame[c] = (
                    pd.to_numeric(frame[c], errors="coerce")
                    if c in frame.columns
                    else frame.get(c)
                )

        prods = sorted(
            set(all_gam["product"])
            & set(best50_df["product"])
            & set(curr_df["product"])
        )
        if not prods:
            return self.empty_fig("No overlapping products across inputs")

        rows = []
        for p in prods:
            try:
                cp_ser = curr_df.loc[curr_df["product"] == p, "current_price"]
                if cp_ser.empty or pd.isna(cp_ser.iloc[0]):
                    continue
                curr_price = float(cp_ser.iloc[0])

                prod_curve = all_gam[
                    (all_gam["product"] == p)
                    & pd.notna(all_gam["asp"])
                    & pd.notna(all_gam["revenue_pred_0.5"])
                ]
                if prod_curve.empty:
                    continue

                # current revenue ~ nearest modeled ASP
                idx = (prod_curve["asp"] - curr_price).abs().idxmin()
                rev_curr = float(prod_curve.loc[idx, "revenue_pred_0.5"])

                # recommended (from best50 rows)
                rec = best50_df.loc[best50_df["product"] == p]
                if rec.empty or pd.isna(rec["revenue_pred_0.5"].iloc[0]):
                    continue
                rev_best = float(rec["revenue_pred_0.5"].iloc[0])

                # elasticity (optional)
                e = elast_df.loc[elast_df["product"] == p, "ratio"]
                elast_val = float(e.iloc[0]) if len(e) and pd.notna(e.iloc[0]) else None

                rows.append(
                    {
                        "product": p,
                        "upside": rev_best - rev_curr,
                        "elasticity": elast_val,
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return self.empty_fig("No computable upside")

        # Top N largest upside; horizontal bar for readability
        df = df.sort_values("upside", ascending=True).tail(12)

        # plotting
        BEIGE = "#EDD4B7"
        # GOLD = "#DAA520"
        fig = px.bar(
            df,
            y="product",
            x="upside",
            hover_data=["elasticity"],
            height=420,
            template=self.template,
            color_discrete_sequence=[BEIGE],  # set bar color
        )
        fig.update_xaxes(
            title_text="Daily Expected Revenue Δ",
            tickprefix="$",
            separatethousands=True,
        )
        fig.update_yaxes(title_text="")

        # keep bars gold even if the template tries to restyle traces
        fig.update_traces(
            marker_color=BEIGE,
            text=df["upside"].map(lambda x: f"${x:,.0f}"),
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=60),
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )
        return fig

    def empty_fig(self, title="No data"):
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template=self.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=60, b=40),
        )
        fig.add_annotation(
            text=title, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper"
        )
        return fig
