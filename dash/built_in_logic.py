"""
oct 13
-
-------

1. need to be more thoughtful about the occurrence of "infrequent" prices
2. inelasticity is not being scaled enough (case of unscented 16lb)
3. treat promo vs BAU differently? promo data is more uncommon. so skip aggregation?
"""

# --------- built_in_logic.py  ---------
# (RMSE-focused; Top-N only; adds annualized opps & data range)
from __future__ import annotations
from typing import Iterable, Dict, Optional
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

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

    # def _rarity_multiplier(self, df: pd.DataFrame, *, col: str) -> np.ndarray:
    #     n = len(df)
    #     if col not in df.columns or n == 0:
    #         return np.ones(n, dtype=float)

    #     counts = df.groupby(col)[col].transform("count").to_numpy()
    #     if np.any(counts > 0):
    #         dens_ref = float(np.nanmedian(counts[counts > 0]))
    #         if not (np.isfinite(dens_ref) and dens_ref > 0):
    #             dens_ref = 1.0
    #     else:
    #         dens_ref = 1.0

    #     counts = np.where(np.isfinite(counts) & (counts > 0), counts, dens_ref)
    #     rarity_raw = dens_ref / counts  # higher when sparser

    #     g = float(self.rarity_gamma)
    #     rarity = 1.0 + g * (rarity_raw - 1.0) if g != 1.0 else rarity_raw

    #     lo, hi = self.rarity_bounds
    #     rarity = np.clip(rarity, float(lo), float(hi))
    #     rarity = np.nan_to_num(rarity, nan=1.0, posinf=hi, neginf=lo)
    #     rarity[rarity <= 0] = 1e-6
    #     return rarity.astype(float, copy=False)


class _TuneResult:
    """Container for best hyperparameter results and trained models."""

    def __init__(
        self,
        n_splines=None,
        lam=None,
        score=None,
        decay_rate=None,
        final_score=None,
        models=None,
        # new fields (default to None so existing call sites still work)
        rmse_val=None,
        interval_score=None,
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.score = score
        self.decay_rate = decay_rate
        self.final_score = final_score
        self.models = models or {}
        # initialize these so they're always present
        self.rmse_val = rmse_val
        self.interval_score = interval_score

    def to_dict(self):
        """optional, nice for debugging)"""
        return {
            "n_splines": self.n_splines,
            "lam": self.lam,
            "score": self.score,
            "final_score": self.final_score,
            "rmse_val": self.rmse_val,
            "interval_score": self.interval_score,
        }


class ParamSearchCV:
    """
    Handles training and returning trained models + metrics.

    Joint expectile tuning for (0.025, 0.5, 0.975) using interval score + optional width penalty.
    Backward-compatible signature: keeps numeric/categorical preprocessing args, n_splits, lam_iters, weighting.
    """

    def __init__(
        self,
        n_splines_grid=sorted(set([14, 16, 18, 21, 24, 26])),
        loglam_range=(np.log(0.005), np.log(1.0)),
        n_lam=(5, 7, 9, 11, 13, 15, 17, 19, 21),  # int (fixed) OR iterable
        expectiles=(0.025, 0.5, 0.975),
        alpha=0.10,
        random_state=42,
        verbose=False,
    ):
        self.n_splines_grid = n_splines_grid
        self.loglam_range = loglam_range
        self.expectiles = expectiles
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

        # normalize n_lam: int -> fixed; iterable -> candidates (tuned like other hypers)
        if isinstance(n_lam, (list, tuple)):
            self.n_lam = None
            self.n_lam_candidates = tuple(int(x) for x in n_lam)
        else:
            self.n_lam = int(n_lam)
            self.n_lam_candidates = None

    def _default_grids(self, n_rows: int = 1000, n_unique_prices: int = 10):
        """Return sensible default grids based on dataset scale."""
        if n_unique_prices <= 6:
            ns_grid = (12, 15, 18)
        elif n_unique_prices <= 10:
            ns_grid = (15, 20, 25)
        else:
            ns_grid = (18, 22, 26)

        if n_rows < 400:
            lam_low, lam_high = 0.02, 5.0
        elif n_rows < 1500:
            lam_low, lam_high = 0.02, 10.0
        else:
            lam_low, lam_high = 0.02, 20.0

        return dict(
            n_splines_grid=ns_grid,
            loglam_range=(np.log(lam_low), np.log(lam_high)),
            n_lam=7,
            expectiles=(0.025, 0.5, 0.975),
            alpha=0.10,
            random_state=42,
            verbose=False,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _fit_one(self, e, ns, lam, X, y, w=None):

        # --- Ensure X is DataFrame ---
        if isinstance(X, np.ndarray):
            # recover column names safely
            candidate_cols = []
            if getattr(self, "numeric_cols", None):
                candidate_cols += list(self.numeric_cols)
            if getattr(self, "categorical_cols", None):
                candidate_cols += list(self.categorical_cols)

            if not candidate_cols or len(candidate_cols) != X.shape[1]:
                candidate_cols = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=candidate_cols)

        # --- Identify columns dynamically ---
        if getattr(self, "numeric_cols", None):
            num_cols = [c for c in self.numeric_cols if c in X.columns]
        else:
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        if getattr(self, "categorical_cols", None):
            cat_cols = [c for c in self.categorical_cols if c in X.columns]
        else:
            cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

        # --- Normalize numeric features safely ---
        if num_cols:
            X[num_cols] = StandardScaler().fit_transform(X[num_cols])

        # --- Build GAM terms ---
        terms = None
        if num_cols:
            terms = s(0, n_splines=ns, basis="ps")
            for i in range(1, len(num_cols)):
                terms += s(i, n_splines=ns, basis="ps")

        if cat_cols:
            for j in range(len(cat_cols)):
                terms = terms + f(len(num_cols) + j) if terms else f(j)

        # --- Fit model ---
        gam = ExpectileGAM(expectile=e, lam=lam, terms=terms, max_iter=1000, tol=1e-4)
        gam.fit(X, y, weights=w)

        self._log(
            f"[DBG FitOne] e={e}, n_splines={ns}, lam={lam}, features={list(X.columns)}"
        )
        return gam

    def _tune_decay(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        w_train=None,
        w_val=None,
        *,
        base_ns=None,
        base_lam=None,
        expectiles=None,
    ):
        """Adaptive tuning of decay rate with gradient-like steps."""
        start = time.time()
        print("🌊 Fine-tuning time-decay rate adaptively...")

        W = Weighting()
        search_range = np.linspace(-0.3, -0.05, 10)
        best_decay, best_score = 0, float("inf")

        for decay in search_range:
            w_train_d = np.asarray(
                W._make_weights(X_train, decay_rate=decay), dtype=float
            )
            w_val_d = np.asarray(W._make_weights(X_val, decay_rate=decay), dtype=float)

            preds_val = []
            for e in expectiles or self.expectiles:
                model = ExpectileGAM(expectile=e, n_splines=base_ns, lam=base_lam)
                model.fit(X_train, y_train, weights=w_train_d)
                preds_val.append(model.predict(X_val))

            preds_val = np.vstack(preds_val).T
            rmse_val = np.sqrt(np.mean((y_val - preds_val[:, 1]) ** 2))
            if rmse_val < best_score:
                best_score = rmse_val
                best_decay = decay
            print(f"   → decay={decay:+.3f} | RMSE={rmse_val:6.2f}")

        print(
            f"🌟 Optimal decay rate ≈ {best_decay:+.3f} (score={best_score:.2f}) after {time.time() - start:.1f}s"
        )
        return best_decay, best_score

    def _nlam_candidate_score(self, metrics: dict) -> float:
        # Prefer your composite if present; otherwise fall back.
        if not isinstance(metrics, dict):
            return np.inf
        for k in ("final_score", "rmse_val", "interval_score"):
            if k in metrics and np.isfinite(metrics[k]):
                return float(metrics[k])
        return np.inf

    def fit_with_nlam_candidates(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        w_train=None,
        w_val=None,
        n_lam_candidates=(5, 7, 9, 11, 13),
    ):
        """
        Try several n_lam values; pick the best based on your existing fit() metrics.
        Does NOT change your original fit() implementation.
        After this returns:
            - self.best_ and self.best_metrics_ refer to the winning run
            - self.best_n_lam holds the winning n_lam
        Returns a dict mirroring your fit() return (best object + metrics).
        """
        best_obj = None
        best_metrics = None
        best_score = np.inf
        best_nlam = None

        # Snapshot current n_lam so we can restore it
        original_nlam = getattr(self, "n_lam", None)

        for nlam in n_lam_candidates:
            # make a shallow clone of this searcher with the same params but a different n_lam
            ps = ParamSearchCV(
                n_splines_grid=self.n_splines_grid,
                loglam_range=self.loglam_range,
                n_lam=int(nlam),
                expectiles=self.expectiles,
                alpha=self.alpha,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            res = ps.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                w_train=w_train,
                w_val=w_val,
            )

            cur_best = getattr(ps, "best_", None) or getattr(res, "best_", None)
            cur_mets = getattr(ps, "best_metrics_", None) or getattr(
                res, "best_metrics_", None
            )
            cur_score = self._nlam_candidate_score(cur_mets)

            # if self.verbose:
            #     print(
            #         f"🔎 n_lam={nlam} → score={cur_score:.6f}, "
            #         f"ns={getattr(cur_best,'n_splines',None)}, λ={getattr(cur_best,'lam',None)}"
            #     )

            if cur_score < best_score:
                best_score = cur_score
                best_obj = cur_best
                best_metrics = cur_mets
                best_nlam = int(nlam)

        # Publish the winner on THIS instance for downstream code to read
        self.best_ = best_obj
        self.best_metrics_ = best_metrics
        self.best_n_lam = best_nlam
        if original_nlam is not None:
            self.n_lam = original_nlam  # restore

        return {
            "best_": best_obj,
            "best_metrics_": best_metrics,
            "best_n_lam": best_nlam,
        }

    def interval_score(
        self, y_true, y_low, y_high, y_pred=None, alpha=0.05, width_penalty=0.1
    ):
        """
        Interval score for (1-alpha) predictive intervals.

        Parameters
            y_true: Actual vals [array]
            y_low: Lower bound of predictive interval [array]
            y_high: Upper bound of predictive interval [array]
            y_pred: Point prediction (median or mean). Used for tie-breaking [array, optional]
            alpha: Miscoverage rate (0.05 = 95% interval) [float]
            width_penalty: Additional penalty on interval width to discourage overly wide bands. [float]

        Returns
            score : Lower is better [float]
        """
        y_true = np.asarray(y_true)
        y_low = np.asarray(y_low)
        y_high = np.asarray(y_high)

        # Interval width
        width = np.maximum(y_high - y_low, 0)

        # Penalty: points outside the interval
        under = (y_true < y_low).astype(float)
        over = (y_true > y_high).astype(float)

        penalty = (2 / alpha) * ((y_low - y_true) * under + (y_true - y_high) * over)

        score = np.mean(width + penalty)

        if y_pred is not None:
            # small tie-breaker: prefer models with lower point RMSE
            rmse = root_mean_squared_error(y_true, y_pred)
            score += width_penalty * rmse

        return float(score)

    # =============================
    #   λ & spline setup helpers
    # =============================
    def _init_lambda_grid(self, X_train):
        """
        Wide, strictly-positive, log-spaced λ grid centered on a data scale.
        n_lam, lam_span_decades, and lam_min auto-tune from data unless explicitly set on self.
        """
        n_obs = len(X_train)
        x = X_train[:, 0] if X_train.ndim == 2 else X_train

        # Rough data support
        n_bins = int(np.clip(np.sqrt(max(n_obs, 1)), 10, 50))
        hist, _ = np.histogram(x, bins=n_bins)
        eff_bins = int(np.sum(hist > 0))
        support_ratio = eff_bins / max(n_bins, 1)  # 0..1

        # Center scale (same spirit as your original)
        lam_base = 0.05 * (40 / max(eff_bins, 8)) * (800 / max(n_obs, 200))

        # -------------------- AUTO SETTINGS (overridden if attrs exist) --------------------
        # lam_min: higher floor when support is sparse, lower floor when rich
        lam_min_auto = float(
            10.0 ** np.interp(support_ratio, [0.0, 1.0], [-3.0, -4.0])
        )  # ~1e-3 .. 1e-4
        lam_min = float(getattr(self, "lam_min", lam_min_auto))

        # span_decades: explore wider when support is sparse, narrower when rich
        span_decades_auto = float(np.interp(support_ratio, [0.0, 1.0], [3.5, 2.5]))
        span_decades = float(getattr(self, "lam_span_decades", span_decades_auto))

        # n_lam: density per decade grows with sample size, then total = density * span
        points_per_decade = float(
            np.interp(min(max(n_obs, 1), 2000), [50, 200, 1000, 2000], [5, 6, 8, 9])
        )
        n_lam_auto = int(max(9, int(np.ceil(points_per_decade * span_decades))))
        n_lam = int(max(getattr(self, "n_lam", n_lam_auto), 9))

        # Hard ceiling stays generous
        lam_max = float(getattr(self, "lam_max", 1e3))
        # -----------------------------------------------------------------------------

        # Compute low/high around lam_base
        low = max(lam_min, lam_base * (10.0 ** (-span_decades)))
        high = min(lam_max, lam_base * (10.0 ** (+span_decades)))

        # Fallback if lam_base degenerates
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low, high = lam_min, lam_max

        # Final, strictly-positive, log-spaced grid
        lam_grid = np.exp(np.linspace(np.log(low), np.log(high), num=n_lam))
        lam_grid = lam_grid[np.isfinite(lam_grid)]
        lam_grid = lam_grid[lam_grid > 0.0]
        lam_grid = np.clip(lam_grid, lam_min, lam_max)

        lam_floor = float(
            lam_grid.min()
        )  # single source of truth; _fit_one_cycle must not re-floor

        self._log(
            "[DBG λ-init] "
            f"n_obs={n_obs} eff_bins={eff_bins} support={support_ratio:.3f} base={lam_base:.4g} | "
            f"lam_min={lam_min:.3g} span={span_decades:.2f} n_lam={n_lam} | "
            f"grid_min={lam_grid.min():.4g} grid_max={lam_grid.max():.4g}"
        )
        return lam_grid, lam_floor

    def _init_spline_grid(self, X_train):
        # derive support
        n_obs = len(X_train)
        x = X_train[:, 0] if X_train.ndim == 2 else X_train
        n_bins = int(np.clip(np.sqrt(max(n_obs, 1)), 10, 50))
        hist, _ = np.histogram(x, bins=n_bins)
        eff_bins = int(np.sum(hist > 0))

        # cap: ~40% of occupied bins + small cushion, and 12% of sample size
        cap = max(6, min(int(0.40 * eff_bins) + 5, int(0.12 * n_obs)))

        # center near a reasonable default but never exceed cap
        base = min(21, cap)

        # tight, symmetric grid around base within [6, cap]
        grid = sorted(
            set(
                [
                    max(6, base - 6),
                    max(6, base - 3),
                    base,
                    min(cap, base + 3),
                    min(cap, base + 6),
                ]
            )
        )
        return grid

    def _maybe_scale_val(self, X, ref_model):
        """
        Build the validation design *in the shape the model expects*.
        If the model was fit on 1 feature (default), return a (n,1) array with scaled col0.
        Otherwise, return a copy of X with only col0 scaled.
        """
        Xv = np.asarray(X)
        n_feats = int(getattr(ref_model, "n_features_", 1))

        # extract column 0
        col0 = Xv[:, 0] if Xv.ndim == 2 else Xv
        xmin = getattr(ref_model, "_x_min", None)
        xsc = getattr(ref_model, "_x_scale", None)
        if (
            getattr(ref_model, "_scaled_first_col", False)
            and xmin is not None
            and xsc not in (None, 0, 0.0)
        ):
            z = (col0.astype(float) - float(xmin)) / float(xsc)
        else:
            z = col0.astype(float)

        if n_feats == 1:
            return z.reshape(-1, 1)  # model expects a single feature
        else:
            Xcopy = np.array(Xv, copy=True)
            Xcopy[:, 0] = z
            return Xcopy

    # =============================
    #   candidate evaluation
    # =============================
    def _evaluate_candidate(
        self, X_train, y_train, X_val, y_val, w_train, w_val, ns, lam
    ):
        """
        Scorer used by the search. Returns (final_score, base_rmse, penalty, models, k_eff)
        so the outer loop can carry forward the exact models that determined the score.
        """
        models = {}
        for e in self.expectiles:
            res = self._fit_one_cycle(
                X_train,
                y_train,
                X_val,
                y_val,
                w_train=w_train,
                w_val=w_val,
                n_splines=ns,
                lam=lam,
                expectile=e,
            )
            models[e] = res["model"]

        # need all three bounds to score
        needed = {min(self.expectiles), 0.5, max(self.expectiles)}
        if any(models.get(e) is None for e in needed):
            return float("inf"), float("inf"), float("inf"), models, int(ns)

        # scale X_val the same way the ref model (prefer p50) was trained
        ref = models.get(0.5) or next(m for m in models.values() if m is not None)
        Xv = self._maybe_scale_val(X_val, ref)

        yhat_p50 = np.maximum(models[0.5].predict(Xv), 0.0)
        yhat_lo = np.maximum(models[min(self.expectiles)].predict(Xv), 0.0)
        yhat_hi = np.maximum(models[max(self.expectiles)].predict(Xv), 0.0)

        eps = 1e-6
        rmse_log = float(
            np.sqrt(np.mean((np.log(y_val + eps) - np.log(yhat_p50 + eps)) ** 2))
        )
        width_med = float(np.median(np.log(yhat_hi + eps) - np.log(yhat_lo + eps)))

        width_weight = 0.12
        final_score = rmse_log + width_weight * width_med

        # k_eff (robust to list/tuple/ndarray)
        try:
            ns_attr = getattr(ref, "n_splines", ns)
            if isinstance(ns_attr, (list, tuple, np.ndarray)):
                k_eff = int(sum(int(x) for x in ns_attr))
            else:
                k_eff = int(ns_attr)
        except Exception as e:
            self._log(
                f"[WARN] k_eff extract failed ({type(e).__name__}: {e}); using ns={ns}"
            )
            k_eff = int(ns)

        return final_score, rmse_log, (width_weight * width_med), models, k_eff

    # =============================
    #   adaptive refresh logic
    # =============================
    def _refresh_lambda_grid(self, lam_best, lam_grid, lam_floor):
        left = lam_best / np.array([1.5, 1.25, 1.1], dtype=float)
        center = lam_best * np.array([1.0], dtype=float)
        right = lam_best * np.array([1.1, 1.25, 1.5], dtype=float)

        lam_next = np.concatenate([left, center, right])
        lam_next = lam_next[np.isfinite(lam_next)]
        lam_next = lam_next[lam_next > 0.0]
        lam_next = np.unique(np.clip(lam_next, lam_floor, np.inf))

        target_n = max(self.n_lam, 9)
        if lam_next.size > target_n:
            qs = np.linspace(0.0, 1.0, num=target_n)
            lam_grid_new = np.exp(np.quantile(np.log(lam_next), qs))
        else:
            lam_grid_new = lam_next

        lam_grid_new = np.clip(lam_grid_new, lam_floor, np.inf)
        if lam_grid_new.min() <= 1.001 * lam_floor:
            self._log(
                f"[DBG λ-refresh] hugging floor: min={lam_grid_new.min():.4g} floor={lam_floor:.4g}"
            )
        return lam_grid_new

    def _update_spline_grid(
        self, best_ns, n_splines_grid, observed_round=None, rel_tol=None
    ):
        """
        Data-driven update for the n_splines grid.
        - Collapse duplicates caused by caps (many ns -> same k_eff).
        - Find elbow: smallest k_eff within (1+rel_tol) of best score.
        - Propose next grid centered on the elbow; expand if we are at a boundary.
        Falls back to a simple local contraction if observed_round is None.
        """
        # Backward-compat fallback
        if observed_round is None or not observed_round:
            # Contract around the best candidate as before
            uniq = sorted(set(n_splines_grid + [best_ns]))
            idx = max(0, uniq.index(best_ns))
            window = sorted(set(uniq[max(0, idx - 1) : idx + 2]))
            return window if window else uniq

        # 1) Build k_eff -> best score map (collapse duplicates)
        # Some ns values are capped to the same k_eff; keep the best score for each k_eff
        best_by_keff = {}
        for rec in observed_round:
            ke = int(rec.get("k_eff", rec["n_splines"]))
            sc = float(rec["score"])
            cur = best_by_keff.get(ke)
            if cur is None or sc < cur:
                best_by_keff[ke] = sc

        if not best_by_keff:
            return n_splines_grid

        # 2) Sort by k_eff and find best + elbow
        pairs = sorted(best_by_keff.items())  # [(k_eff, score), ...] ascending k_eff
        k_list = [k for k, _ in pairs]
        s_list = [s for _, s in pairs]
        s_best = min(s_list)
        rel_tol = float(
            self.rel_tol
            if hasattr(self, "rel_tol")
            else (rel_tol if rel_tol is not None else 1e-3)
        )
        tol = rel_tol * max(1.0, abs(s_best))

        # Elbow = smallest k_eff whose score within tol of s_best
        elbow_keff = next(k for k, s in pairs if (s - s_best) <= tol)

        # 3) Propose next grid around elbow, with gentle exploration
        # include a few neighbors in k_eff space (±2, ±4), mapped to ns suggestions
        neighbors = sorted(
            set(
                [
                    max(6, elbow_keff - 4),
                    max(6, elbow_keff - 2),
                    elbow_keff,
                    elbow_keff + 2,
                    elbow_keff + 4,
                ]
            )
        )

        # If the elbow is at boundary of observed k_eff, expand outward for exploration
        if elbow_keff == k_list[0]:
            neighbors = sorted(
                set([max(6, elbow_keff - 6)] + neighbors + [elbow_keff + 6])
            )
        if elbow_keff == k_list[-1]:
            neighbors = sorted(
                set([max(6, elbow_keff - 6)] + neighbors + [elbow_keff + 6])
            )

        # 4) Convert proposed k_eff targets into ns candidates.
        # Since k_eff = min(ns, cap), using ns == proposed k_eff is sufficient.
        ns_next = sorted(set(neighbors))

        # Keep grid small and ordered
        return ns_next[:7]  # cap size to keep search fast

    def _make_lam_grid_like(self, current_grid, nlam):
        """Preserve current log-range; change only the resolution (nlam)."""
        loglo, loghi = np.log(current_grid[0]), np.log(current_grid[-1])
        return np.exp(np.linspace(loglo, loghi, int(nlam)))

    # =============================
    #   main orchestration
    # =============================
    def fit(self, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
        """ """
        # print the banner ONCE per process
        if not getattr(self.__class__, "_banner_printed", False):
            print(
                "\n"
                + " " * 51
                + "─" * 15
                + " ⚙️  Adaptive Hyperparameter Search "
                + "-" * 15
                + "\n"
            )
            self.__class__._banner_printed = True

        start = time.time()

        patience = getattr(self, "patience", 25)  
        max_steps = getattr(self, "max_steps", 30)

        rel_tol = getattr(self, "rel_tol", 5e-4)

        n_splines_grid = self._init_spline_grid(X_train)
        lam_grid, lam_floor = self._init_lambda_grid(X_train)

        # global cap
        price_tr = X_train[:, 0] if X_train.ndim == 2 else X_train
        n_obs = len(price_tr)
        n_bins = int(np.clip(np.sqrt(max(n_obs, 1)), 10, 50))
        hist, _ = np.histogram(price_tr, bins=n_bins)
        eff_bins = int(np.sum(hist > 0))
        ns_cap = max(6, min(int(0.40 * eff_bins) + 5, int(0.12 * n_obs)))
        n_splines_grid = sorted({min(int(ns), ns_cap) for ns in n_splines_grid})

        self._log(
            f"[DBG ns-cap] n_obs={n_obs} eff_bins={eff_bins} ns_cap={ns_cap} init_grid={n_splines_grid}"
        )

        # handle n_lam: single int or iterable of ints (candidates)
        if hasattr(self, "n_lam_candidates") and self.n_lam_candidates:
            nlam_list = tuple(int(x) for x in self.n_lam_candidates)
        else:
            # if __init__ was updated to overload n_lam, this stays robust;
            # otherwise self.n_lam is already an int
            nlam_list = (int(getattr(self, "n_lam", 6)),)

        print(f"   🔧 Tuning n_lam candidates: {tuple(int(x) for x in nlam_list)}")

        # tuning
        best_score, best_result = float("inf"), None
        no_improve, step = 0, 0

        while True:
            print(f"🔍 Round {step+1}:") 

            round_scores = []
            round_best_nlam = None

            # try one or more n_lam resolutions this round
            for _nlam in nlam_list:
                lam_grid_this = self._make_lam_grid_like(lam_grid, _nlam)

                print(f"   🧭 Tuning λ with n_lam={int(_nlam)} | λ candidates (n={len(lam_grid_this)}): "
                    f"{[round(float(v), 6) for v in lam_grid_this]}")
                
                scores = []
                for ns in n_splines_grid:
                    for lam in lam_grid_this:
                        score_with_penalty, mean_score, penalty, models, k_eff = (
                            self._evaluate_candidate(
                                X_train, y_train, X_val, y_val, w_train, w_val, ns, lam
                            )
                        )
                        scores.append(
                            {
                                "n_splines": ns,
                                "k_eff": int(k_eff),
                                "lam": lam,
                                "score": score_with_penalty,
                                "base_score": mean_score,
                                "penalty": penalty,
                                "models": models,
                                "n_lam": int(_nlam),
                            }
                        )

                scores.sort(key=lambda x: x["score"])
                best_local = scores[0]
                round_scores.extend(scores)

                # keep the best across n_lam candidates for this round
                if (not round_best_nlam) or (
                    best_local["score"]
                    < min(
                        s["score"]
                        for s in round_scores
                        if s["n_lam"] == round_best_nlam
                    )
                ):
                    round_best_nlam = int(best_local["n_lam"])
                    # stash the best_local chosen so far (we'll re-evaluate below anyway)
                    chosen_for_round = dict(best_local)

            # finalize the round's winner
            round_scores.sort(key=lambda x: x["score"])
            best_local = round_scores[0]  # overall best this round

            # report improvement
            if best_result is None:
                best_result = dict(best_local)
                best_score = best_local["score"]
                print(
                    f"   🌟 Baseline: n_splines={best_local['n_splines']}, λ={best_local['lam']:.4f} (score={best_score:.4f})"
                )
            else:
                improved = (best_score - best_local["score"]) > rel_tol * max(
                    1.0, abs(best_score)
                )
                if improved:
                    best_result = dict(best_local)
                    best_score = best_local["score"]
                    no_improve = 0
                    print(
                        f"   🌟 New best: n_splines={best_local['n_splines']}, λ={best_local['lam']:.4f} (score={best_score:.4f})"
                    )
                else:
                    no_improve += 1
                    print(f"   💤 No improvement (streak {no_improve}/{patience})")

            # small breadcrumb for which n_lam won this round
            if "n_lam" in best_local:
                print(f"   (chosen n_lam this round = {int(best_local['n_lam'])})")

            # refresh λ grid around the current best λ using the resolution that won this round
            # 1) rebuild lam_grid to the winning resolution, preserving current bounds
            lam_grid = self._make_lam_grid_like(
                lam_grid, int(best_local.get("n_lam", nlam_list[0]))
            )
            # 2) then let your existing refresher recentre/narrow as before
            lam_grid = self._refresh_lambda_grid(best_local["lam"], lam_grid, lam_floor)

            # update spline grid as before
            n_splines_grid = self._update_spline_grid(
                best_local["n_splines"],
                n_splines_grid,
                observed_round=round_scores,
            )
            n_splines_grid = sorted({min(int(ns), ns_cap) for ns in n_splines_grid})

            step += 1
            if no_improve >= patience or step >= max_steps:
                print("   🧯 Patience exhausted — stopping adaptive search.")
                break

        # best_result is the dict from the top of `round_scores` after sorting by "score"
        self.best_ = _TuneResult(
            n_splines=int(best_result["n_splines"]),
            lam=float(best_result["lam"]),
            score=float(best_result["score"]),
            final_score=float(best_result.get("score", float("nan"))),
            rmse_val=float(best_result.get("base_score", float("nan"))),
            interval_score=float(best_result.get("penalty", float("nan"))),
            models=dict(
                best_result.get("models", {})
            ),  # carry the winning models (no refit)
        )

        # Optional: log effective splines if present
        best_keff = int(best_result.get("k_eff", self.best_.n_splines))
        duration = time.time() - start
        print(
            f"🌟 Best config: n_splines={self.best_.n_splines} (k_eff={best_keff}), "
            f"λ={self.best_.lam:.4f} (score={self.best_.score:.4f}) after {duration:.1f}s\n"
        )
        return self.best_

    def _fit_one_cycle(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        w_train=None,
        w_val=None,
        n_splines=None,
        lam=None,
        expectile=0.5,
    ):
        """
        Fit one ExpectileGAM and return {"expectile", "model", "score"} for selection.
        """

        # -------------------- config knobs (single place to flip) --------------------
        USE_LOG_RMSE = True  # True: Huberized log-RMSE; False: Huberized linear-RMSE
        USE_EDF_PENALTY = (
            False  # keep False unless you explicitly want extra complexity push
        )

        # --- weight safety (train) ---
        if w_train is not None:
            w_train = np.asarray(w_train, dtype=float)
            w_train = np.nan_to_num(w_train, nan=1.0, posinf=3.0, neginf=1e-6)
            w_train[w_train <= 0] = 1e-6

        # --- feature scaling on the single feature used by GAM (col 0) ---
        x_tr = X_train[:, 0] if X_train.ndim == 2 else X_train
        x_val = X_val[:, 0] if X_val.ndim == 2 else X_val

        x_min = float(np.nanmin(x_tr))
        x_ptp = float(np.nanmax(x_tr) - x_min)
        scale = x_ptp if x_ptp > 0 else 1.0

        x_tr_s = (x_tr - x_min) / scale
        x_val_s = (x_val - x_min) / scale

        # pygam default model is 1 feature → make 2D with a single column
        Xtr_s = x_tr_s.reshape(-1, 1)
        Xval_s = x_val_s.reshape(-1, 1)

        # --- support occupancy / cap computed on the same scaled feature ---
        n_obs = len(Xtr_s)
        n_bins = int(np.clip(np.sqrt(max(n_obs, 1)), 10, 50))
        hist, _ = np.histogram(Xtr_s[:, 0], bins=n_bins)
        eff_bins = int(np.sum(hist > 0))

        n_splines_cap = max(6, min(int(eff_bins // 3 + 5), int(0.10 * n_obs)))
        k_eff = int(min(int(n_splines or 20), n_splines_cap))

        lam_eff = float(lam if lam is not None else 1e-6)

        # --- fit & predict on the SAME 1-D representation ---
        model = ExpectileGAM(expectile=float(expectile), n_splines=k_eff, lam=lam_eff)
        try:
            model.fit(Xtr_s, y_train, weights=w_train)
        except Exception as e:
            if isinstance(e, np.linalg.LinAlgError) or "SVD did not converge" in str(e):
                return {"expectile": expectile, "model": None, "score": np.inf}
            raise

        # stash scaling so evaluator can reproduce the exact input representation
        model._scaled_first_col = True
        model._x_min = x_min
        model._x_scale = scale
        # pygam sets n_features_ during fit; it will be 1 here

        preds_val = np.maximum(model.predict(Xval_s), 0.0)

        # --- Huberized RMSE with fold-driven scale (automatic) ---
        eps = 1e-6
        if USE_LOG_RMSE:
            r = np.log(y_val + eps) - np.log(preds_val + eps)
        else:
            r = y_val - preds_val

        # robust scale via MAD (no hand tuning)
        mad = np.median(np.abs(r - np.median(r))) + eps
        sigma_hat = 1.4826 * mad
        c = 1.345 * sigma_hat

        abs_r = np.abs(r)
        w = np.where(abs_r <= c, 1.0, c / (abs_r + eps))
        rmse = float(np.sqrt(np.mean((w * r) ** 2)))

        # --- optional EDf-based overflex penalty (off by default) ---
        if USE_EDF_PENALTY:
            try:
                edf = getattr(model.statistics_, "edf", None)
                if edf is not None and np.isfinite(edf):
                    flex_ratio = float(edf) / max(model.n_splines, 1)
                    # mild, smooth bump beyond ~0.9 of max effective flex
                    over = max(0.0, flex_ratio - 0.9)
                    rmse *= float(min(1.0 + 0.12 * (over**1.5), 1.25))
            except Exception:
                pass

        return {"expectile": expectile, "model": model, "score": rmse, "k_eff": k_eff}


class GAMModeler:
    """
    Thin orchestrator:
    - ensures 4-column schema
    - loops over expectiles and delegates to ParamSearchCV
    - transforms X_pred via param_search.transform() and predicts
    """

    EXPECTILES = (0.025, 0.5, 0.975)

    def __init__(
        self,
        feature_cols: Iterable[str] = ("price",),
        base_gam_kwargs: Optional[dict] = None,
        per_expectile_kwargs: Optional[Dict[float, dict]] = None,
    ):
        """
        Args:
            feature_cols: columns used for X.
            base_gam_kwargs: default kwargs for ExpectileGAM (lam, n_splines, etc.).
            per_expectile_kwargs: dict mapping expectile -> kwargs to override base.
        """
        self.feature_cols = list(feature_cols)
        self.base_gam_kwargs = base_gam_kwargs or {}
        self.per_expectile_kwargs = per_expectile_kwargs or {}
        self.models: Dict[float, ExpectileGAM] = {}

    def _log(self, msg: str):
        """Timestamped console log for debugging."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def _to_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X[EXPECTED_COLS].copy()
        else:
            df = pd.DataFrame(np.asarray(X), columns=EXPECTED_COLS)
        for c in CAT_COLS:
            df[c] = df[c].astype("string")
        return df

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def _make_X(self, df_like: pd.DataFrame) -> np.ndarray:
        """
        Build X from df_like using self.feature_cols, enforcing 2D shape.
        """
        X = df_like[self.feature_cols].to_numpy()
        return self._ensure_2d(X)

    def _assert_feature_match(self, model: ExpectileGAM, X: np.ndarray):
        """
        Guard that predict-time features match the model's fitted dimensionality.
        """
        m_expected = getattr(model, "statistics_", {}).get("m", None)
        if m_expected is not None and X.shape[1] != m_expected:
            raise ValueError(
                f"Feature mismatch: predict has {X.shape[1]} features, "
                f"but model expects {m_expected}."
            )

    def _sanitize_prediction_rows(
        self, df: pd.DataFrame, pred_cols: Iterable[str]
    ) -> pd.DataFrame:
        """
        Drop rows with NaN price or NaN in any prediction columns (if present).
        """
        keep = pd.Series(True, index=df.index)
        if "price" in df.columns:
            keep &= df["price"].notna()
        pred_cols = [c for c in pred_cols if c in df.columns]
        if pred_cols:
            keep &= df[pred_cols].notna().all(axis=1)
        return df.loc[keep].copy()

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train: pd.Series,
        *,
        expectiles: Iterable[float] = EXPECTILES,
        verbose: bool = False,
    ) -> "GAMModeler":
        """
        Fit ExpectileGAM models for each expectile in `expectiles`.
        Attaches feature metadata for safe predict later.
        """
        X_train = self._make_X(train_df)

        self.models = {}
        for q in expectiles:
            kw = dict(self.base_gam_kwargs)
            kw.update(self.per_expectile_kwargs.get(q, {}))
            kw["expectile"] = q

            if verbose:
                print(f"[GAMModeler] Fitting ExpectileGAM(q={q}) with kwargs={kw}")

            model = ExpectileGAM(**kw).fit(X_train, y_train)
            # remember feature metadata for safety at predict-time
            model.feature_cols_ = list(self.feature_cols)
            self.models[q] = model

        return self

    def predict_expectiles(
        self,
        df_like: pd.DataFrame,
        *,
        expectiles: Iterable[float] = EXPECTILES,
    ) -> Dict[float, np.ndarray]:
        """
        Predict expectile values for rows in df_like.
        Returns a dict: expectile -> np.ndarray
        """
        if not self.models:
            raise RuntimeError("GAMModeler has no fitted models. Call fit() first.")

        X_pred = self._make_X(df_like)

        out: Dict[float, np.ndarray] = {}
        for q in expectiles:
            model = self.models.get(q, None)
            if model is None:
                raise KeyError(f"No model fitted for expectile {q}.")
            self._assert_feature_match(model, X_pred)
            out[q] = model.predict(X_pred)
        return out

    def add_predictions_to_df(
        self,
        df_like: pd.DataFrame,
        *,
        write_units=True,
        write_revenue=True,
        price_col: str = "price",
        expectiles: Iterable[float] = EXPECTILES,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Adds units_pred_{q} columns (and revenue_pred_{q} if price exists) to a copy of df_like
        by default (set inplace=True to modify in place).
        """
        df = df_like if inplace else df_like.copy()

        preds = self.predict_expectiles(df, expectiles=expectiles)

        # Units predictions
        if write_units:
            for q, yhat in preds.items():
                df[f"units_pred_{q}"] = yhat

        # Revenue predictions if we have prices
        if write_revenue and (price_col in df.columns):
            for q in expectiles:
                units_col = f"units_pred_{q}"
                if units_col in df.columns:
                    df[f"revenue_pred_{q}"] = (
                        df[price_col].astype(float) * df[units_col]
                    )

        return df

    def sanitize_results_for_downstream(self, df_like: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience: clean a post-prediction frame before ranking/selection.
        Removes rows with NaN price/preds if present.
        """
        pred_cols = [
            "units_pred_0.025",
            "units_pred_0.5",
            "units_pred_0.975",
            "revenue_pred_0.025",
            "revenue_pred_0.5",
            "revenue_pred_0.975",
        ]
        return self._sanitize_prediction_rows(df_like, pred_cols)


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
        self.engineer = DataEngineer(pricing_df, product_df, top_n)
        self.pricing_df = pricing_df
        self.product_df = product_df

        # param search
        if param_search_kwargs is None:
            # derive adaptive defaults dynamically
            n_rows = len(pricing_df)
            n_unique_prices = (
                pricing_df["price"].nunique() if "price" in pricing_df else 10
            )
            temp = ParamSearchCV()
            param_search_kwargs = temp._default_grids(n_rows, n_unique_prices)

        self.param_search = ParamSearchCV(**param_search_kwargs)

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

    def _build_core_frames(self):
        """
        pipeline glue
        return
            dfs
        """
        print(
            "\n─────────────────────────────── 🧮 Starting Data Engineering ───────────────────────────────"
        )
        topsellers = self.engineer.prepare()

        # Keep BOTH BAU + promo
        if topsellers.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Ensure ASP and intercept
        if "asp" not in topsellers.columns and "price" in topsellers.columns:
            topsellers["asp"] = pd.to_numeric(topsellers["price"], errors="coerce")
        if "__intercept__" not in topsellers.columns:
            topsellers["__intercept__"] = 1.0

        print("✅ Data loaded & preprocessed. Proceeding to weight computation...")

        # -------------------------------------------- Weights --------------------------------------------
        W = Weighting()
        w_stable = W.build(topsellers)
        topsellers[WEIGHT_COL] = w_stable
        print(
            f"⚖️  Weights computed | median={np.nanmedian(w_stable):.3f} | p95={np.nanpercentile(w_stable,95):.3f}"
        )

        # --- Assemble features/target with strict alignment ---
        need_cols = FEAT_COLS + [TARGET_COL, WEIGHT_COL]
        ts = topsellers[need_cols].copy()

        # Ensure numerics (encoded cols should already be numeric; this is safety)
        for c in FEAT_COLS + [TARGET_COL, WEIGHT_COL]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

        # Drop rows that can’t be used
        ts = ts.dropna(subset=need_cols).reset_index(drop=True)

        # Build matrices (kept for debug/compat)
        X = ts[FEAT_COLS].to_numpy(dtype=float)
        y = ts[TARGET_COL].to_numpy(dtype=float)
        w = ts[WEIGHT_COL].to_numpy(dtype=float)

        # Safety for downstream learners (no NaN/Inf/≤0 in weights)
        w = np.nan_to_num(w, nan=1.0, posinf=3.0, neginf=1e-6)
        w[w <= 0] = 1e-6

        # Lightweight weight summary (debug)
        w_med = float(np.nanmedian(w))
        w_p95 = float(np.nanpercentile(w, 95))
        print(f"⚖️  Weights ready | median={w_med:.3f} | p95={w_p95:.3f} | n={len(w):,}")

        # --------------- Elasticity diagnostic (best-effort) ---------------
        try:
            elasticity_df = ElasticityAnalyzer.compute(topsellers)
        except Exception:
            elasticity_df = pd.DataFrame(
                columns=["product", "ratio", "elasticity_score"]
            )

        # ---- NUMERIC STABILIZATION (prevents ill-conditioned SVD) ----
        _cont = [
            c
            for c in ["price", "deal_discount_percent", "year", "month", "week"]
            if c in FEAT_COLS
        ]
        idx = [FEAT_COLS.index(c) for c in _cont]
        if idx:
            Z = X[:, idx]
            mu = np.nanmean(Z, axis=0)
            sd = np.nanstd(Z, axis=0)
            sd[sd == 0] = 1.0
            X[:, idx] = (Z - mu) / sd

        # ──────────────────────────────
        # 🤖 Tuning with ParamSearchCV
        # ──────────────────────────────
        print("\n\n" + 35 * "- " + " 🤖 Modeling, Tuning & Prediction " + "- " * 35)

        if "price" not in ts.columns:
            raise ValueError(
                "Expected 'price' column to exist for univariate GAM tuning."
            )

        X_price = ts[["price"]].to_numpy(dtype=float)
        y_all = y.copy()
        w_all = w.copy()

        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X_price, y_all, w_all, test_size=0.20, random_state=42
        )

        w_tr = np.clip(
            np.nan_to_num(w_tr, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None
        )
        w_val = np.clip(
            np.nan_to_num(w_val, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None
        )

        param_search = ParamSearchCV(
            n_splines_grid=sorted({9, 11, 13, 16, 20, 24, 30, 38}),
            loglam_range=(np.log(1e-6), np.log(5.0)),
            n_lam=(5, 7, 9, 11, 13, 15, 17, 19, 21),
            expectiles=(0.025, 0.5, 0.975),
            alpha=0.10,
            random_state=42,
            verbose=True,
        )

        # try several n_lam values inside ParamSearchCV
        param_search.fit_with_nlam_candidates(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            w_train=w_tr,
            w_val=w_val,
            n_lam_candidates=(5, 7, 9, 11, 13),
        )

        # read results from the instance
        best = getattr(param_search, "best_", None)
        best_lam = float(
            getattr(best, "lam", np.exp(np.mean(param_search.loglam_range)))
        )
        best_ns = int(
            getattr(
                best,
                "n_splines",
                sorted(param_search.n_splines_grid)[
                    len(param_search.n_splines_grid) // 2
                ],
            )
        )
        metrics = getattr(param_search, "best_metrics_", None)

        if metrics is not None:
            print(f"    Validation metrics: {metrics}")

        # Use the EXACT tuned params downstream (no drift)
        modeler = GAMModeler(
            feature_cols=["price"],
            base_gam_kwargs={"lam": best_lam, "n_splines": best_ns},
        )

        # Fit on ALL data (using the aligned DataFrame for feature selection)
        modeler.fit(
            train_df=ts[modeler.feature_cols], y_train=ts[TARGET_COL], verbose=True
        )

        print(
            "\n"
            + " " * 51
            + "─" * 15
            + " ✅ Generating Prediction Frames "
            + "-" * 15
            + "\n"
        )

        # --------------- Results Assembly ---------------
        all_gam_results = (
            topsellers[["product", "price", "asin", "asp"]]
            .copy()
            .reset_index(drop=True)
        )

        # --- Local support: observed neighbors near each predicted price ---
        base_df = getattr(self, "pricing_df", None)
        if (
            (base_df is None)
            and hasattr(self, "engineer")
            and hasattr(self.engineer, "pricing_df")
        ):
            base_df = self.engineer.pricing_df

        all_gam_results["support_count"] = 0
        for prod_key in all_gam_results["product"].dropna().unique():
            g_pred = all_gam_results.loc[all_gam_results["product"] == prod_key]
            if base_df is not None and {"product", "price"}.issubset(base_df.columns):
                obs_prices = base_df.loc[
                    base_df["product"].astype(str) == str(prod_key), "price"
                ].to_numpy()
                if obs_prices.size == 0:
                    obs_prices = g_pred["price"].to_numpy()
            else:
                obs_prices = g_pred["price"].to_numpy()

            u = np.unique(np.sort(obs_prices))
            step = (
                np.nanmedian(np.diff(u))
                if u.size >= 2
                else max(0.25, np.nanstd(obs_prices) / 25.0)
            )
            win = 1.25 * step
            P_pred = g_pred["price"].to_numpy()
            supp = np.array(
                [(np.abs(obs_prices - p) <= win).sum() for p in P_pred], dtype=int
            )
            all_gam_results.loc[g_pred.index, "support_count"] = supp

        all_gam_results["support_count"] = (
            all_gam_results["support_count"].fillna(0).astype(int)
        )

        # ------------------------------------------------------------------------------------------
        # Add unit + revenue predictions using the tuned modeler
        all_gam_results = modeler.add_predictions_to_df(
            all_gam_results,
            write_units=True,
            write_revenue=True,
            price_col="price",
            inplace=False,
        )

        # Clamp negative unit predictions to 0 for safety (recompute revenue accordingly)
        for q in (0.025, 0.5, 0.975):
            ucol = f"units_pred_{q}"
            rcol = f"revenue_pred_{q}"
            if ucol in all_gam_results.columns:
                all_gam_results[ucol] = np.maximum(
                    all_gam_results[ucol].to_numpy(), 0.0
                )
                if rcol in all_gam_results.columns:
                    all_gam_results[rcol] = (
                        all_gam_results["price"].to_numpy()
                        * all_gam_results[ucol].to_numpy()
                    )

        # Sanity check
        pred_cols = [
            c
            for c in all_gam_results.columns
            if c.startswith(("units_pred_", "revenue_pred_"))
        ]
        if not pred_cols:
            raise ValueError(
                f"[_build_core_frames] Prediction assembly failed. DataFrame cols={list(all_gam_results.columns)}"
            )

        # deal discount passthrough + actual revenue
        all_gam_results["deal_discount_percent"] = (
            topsellers["deal_discount_percent"]
            .fillna(0)
            .clip(lower=0)
            .reset_index(drop=True)
        )
        all_gam_results["revenue_actual"] = topsellers["price"].to_numpy() * np.asarray(
            topsellers["shipped_units"], dtype=float
        )
        all_gam_results["daily_rev"] = all_gam_results["revenue_actual"]
        all_gam_results["actual_revenue_scaled"] = (
            topsellers["price"].to_numpy()
            * (1 - all_gam_results["deal_discount_percent"].to_numpy() / 100.0)
            * np.asarray(topsellers["shipped_units"], dtype=float)
        )

        # Optional: sanitize rows for downstream (drops NaN price/preds)
        all_gam_results = modeler.sanitize_results_for_downstream(all_gam_results)

        print(
            "\n"
            + 32 * "- "
            + " 🎯 Pipeline Complete at "
            + datetime.now().strftime("%H:%M:%S")
            + " "
            + 32 * "- "
            + "\n"
        )

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


if __name__ == "__main__":
    pricing_df, product_df = pd.read_csv("data/pricing.csv"), pd.read_csv(
        "data/products.csv"
    )

    PricingPipeline(
        pricing_df,
        product_df,
        top_n=10,
    ).assemble_dashboard_frames()


# all_gam_results = GAMModeler(
#     DataEngineer(pricing_df, product_df, top_n=10).prepare()).run()
# # Create a viz instance first, then call the method
# viz_instance = viz()
# viz_instance.gam_results(all_gam_results)
