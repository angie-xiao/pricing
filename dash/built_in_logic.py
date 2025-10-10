"""
1. Flat or near-zero curves
2. Explosive spikes
3. Misaligned uncertainty bands
4. Categorical signals ignored
"""

# --------- built_in_logic.py  ---------
# (RMSE-focused; Top-N only; adds annualized opps & data range)
from __future__ import annotations
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
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.score = score
        self.decay_rate = decay_rate
        self.final_score = final_score
        self.models = models or {}


class ParamSearchCV:
    """
    Handles training and returning trained models + metrics.

    Joint expectile tuning for (0.025, 0.5, 0.975) using interval score + optional width penalty.
    Backward-compatible signature: keeps numeric/categorical preprocessing args, n_splits, lam_iters, weighting.
    """

    def __init__(
        self,
        n_splines_grid=(15, 20, 25),
        loglam_range=(np.log(0.005), np.log(1.0)),
        n_lam=6,
        expectiles=(0.025, 0.5, 0.975),
        alpha=0.10,
        random_state=42,
        verbose=False,
    ):
        self.n_splines_grid = n_splines_grid
        self.loglam_range = loglam_range
        self.n_lam = n_lam
        self.expectiles = expectiles
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

    def _default_grids(self, n_rows: int = 1000, n_unique_prices: int = 10):
        """Return sensible default grids based on dataset scale."""
        if n_unique_prices <= 6:
            ns_grid = (12, 15, 18)
        elif n_unique_prices <= 10:
            ns_grid = (15, 20, 25)
        else:
            ns_grid = (18, 22, 26)

        if n_rows < 400:
            lam_low, lam_high = 0.006, 1.2
        elif n_rows < 1500:
            lam_low, lam_high = 0.005, 0.8
        else:
            lam_low, lam_high = 0.004, 0.5

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

    def fit(self, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
        """
        Adaptive grid search for (n_splines, λ) with directional expansion and patience.
        Explores both sides around best λ; stops only after several non-improving rounds.
        Robust to unstable candidates (SVD) by skipping np.inf scores.
        """
        print(
            "\n─────────────────────────────── ⚙️  Adaptive Hyperparameter Search ───────────────────────────────"
        )
        start = time.time()

        # derive defaults from data if using the class defaults
        if self.n_splines_grid in ((15, 20, 25),) and self.n_lam == 6:
            n_rows = len(X_train) + len(X_val)
            # infer unique price support if column exists; fallback if X is ndarray
            try:
                n_unique_prices = int(pd.concat([X_train, X_val])["price"].nunique())
            except Exception:
                n_unique_prices = 8
            ns_grid, loglam_range, n_lam = self._default_grids(n_rows, n_unique_prices)
            self.n_splines_grid = ns_grid
            self.loglam_range = loglam_range
            self.n_lam = n_lam
            print(
                f"   🧮 Grids → n_splines={self.n_splines_grid}, λ∈[e^{self.loglam_range[0]:.3f}, e^{self.loglam_range[1]:.3f}], n_lam={self.n_lam}"
            )

        patience = getattr(self, "patience", 2)
        max_steps = getattr(self, "max_steps", 8)
        rel_tol = getattr(self, "rel_tol", 1e-3)

        best_score, best_result = float("inf"), None
        no_improve = 0
        n_splines_grid = list(self.n_splines_grid)

        # λ × support scaling — enforce MORE smoothing when data are sparse
        n_support = max(1, len(X_train))
        lam_low, lam_high = np.exp(self.loglam_range)

        # small support -> lam_scale > 1 (stronger smoothing)
        lam_scale = np.clip(500.0 / n_support, 0.8, 2.0)
        lam_grid = np.geomspace(
            lam_low * lam_scale, lam_high * lam_scale, num=self.n_lam
        )

        step = 0
        last_scores = None  # keep for fallback
        retried_empty_round = False

        while True:
            scores = []
            print(f"🔍 Round {step+1}: λ grid = {[round(x,4) for x in lam_grid]}")
            for ns in n_splines_grid:
                for lam in lam_grid:
                    cycle_scores = []
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
                        cycle_scores.append(res["score"])
                    mean_score = float(np.mean(cycle_scores))
                    # skip unstable candidates (np.inf / NaN)
                    if not np.isfinite(mean_score):
                        continue
                    scores.append(
                        {"n_splines": int(ns), "lam": float(lam), "score": mean_score}
                    )

            # if an entire round produced no finite candidates, try a one-time wider λ sweep
            if not scores:
                if not retried_empty_round:
                    retried_empty_round = True
                    print(
                        "   ⚠ No finite candidates this round. Widening λ once and retrying..."
                    )
                    lam_min, lam_max = float(min(lam_grid)), float(max(lam_grid))
                    lam_grid = np.geomspace(
                        max(lam_min * 0.1, 1e-6),
                        max(lam_max * 10, lam_min * 10),
                        num=max(self.n_lam, 6),
                    )
                    continue
                else:
                    raise RuntimeError(
                        "ParamSearchCV.fit found no stable (n_splines, λ) candidates. Check features/weights."
                    )

            scores.sort(key=lambda x: x["score"])
            best_local = scores[0]
            last_scores = scores  # save for fallback

            # --- First round: always accept as baseline
            if best_result is None:
                best_result = dict(best_local)
                best_score = float(best_local["score"])
                no_improve = 0
                print(
                    f"   🌟 Baseline: n_splines={best_local['n_splines']}, λ={best_local['lam']:.4f} (score={best_score:.4f})"
                )
            else:
                # relative improvement check (finite-safe)
                denom = max(1.0, abs(best_score))
                improved = (best_score - best_local["score"]) > (rel_tol * denom)
                if improved:
                    best_result = dict(best_local)
                    best_score = float(best_local["score"])
                    no_improve = 0
                    print(
                        f"   🌟 New best: n_splines={best_local['n_splines']}, λ={best_local['lam']:.4f} (score={best_score:.4f})"
                    )
                else:
                    no_improve += 1
                    print(f"   💤 No improvement (streak {no_improve}/{patience})")

            # --- Build next λ grid: always explore both sides
            lam_best = float(best_local["lam"])
            lam_min, lam_max = float(min(lam_grid)), float(max(lam_grid))
            left = np.geomspace(
                max(lam_best * 1e-3, lam_min / 3),
                max(lam_best / 2, lam_min * 0.5),
                num=max(3, self.n_lam // 2),
            )
            center = np.geomspace(
                max(lam_best / 2, 1e-12), lam_best * 2, num=max(3, self.n_lam // 2)
            )
            right = np.geomspace(
                min(lam_best * 2, lam_max * 2),
                max(lam_max * 4, lam_best * 4),
                num=max(3, self.n_lam // 2),
            )
            lam_next = np.unique(
                np.clip(np.concatenate([left, center, right]), 1e-12, np.inf)
            )
            if lam_next.size > max(self.n_lam, 6):
                qs = np.linspace(0, 1, num=max(self.n_lam, 6))
                lam_grid = np.quantile(lam_next, qs)
            else:
                lam_grid = lam_next
            print(f"   🔁 Bi-directional λ refresh: {np.round(lam_grid, 6)}")

            # --- Adaptive n_splines refinement (keep your existing logic)
            ns_best = int(best_local["n_splines"])
            if ns_best == max(n_splines_grid):
                n_splines_grid = sorted(
                    set([max(5, ns_best - 5), ns_best, ns_best + 5, ns_best + 10])
                )
                print(f"   ↗ Expanding n_splines up: {n_splines_grid}")
            elif ns_best == min(n_splines_grid):
                n_splines_grid = sorted(
                    set(
                        [
                            max(5, ns_best - 10),
                            max(5, ns_best - 5),
                            ns_best,
                            ns_best + 5,
                        ]
                    )
                )
                print(f"   ↙ Expanding n_splines down: {n_splines_grid}")
            else:
                n_splines_grid = sorted(
                    set([max(5, ns_best - 5), ns_best, ns_best + 5])
                )
                print(f"   🔍 Contracting n_splines around {ns_best}: {n_splines_grid}")

            step += 1
            if no_improve >= patience:
                print("   🧯 Patience exhausted — stopping adaptive search.")
                break
            if step >= max_steps:
                print("   🧭 Max adaptive iterations reached.")
                break

        # --- Fallback if we ended with no accepted improvement beyond baseline
        if best_result is None and last_scores:
            best_result = dict(last_scores[0])
            best_score = float(last_scores[0]["score"])

        # --- Final safety: if still None, raise with guidance (avoid silent None subscript)
        if best_result is None or not np.isfinite(best_result["score"]):
            raise RuntimeError(
                "ParamSearchCV.fit failed to find any stable configuration. Check features/weights."
            )

        # --- Save and train final models
        self.best_ = _TuneResult(
            n_splines=int(best_result["n_splines"]),
            lam=float(best_result["lam"]),
            score=float(best_score),
        )

        final_models = {}
        for e in self.expectiles:
            res = self._fit_one_cycle(
                X_train,
                y_train,
                X_val,
                y_val,
                w_train=w_train,
                w_val=w_val,
                n_splines=self.best_.n_splines,
                lam=self.best_.lam,
                expectile=e,
            )
            if res["model"] is None or not np.isfinite(res["score"]):
                raise RuntimeError(
                    f"Final refit failed for expectile {e} with the chosen (n_splines, λ)."
                )
            final_models[e] = res["model"]
        self.best_.models = final_models

        duration = time.time() - start
        print(
            f"🌟 Best config: n_splines={self.best_.n_splines}, λ={self.best_.lam:.4f} (score={self.best_.score:.4f}) after {duration:.1f}s\n"
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
        """Fit a single ExpectileGAM and return validation score + model (robust)."""
        import numpy as np
        from pygam import ExpectileGAM

        # --- weight safety (train) ---
        if w_train is not None:
            w_train = np.asarray(w_train, dtype=float)
            w_train = np.nan_to_num(w_train, nan=1.0, posinf=3.0, neginf=1e-6)
            w_train[w_train <= 0] = 1e-6

        # --- stabilize k & lam ---
        # assume FEAT_COLS[0] == "price" -> price is X[:,0]
        uniq_price = int(np.unique(X_train[:, 0]).size) if X_train.size else 0
        k_eff = int(max(8, min(int(n_splines or 20), max(uniq_price - 1, 8))))
        lam_eff = float(max(float(lam if lam is not None else 0.1), 1e-4))

        model = ExpectileGAM(expectile=expectile, n_splines=k_eff, lam=lam_eff)

        # --- robust fit (skip unstable candidates) ---
        try:
            model.fit(X_train, y_train, weights=w_train)
        except Exception as e:
            # reject candidate cleanly on SVD/ill-conditioning
            if isinstance(e, np.linalg.LinAlgError) or "SVD did not converge" in str(e):
                return {"expectile": expectile, "model": None, "score": np.inf}
            raise

        # --- predict (clip at 0 per business rule) & score ---
        preds_val = model.predict(X_val)
        preds_val = np.maximum(preds_val, 0.0)

        rmse = float(np.sqrt(np.mean((y_val - preds_val) ** 2)))
        return {"expectile": expectile, "model": model, "score": rmse}

    def predict_expectiles(self, X):
        if self.best_ is None:
            raise RuntimeError("ParamSearchCV has not been fit yet.")
        return {k: m.predict(X) for k, m in self.best_.models.items()}


class GAMModeler:
    """
    Thin orchestrator:
    - ensures 4-column schema
    - loops over expectiles and delegates to ParamSearchCV
    - transforms X_pred via param_search.transform() and predicts
    """

    def __init__(self, param_search: ParamSearchCV):
        self.param_search = param_search

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

    def fit_predict_expectiles(
        self, X_train, y_train, X_val, y_val, w_train=None, w_val=None, X_pred=None
    ):
        """
        Fit expectile GAMs with validation and produce predictions.

        Return
            a structured dict:
                {"params": ..., "metrics": ..., "predictions": {...}}
        """
        if X_pred is None:
            X_pred = X_val

        # ──────────────────────────── Fit model ────────────────────────────
        best = self.param_search.fit(
            X_train, y_train, X_val, y_val, w_train=w_train, w_val=w_val
        )

        preds = {}
        for e, model in best.models.items():
            try:
                preds[f"units_pred_{e}"] = model.predict(X_pred)
            except Exception as err:
                self._log(f"[WARN] Prediction failed for expectile={e}: {err}")
                preds[f"units_pred_{e}"] = np.full(len(X_pred), np.nan)

        # add revenue predictions if price column available
        if isinstance(X_pred, pd.DataFrame) and "price" in X_pred.columns:
            for e in best.models.keys():
                preds[f"revenue_pred_{e}"] = (
                    preds[f"units_pred_{e}"] * X_pred["price"].to_numpy()
                )

        # ──────────────────────────── Collect metadata ────────────────────────────
        metrics = {
            "interval_score": getattr(best, "interval_score", None),
            "rmse_val": getattr(best, "rmse_val", None),
            "n_splines": getattr(best, "n_splines", None),
            "lam": getattr(best, "lam", None),
            "decay_rate": getattr(best, "decay_rate", None),
            "final_score": getattr(best, "final_score", None),
        }

        params = {
            "expectiles": self.param_search.expectiles,
            "n_splines": best.n_splines,
            "lam": best.lam,
        }

        self._log(
            f"[DBG GAMModeler] Using ParamSearchCV.best_ = {best}\n"
            f"    Validation metrics: {metrics}\n"
            f"    Prediction keys: {list(preds.keys())}"
        )

        print("\n" + "─ " * 10 + "✅ Generating Prediction Frames" + "─ " * 10 + "\n")
        return {
            "params": params,
            "metrics": metrics,
            "predictions": preds,
        }


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

        # compute weighted_pred
        if avg_col is not None:
            max_ratio = df["ratio"].max() if "ratio" in df.columns else 1.0
            confidence_weight = (
                1 - (df["ratio"] / max_ratio) if "ratio" in df.columns else 0.5
            )
            df["weighted_pred"] = df[base_col] * confidence_weight + df[avg_col] * (
                1 - confidence_weight
            )
        else:
            df["weighted_pred"] = df[base_col]

        return {
            "best_avg": DataEng.pick_best_by_group(df, "product", base_col),
            "best_weighted": DataEng.pick_best_by_group(df, "product", "weighted_pred"),
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

        # Build matrices
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

        # --------------- Fit + Predict ---------------
        print(
            "\n─────────────────────────────── 🤖 Modeling & Prediction ───────────────────────────────"
        )

        param_search = ParamSearchCV(
            n_splines_grid=(16, 20, 24, 30),
            loglam_range=(np.log(0.002), np.log(2.0)),
            n_lam=7,
            expectiles=(0.025, 0.5, 0.975),
            alpha=0.10,
            random_state=42,
            verbose=False,
        )
        modeler = GAMModeler(param_search)

        # Split X, y, and w TOGETHER to keep perfect alignment
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X, y, w, test_size=0.20, random_state=42
        )

        # Weight safety
        w_tr = np.clip(
            np.nan_to_num(w_tr, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None
        )
        w_val = np.clip(
            np.nan_to_num(w_val, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None
        )

        # ---- Call modeler with the required arguments ----
        res = modeler.fit_predict_expectiles(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            w_train=w_tr,
            w_val=w_val,
            X_pred=X,  # keep full-frame predictions for downstream assembly
        )
        print("\n" + "─ " * 10 + "✅ Generating Prediction Frames" + "─ " * 10 + "\n")

        # --------------- Results Assembly ---------------
        all_gam_results = (
            topsellers[["product", "price", "asin", "asp"]]
            .copy()
            .reset_index(drop=True)
        )

        # --- A) Add local support on the prediction grid (per product) ---
        # pick a base df holding observed rows (the data you fitted on)
        base_df = getattr(self, "pricing_df", None)
        if base_df is None and hasattr(self, "engineer") and hasattr(self.engineer, "pricing_df"):
            base_df = self.engineer.pricing_df


        # 1) iterate unique products (no groupby tuple confusion)
        for prod_key in all_gam_results["product"].dropna().unique():
            g_pred = all_gam_results.loc[all_gam_results["product"] == prod_key]

            # 2) observed prices must come from the training data, not the prediction grid
            if base_df is not None and "price" in base_df.columns:
                # robust match in case dtypes differ
                mask = (
                    base_df.get("product", "").astype(str) == str(prod_key)
                    if "product" in base_df.columns
                    else None
                )
                if mask is not None and mask.any():
                    obs_prices = base_df.loc[mask, "price"].to_numpy()
                else:
                    # fallback: if your training key isn’t named 'product', try asin/sku etc., or finally fallback to g_pred
                    obs_prices = g_pred["price"].to_numpy()
            else:
                obs_prices = g_pred["price"].to_numpy()  # last-resort fallback

            # 3) compute local window from observed price spacing
            u = np.unique(np.sort(obs_prices))
            if u.size >= 2:
                step = np.nanmedian(np.diff(u))
            else:
                step = max(0.25, np.nanstd(obs_prices) / 25.0)
            win = 1.25 * step

            # 4) support counts: how many observed prices within ±win of each predicted price
            P_pred = g_pred["price"].to_numpy()
            supp = np.array(
                [(np.abs(obs_prices - p) <= win).sum() for p in P_pred], dtype=int
            )

            # 5) write support back onto the prediction rows
            all_gam_results.loc[g_pred.index, "support_count"] = supp

        all_gam_results["support_count"] = (
            all_gam_results["support_count"].fillna(0).astype(int)
        )

        # ------------------------------------------------------------------------------------------

        # Get predictions flexibly
        res_preds = res.get("predictions", res.get("pred", {}))
        if not res_preds:
            raise ValueError(
                f"[GAMModeler] No predictions found. Available keys: {list(res.keys())}"
            )

        print(f"✨ Predictions added: {list(res_preds.keys())}")

        # Add unit + revenue predictions
        for k, arr in res_preds.items():
            if k.startswith("units_pred_"):
                all_gam_results[k] = np.asarray(arr, dtype=float)
                all_gam_results[k] = np.maximum(arr, 0.0)  # <-- hard floor at 0

                rev_key = k.replace("units_", "revenue_")
                all_gam_results[rev_key] = (
                    all_gam_results["price"].to_numpy() * all_gam_results[k]
                )

        # Sanity check
        pred_cols = [
            c
            for c in all_gam_results.columns
            if c.startswith(("units_pred_", "revenue_pred_"))
        ]
        if not pred_cols:
            raise ValueError(
                f"[_build_core_frames] Prediction assembly failed. Found keys={list(res_preds.keys())}, "
                f"DataFrame cols={list(all_gam_results.columns)}"
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

        print(
            "\n─────────────────────────────── 🎯 Pipeline Complete ───────────────────────────────\n"
        )
        return topsellers, elasticity_df, all_gam_results

    def _compute_best_tables(self, all_gam_results, topsellers):
        """
        Run Optimizer and ensure required cols are present.
        """
        bests = Optimizer.run(all_gam_results)
        best_avg = bests["best_avg"].copy()

        if "asin" not in best_avg.columns:
            pk_map = topsellers[["product", "asin"]].drop_duplicates()
            best_avg = best_avg.merge(pk_map, on="product", how="left")

        if "revenue_actual" not in best_avg.columns:
            if {"price", "shipped_units"}.issubset(best_avg.columns):
                best_avg["revenue_actual"] = (
                    best_avg["price"] * best_avg["shipped_units"]
                )
            else:
                ra = all_gam_results[
                    ["product", "price", "revenue_actual"]
                ].drop_duplicates()
                best_avg = best_avg.merge(ra, on=["product", "price"], how="left")

        return best_avg

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
