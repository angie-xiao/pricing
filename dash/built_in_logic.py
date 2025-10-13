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
import itertools
import math
import os
import pandas as pd
import numpy as np
from datetime import datetime

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
    def __init__(self, lam=None, n_splines=None, score=float("inf")):
        self.lam = lam
        self.n_splines = n_splines
        self.score = score


class ParamSearchCV:
    """
    Adaptive (n_splines, λ) tuner with:
      - λ floor expansion (prevents 'hugging the floor'),
      - periodic n_splines re-probe when floor-hugging persists,
      - golden-section refinement on log10(λ).
    Expects a scorer or falls back to MAE on validation.
    """

    def __init__(
        self,
        n_splines_grid=(11, 14, 17),
        n_lam=9,
        lam_span_decades=2.6,  # ~ 10**2.6 ≈ 400× span
        lam_floor_init=1.218e-4,  # observed floor
        ns_cap=None,  # will default below based on data hints
        patience=30,
        tol_rel=1e-3,  # relative tolerance on score improvements
        expectile_center=0.5,  # tune at the median expectile
        scorer=None,  # callable(model, X_val, y_val, w_val) -> lower is better
        random_state=None,
        verbose=True,
    ):
        self.n_splines_grid = tuple(int(x) for x in n_splines_grid)
        self.n_lam = int(n_lam)
        self.lam_span_decades = float(lam_span_decades)
        self.lam_floor_init = float(lam_floor_init)
        self.ns_cap = None if ns_cap is None else int(ns_cap)
        self.patience = int(patience)
        self.tol_rel = float(tol_rel)
        self.expectile_center = float(expectile_center)
        self.scorer = scorer
        self.random_state = random_state
        self.verbose = verbose

        self.best_ = None
        self.history_ = []
        self._cache = {}

        # backup
        self.n_splines_grid = tuple(int(x) for x in n_splines_grid)
        self.ns_grid = self.n_splines_grid  # ← alias so both names work

    # ---------- utilities ----------
    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _score_tuple(self, ns, lam, X_tr, y_tr, w_tr, X_va, y_va, w_va):
        """
        Fit one ExpectileGAM for (ns, lam) and return the validation score.
        Any numerical failure (e.g., non-PD matrix) yields +inf so the combo is ignored.
        """
        # clamp inputs to safe ranges
        ns = int(
            max(
                6,
                min(
                    int(ns),
                    self.ns_cap if self.ns_cap is not None else int(ns),
                    X_tr.shape[0] - 2,
                ),
            )
        )
        lam = float(max(lam, 1e-8))

        key = (ns, lam)
        if key in self._cache:
            return self._cache[key]

        from pygam import ExpectileGAM

        try:
            model = ExpectileGAM(expectile=self.expectile_center, lam=lam, n_splines=ns)
            model.fit(X_tr, y_tr, weights=w_tr)

            if self.scorer is not None:
                s = float(self.scorer(model, X_va, y_va, w_va))
            else:
                pred = model.predict(X_va)
                s = float(np.mean(np.abs(pred - y_va)))  # MAE

        except Exception:
            # any fitting failure → treat as very bad so tuner avoids it
            s = float("inf")

        self._cache[key] = s
        return s

    def _logspace10(self, lam_min, lam_max, n):
        a, b = math.log10(lam_min), math.log10(lam_max)
        # include endpoints
        return np.power(10.0, np.linspace(a, b, int(n))).tolist()

    def _symmetric_log_grid(self, center, width_decades, n):
        mid = math.log10(center)
        a = mid - width_decades / 2.0
        b = mid + width_decades / 2.0
        return np.power(10.0, np.linspace(a, b, int(n))).tolist()

    def _probe_ns_around(self, best_ns, cap):
        base = int(best_ns)
        cand = {base}
        for d in (2, 4, 6):
            if base + d <= cap:
                cand.add(base + d)
            if base - d >= 6:
                cand.add(base - d)
        return sorted(cand)

    def _golden_section_log10(self, f, a_log10, b_log10, n_eval=8):
        phi = (1 + 5**0.5) / 2
        invphi = 1 / phi
        c = b_log10 - invphi * (b_log10 - a_log10)
        d = a_log10 + invphi * (b_log10 - a_log10)
        fc, fd = f(c), f(d)
        for _ in range(max(2, int(n_eval)) - 2):
            if fc < fd:
                b_log10, d, fd = d, c, fc
                c = b_log10 - invphi * (b_log10 - a_log10)
                fc = f(c)
            else:
                a_log10, c, fc = c, d, fd
                d = a_log10 + invphi * (b_log10 - a_log10)
                fd = f(d)
        return (a_log10 + b_log10) / 2.0

    def fit_with_nlam_candidates(
        self,
        *,
        X_train,
        y_train,
        X_val,
        y_val,
        w_train=None,
        w_val=None,
        n_lam_candidates=(5, 7, 9, 11, 13),
        eff_bins=None,
    ):
        """
        Try several n_lam values on THIS instance and keep the best result in self.best_.
        Returns self.
        """
        best_snapshot = None
        best_score = float("inf")
        orig_n_lam = self.n_lam

        for nl in n_lam_candidates:
            self.n_lam = int(nl)
            # reset per-run state
            try:
                self.history_.clear()
            except Exception:
                self.history_ = []
            try:
                self._cache.clear()
            except Exception:
                self._cache = {}

            # run the tuner with current n_lam
            self.fit(
                X_tr=X_train,
                y_tr=y_train,
                X_va=X_val,
                y_va=y_val,
                w_tr=w_train,
                w_va=w_val,
                eff_bins=eff_bins,
            )

            cand = getattr(self, "best_", None)
            score = float(getattr(cand, "score", float("inf")))
            if score < best_score:
                best_score = score
                best_snapshot = {
                    "best_": cand,
                    "history_": list(getattr(self, "history_", [])),
                    "_cache": dict(getattr(self, "_cache", {})),
                    "n_lam": self.n_lam,
                }

        # restore the best run (or original n_lam if none improved)
        if best_snapshot is not None:
            self.best_ = best_snapshot["best_"]
            self.history_ = best_snapshot["history_"]
            self._cache = best_snapshot["_cache"]
            self.n_lam = best_snapshot["n_lam"]
        else:
            self.n_lam = orig_n_lam

        return self

    # ---------- main entry ----------
    def fit(self, X_tr, y_tr, X_va, y_va, w_tr=None, w_va=None, eff_bins=None):
        """
        Returns self with self.best_ set to a _TuneResult.
        """
        # default ns_cap if not provided
        if self.ns_cap is None:
            if eff_bins is not None:
                self.ns_cap = int(min(max(25, int(eff_bins)), 40))
            else:
                self.ns_cap = 25

        # ensure n_splines_grid reaches toward the cap
        base_grid = sorted(
            set(
                list(self.n_splines_grid)
                + [min(self.ns_cap, max(self.n_splines_grid) + 4), self.ns_cap]
            )
        )
        n_lam = max(5, int(self.n_lam))

        lam_floor = max(self.lam_floor_init, 1e-12)
        span_dec = float(self.lam_span_decades)

        # clamp span to a sane range and make sure it's finite/positive
        if not np.isfinite(span_dec) or span_dec <= 0:
            span_dec = 2.6
        span_dec = min(span_dec, 12.0)  # cap to avoid overflow

        # safe ceiling computation
        try:
            factor = 10.0**span_dec
        except OverflowError:
            factor = 1e12  # fallback cap

        lam_ceiling = lam_floor * factor
        if not np.isfinite(lam_ceiling) or lam_ceiling <= lam_floor:
            lam_ceiling = lam_floor * 1e6  # last-resort guard

        best = _TuneResult()  # (lam=None, ns=None, score=inf)
        no_improve = 0
        floor_hugging_streak = 0
        base_score_for_tol = None

        self._log("  🔧 Tuning n_lam candidates:", (n_lam,))

        for round_idx in range(1, self.patience + 1):
            lam_candidates = self._logspace10(lam_floor, lam_ceiling, n_lam)

            ns_list = base_grid if best.n_splines is None else [best.n_splines]
            improved = False

            self._log(
                f"🔍 Round {round_idx}:   🧭 Tuning λ with n_lam={n_lam} | floor={lam_floor:.6g} ceil={lam_ceiling:.6g}"
            )

            for ns in ns_list:
                for lam in lam_candidates:
                    s = self._score_tuple(ns, lam, X_tr, y_tr, w_tr, X_va, y_va, w_va)
                    self.history_.append((round_idx, ns, lam, s))
                    if s < best.score:
                        best = _TuneResult(lam=lam, n_splines=ns, score=s)
                        improved = True
                        if base_score_for_tol is None:
                            base_score_for_tol = s
                        self._log(
                            f"  🌟 New best: n_splines={ns}, λ={lam:.4g} (score={s:.4f})"
                        )

            if improved:
                no_improve = 0
                floor_hugging_streak = 0
            else:
                no_improve += 1

            # detect hugging the floor (lowest candidate repeatedly used with no gains)
            hugging = min(lam_candidates) <= lam_floor * 1.0001
            if hugging and not improved:
                floor_hugging_streak += 1
            else:
                floor_hugging_streak = 0

            # Strategy 1: expand floor & span when hugging persists
            if floor_hugging_streak >= 2:

                # old_floor = lam_floor
                lam_floor = max(lam_floor * 0.2, 1e-12)

                # expand but clamp span to avoid overflow
                span_dec = min(span_dec * 1.2, 12.0)

                try:
                    factor = 10.0**span_dec
                except OverflowError:
                    factor = 1e12

                lam_ceiling = lam_floor * factor
                if not np.isfinite(lam_ceiling) or lam_ceiling <= lam_floor:
                    lam_ceiling = lam_floor * 1e6

                # Strategy 2: re-probe n_splines around best
                probe_improved = False
                if best.n_splines is None:
                    center_ns = base_grid[len(base_grid) // 2]
                else:
                    center_ns = best.n_splines

                for ns2 in self._probe_ns_around(center_ns, self.ns_cap):
                    s2 = self._score_tuple(
                        ns2, best.lam, X_tr, y_tr, w_tr, X_va, y_va, w_va
                    )
                    if s2 < best.score * (1 - self.tol_rel):
                        self._log(
                            f"  🔧 ns re-probe improved: {best.n_splines}→{ns2} @ λ={best.lam:.4g} | {best.score:.4f}→{s2:.4f}"
                        )
                        best = _TuneResult(lam=best.lam, n_splines=ns2, score=s2)
                        probe_improved = True

                if probe_improved:
                    no_improve = 0
                    # next round will sweep λ around this improved ns
                    continue

            # plateau → finish with golden-section on log-λ
            if no_improve >= self.patience // 2:
                self._log(
                    "  🧯 Plateau detected — finishing with golden-section on log10(λ)."
                )
                a = math.log10(max(best.lam / 10.0, 1e-9))
                b = math.log10(best.lam * 3.0)

                def f(loglam):
                    lam = 10**loglam
                    return self._score_tuple(
                        best.n_splines, lam, X_tr, y_tr, w_tr, X_va, y_va, w_va
                    )

                best_loglam = self._golden_section_log10(f, a, b, n_eval=8)
                lam_refined = 10**best_loglam
                s_refined = self._score_tuple(
                    best.n_splines, lam_refined, X_tr, y_tr, w_tr, X_va, y_va, w_va
                )
                if s_refined < best.score:
                    best = _TuneResult(
                        lam=lam_refined, n_splines=best.n_splines, score=s_refined
                    )
                break

            if no_improve >= self.patience:
                self._log("  🧯 Patience exhausted — stopping adaptive search.")
                break

        self.best_ = best
        self._log(
            f"🌟 Best config: n_splines={best.n_splines} (k_eff={best.n_splines}), λ={best.lam:.4g} (score={best.score:.4f})"
        )
        return self


class GAMModeler:
    """
    Thin orchestrator:
    - builds X from feature_cols
    - fits one ExpectileGAM per expectile
    - if self.param_search.best_ is available, uses tuned (lam, n_splines) for final fits
      otherwise uses base_gam_kwargs (+ per_expectile overrides)
    - can append predictions back onto a DataFrame
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

    # ---------------- internal helpers ----------------

    def _log(self, msg: str):
        """Timestamped console log for debugging."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def _to_df(self, X) -> pd.DataFrame:
        """
        Convert arbitrary X into a normalized DataFrame with expected schema.
        NOTE: This references EXPECTED_COLS and CAT_COLS if you have them defined globally.
        If not, we simply keep X as-is with feature_cols.
        """
        # Prefer using the provided feature_cols to avoid hidden global dependencies.
        if isinstance(X, pd.DataFrame):
            # If EXPECTED_COLS exists in your codebase, keep your old behavior,
            # else just use feature_cols subset.
            try:
                df = X[EXPECTED_COLS].copy()  # type: ignore[name-defined]
            except Exception:
                df = X[self.feature_cols].copy()
        else:
            arr = np.asarray(X)
            # If you previously enforced EXPECTED_COLS, fallback to feature_cols names
            try:
                cols = EXPECTED_COLS  # type: ignore[name-defined]
            except Exception:
                cols = self.feature_cols
            df = pd.DataFrame(arr, columns=cols)

        # Cast categorical columns if your pipeline defines them
        try:
            for c in CAT_COLS:  # type: ignore[name-defined]
                if c in df.columns:
                    df[c] = df[c].astype("string")
        except Exception:
            pass

        return df

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _make_X(self, df_like: pd.DataFrame) -> np.ndarray:
        """Build X from df_like using self.feature_cols, enforcing 2D shape."""
        X = df_like[self.feature_cols].to_numpy()
        return self._ensure_2d(X)

    def _assert_feature_match(self, model: ExpectileGAM, X: np.ndarray):
        """Guard that predict-time features match the model's fitted dimensionality."""
        m_expected = getattr(model, "statistics_", {}).get("m", None)
        if m_expected is not None and X.shape[1] != m_expected:
            raise ValueError(
                f"Feature mismatch: predict has {X.shape[1]} features, "
                f"but model expects {m_expected}."
            )

    def _sanitize_prediction_rows(
        self, df: pd.DataFrame, pred_cols: Iterable[str]
    ) -> pd.DataFrame:
        """Drop rows with NaN price or NaN in any prediction columns (if present)."""
        keep = pd.Series(True, index=df.index)
        if "price" in df.columns:
            keep &= df["price"].notna()
        pred_cols = [c for c in pred_cols if c in df.columns]
        if pred_cols:
            keep &= df[pred_cols].notna().all(axis=1)
        return df.loc[keep].copy()

    # ---------------- tuned-params plumbing ----------------

    def tuned_gam_kwargs(self, q: float) -> dict:
        """
        Build kwargs for final ExpectileGAM fits using tuned params.
        Requires self.param_search.best_ with .lam and .n_splines.
        """
        if not hasattr(self, "param_search") or not hasattr(self.param_search, "best_"):
            raise RuntimeError(
                "ParamSearchCV.best_ missing; tune() must run before fit()."
            )
        b = self.param_search.best_
        return {
            "expectile": float(q),
            "lam": float(b.lam),
            "n_splines": int(b.n_splines),
        }

    def _final_kwargs_for(self, q: float) -> dict:
        """
        Decide final kwargs for a given expectile:
        - If tuned params exist (self.param_search.best_), use those (and only override expectile).
        - Else fall back to base_gam_kwargs + per_expectile_kwargs overrides.
        """
        has_tuned = hasattr(self, "param_search") and hasattr(
            self.param_search, "best_"
        )
        if has_tuned:
            return self.tuned_gam_kwargs(q)

        # kw = dict(self.base_gam_kwargs)
        # kw.update(self.per_expectile_kwargs.get(q, {}))
        # kw["expectile"] = float(q)
        try:
            kw = self.tuned_gam_kwargs(q)  # will use self.param_search.best_
        except Exception:
            kw = dict(self.base_gam_kwargs)
            kw.update(self.per_expectile_kwargs.get(q, {}))
            kw["expectile"] = float(q)

        return kw

    # ---------------- public API ----------------

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train: pd.Series,
        *,
        expectiles: Iterable[float] = EXPECTILES,
        weights: Optional[np.ndarray] = None,  # optional; keeps backward-compat
        verbose: bool = False,
    ) -> "GAMModeler":
        """
        Fit ExpectileGAM models for each expectile in `expectiles`.
        If a tuner has been run (self.param_search.best_), uses tuned (lam, n_splines).
        Otherwise, uses base_gam_kwargs (+ per-expectile overrides).
        """
        X_train = self._make_X(train_df)
        self.models = {}

        for q in expectiles:
            kw = self._final_kwargs_for(q)
            if verbose:
                print(f"[GAMModeler] Fitting ExpectileGAM(q={q}) with kwargs={kw}")
            model = ExpectileGAM(**kw).fit(X_train, y_train, weights=weights)

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
            temp = ParamSearchCV()
            param_search_kwargs = {
                "n_splines_grid": temp.n_splines_grid,
                "n_lam": temp.n_lam,
                "lam_span_decades": temp.lam_span_decades,
                "lam_floor_init": temp.lam_floor_init,
                "ns_cap": temp.ns_cap,
                "patience": temp.patience,
                "tol_rel": temp.tol_rel,
                "expectile_center": temp.expectile_center,
                "scorer": temp.scorer,
                "random_state": temp.random_state,
                "verbose": temp.verbose,
            }

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

        # --- try a small grid of candidate settings; pick best on validation ---
        candidates = {
            # vary spline flexibility a bit around your current grid
            "n_splines_grid": [
                sorted({9, 11, 13, 16, 20, 24, 30, 38}),  # your current
                [9, 11, 13, 15, 17, 21, 25, 32],  # slightly denser mid-range
                [11, 14, 17, 21, 25, 32, 38],  # push upper side a touch
            ],
            # vary λ floor; keep same upper bound
            "loglam_range": [
                (np.log(1e-6), np.log(5.0)),
                (np.log(1e-5), np.log(5.0)),
                (np.log(1e-4), np.log(5.0)),
            ],
            # let the inner routine sweep a couple of n_lam tuples
            "n_lam_candidates": [
                (5, 7, 9, 11, 13),
                (7, 9, 11, 13, 15),
            ],
        }

        best_ps = None
        best_score = float("inf")

        keys = list(candidates.keys())
        for combo in itertools.product(*[candidates[k] for k in keys]):

            lam_min = 1e-6
            lam_max = 5.0

            ps = ParamSearchCV(
                n_splines_grid=sorted({9, 11, 13, 16, 20, 24, 30, 38}),
                n_lam=9,  # int here; you'll sweep others below
                lam_floor_init=lam_min,  # from old loglam_range lower bound
                lam_span_decades=np.log10(
                    lam_max / lam_min
                ),  # from old loglam_range span
                random_state=42,
                verbose=True,
            )
            ps.fit_with_nlam_candidates(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_val,
                y_val=y_val,
                w_train=w_tr,
                w_val=w_val,
                n_lam_candidates=(5, 7, 9, 11, 13),
            )

            # prefer best_.score if your ParamSearchCV exposes it; otherwise skip if absent
            score = float(getattr(getattr(ps, "best_", None), "score", np.inf))
            if score < best_score:
                best_ps, best_score = ps, score

        # keep the winner for downstream as before
        param_search = best_ps

        # --- unchanged from here on ---
        best = getattr(param_search, "best_", None)

        best_lam = float(
            getattr(
                best,
                "lam",
                param_search.lam_floor_init
                * (
                    10.0 ** (param_search.lam_span_decades / 2.0)
                ),  # geometric mid of the span
            )
        )

        best_ns = int(
            getattr(
                best,
                "n_splines",
                sorted(param_search.ns_grid)[len(param_search.ns_grid) // 2],
            )
        )

        metrics = getattr(param_search, "best_metrics_", None)
        if metrics is not None:
            print(f"    Validation metrics: {metrics}")

        modeler = GAMModeler(
            feature_cols=["price"],
            base_gam_kwargs={"lam": best_lam, "n_splines": best_ns},
        )
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
