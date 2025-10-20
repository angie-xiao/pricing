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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from pygam import ExpectileGAM, s, GammaGAM


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
        self.best_ns_ = None
        self.best_lam_ = None
        self.best_params_ = {}
        self.best_score_ = float("inf")

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
    def __init__(
        self,
        n_splines_grid=None,
        n_lam: int = 12,
        lam_floor_init: float = 1e-4,
        lam_span_decades: float = 5.0,  # lam_max = lam_floor_init * 10**lam_span_decades
        random_state: int | None = 42,
        verbose: bool = True,
        objective=None,  # <-- closure: objective(ns, lam) -> float
        logger_print=None,
        **kwargs,
    ):

        noop = lambda *a, **k: None
        self._log = logger_print or (print if verbose else noop)
        self.objective = objective

        # sane defaults; caller can override via arg
        self.n_splines_grid = (
            list(n_splines_grid)
            if n_splines_grid is not None
            else [8, 10, 12, 14, 17, 20, 24, 28, 32]
        )
        lam_min = float(lam_floor_init)
        lam_max = lam_min * (10.0 ** float(lam_span_decades))
        self.lam_grid = np.geomspace(lam_min, lam_max, int(n_lam))

        self.random_state = random_state
        self.verbose = verbose

    def ns_neighbors(self, ns):
        # Tight neighborhood keeps budget small but allows refinement
        return [x for x in (ns - 1, ns, ns + 1) if 8 <= x <= 40]

    def lam_p_zoom_grid(self, lam_p0):
        # ±0.25 decades around the seed → ~×0.56..×1.78 multiplicative band
        return list(np.power(10.0, np.linspace(-0.25, +0.25, 7)) * float(lam_p0))

    def _pp_kv(self, d: dict) -> str:
        """
        Build a single-line, fixed-layout KV string with persistent column widths.
        Column widths grow as longer values appear to keep pipes aligned across lines.
        """
        # Canonical column order for tuning logs
        order = ["stage", "ns", "lam", "lam_p", "loss"]

        # Initialize persistent widths on first use
        if not hasattr(self, "_tlog_w"):
            # Seed with sensible minima so early lines don't look jagged
            self._tlog_w = {"stage": 8, "ns": 3, "lam": 12, "lam_p": 12, "loss": 10}
            self._tlog_kw = max(len(k) for k in order)

        def _val_str(v):
            if isinstance(v, (int, float)):
                return f"{float(v):.6g}"  # compact, consistent numeric formatting
            return str(v)

        parts = []

        # Emit canonical columns first (if present), right-aligned to sticky widths
        for k in order:
            if k in d:
                s = _val_str(d[k])
                self._tlog_w[k] = max(self._tlog_w.get(k, 0), len(s))
                parts.append(f"{k:>{self._tlog_kw}} = {s:>{self._tlog_w[k]}}")

        # Append any extras in alpha order, maintaining sticky widths
        extras = sorted(k for k in d.keys() if k not in order)
        for k in extras:
            s = _val_str(d[k])
            self._tlog_w[k] = max(self._tlog_w.get(k, 0), len(s))
            self._tlog_kw = max(self._tlog_kw, len(k))
            parts.append(f"{k:>{self._tlog_kw}} = {s:>{self._tlog_w[k]}}")

        return " | ".join(parts)

    def _tlog(self, stage: str, badge: str | None = None, **kwargs):
        from datetime import datetime

        line = self._pp_kv(dict(stage=stage, **kwargs))
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = (badge + " ") if badge else ""
        print(f"[{ts}] {prefix}{line}", flush=True)

    def _lam_from_lam_prime(self, ns: int, lam_prime: float) -> float:
        """lam = lam' / ns^2  (scale-invariant parameterization)"""
        ns2 = max(1, int(ns)) ** 2
        return float(lam_prime) / float(ns2)

    def _stage_A_coarse(self):
        """
        Stage A: coarse, wide, scale-invariant sweep over ns and λ' (lam_p).
        Returns:
            best_ns, best_lam_p, best_loss, scores_A  (scores_A: list[(loss, ns, lam_p)])
        """
        # Coarse grids (bounded, SKU-agnostic)
        ns_grid_A = [8, 12, 16, 20, 24, 28, 32, 40]
        lam_p_gridA = np.power(10.0, np.linspace(-3.0, +3.0, 9))  # λ' in decades

        scores_A = []
        best_loss, best_ns, best_lam_p = float("inf"), None, None

        for ns in ns_grid_A:
            for lam_p in lam_p_gridA:
                lam = self._lam_from_lam_prime(ns, lam_p)  # λ = λ'/ns^2
                loss = float(self.objective(ns, lam))
                improved = loss < best_loss - 1e-12
                self._tlog(
                    "coarse",
                    badge=("🌟 new best score" if improved else "💤 no improvement"),
                    ns=ns, lam_p=lam_p, lam=lam, loss=loss
                )
                scores_A.append((loss, ns, lam_p))
                if improved:
                    best_loss, best_ns, best_lam_p = loss, ns, lam_p

        scores_A.sort(key=lambda t: t[0])
        self._tlog("A_best", badge="🏁 end", ns=best_ns, lam_p=best_lam_p, loss=best_loss)
        return best_ns, best_lam_p, best_loss, scores_A


    def _stage_B_zoom(
        self,
        scores_A,
        *,
        best_ns,
        best_lam_p,
        best_loss,
        top_k: int = 3,
        eps: float = 1.25,
        rounds: int = 3,
        n_grid: int = 9,
        zoom_factor: float = 5.0,
    ):
        """
        Stage B: local zoom around top-K seeds from Stage A.
        Returns:
            best_ns, best_lam_p, best_loss
        Notes:
            - This uses the class helpers `_zoom_bounds_around(...)` and `_search_lambda_zoom(...)`
            you already have. If your current Stage B has slightly different names/signatures,
            keep the inner logic the same—only the function boundary is new.
        """
        # Keep only strong seeds within an epsilon band around the best and cap K.
        topK = [t for t in scores_A if t[0] <= eps * best_loss][:top_k]

        for loss0, ns0, lam_p0 in topK:
            # define an evaluator for a fixed ns, searching over lam'
            def _eval(lam_p):
                lam = self._lam_from_lam_prime(ns0, lam_p)
                loss = float(self.objective(ns0, lam))
                # light tracing for zoom evaluations
                self._tlog("zoom", ns=ns0, lam_p=lam_p, lam=lam, loss=loss)
                return loss

            # choose zoom window around current lam_p seed
            lo, hi = self._zoom_bounds_around(float(lam_p0), factor=zoom_factor, lo_floor=1e-8, hi_ceiling=1e8)

            # run your existing zoom/search helper
            # EXPECTED signature (adjust to match your actual implementation if needed):
            #   best_lam_p_i, best_loss_i, history = self._search_lambda_zoom(
            #       evaluate=_eval, lam_min=lo, lam_max=hi, rounds=rounds, n_grid=n_grid, zoom_factor=zoom_factor
            #   )
            best_lam_p_i, best_loss_i, _hist = self._search_lambda_zoom(
                evaluate=_eval,
                lam_min=lo,
                lam_max=hi,
                rounds=rounds,
                n_grid=n_grid,
                zoom_factor=zoom_factor,
            )

            if best_loss_i < best_loss - 1e-12:
                best_loss, best_ns, best_lam_p = best_loss_i, ns0, best_lam_p_i
                self._tlog("zoom_best", badge="🌟 new best score", ns=best_ns, lam_p=best_lam_p, loss=best_loss)
            else:
                self._tlog("zoom_best", badge="💤 no improvement", ns=ns0, lam_p=best_lam_p_i, loss=best_loss_i)

        self._tlog("B_best", badge="🏁 end", ns=best_ns, lam_p=best_lam_p, loss=best_loss)
        return best_ns, best_lam_p, best_loss


    def fit(self):
        """
        Progressive Zoom, scale-invariant tuner.
        - Stage A: coarse, wide search over n_splines (ns) and λ' (lam_prime)
        - Stage B: local zoom around the top-K Stage-A seeds
        Notes:
        * Actual λ used for fitting is λ = λ' / ns^2  (handled by _lam_from_lam_prime).
        * Attributes exposed:
            - self.best_ns_   : int (n_splines)
            - self.best_lam_  : float (actual λ used to fit final model)
            - self.best_params_: {"n_splines": int, "lam": float}
            - self.best_score_: float (loss)
        """
        if self.objective is None:
            raise RuntimeError("ParamSearchCV requires an objective(ns, lam).")

        # ---------- Stage A ----------
        best_ns, best_lam_p, best_loss, scores_A = self._stage_A_coarse()

        # ---------- Stage B ----------
        best_ns, best_lam_p, best_loss = self._stage_B_zoom(
            scores_A,
            best_ns=best_ns,
            best_lam_p=best_lam_p,
            best_loss=best_loss,
            # keep defaults, or override to match your existing constants:
            # top_k=3, eps=1.25, rounds=3, n_grid=9, zoom_factor=5.0,
        )

        # ---------- Finalize ----------
        best_lam = self._lam_from_lam_prime(best_ns, best_lam_p)  # actual λ
        self.best_params_ = {"n_splines": int(best_ns), "lam": float(best_lam)}
        self.best_ns_ = int(best_ns)
        self.best_lam_ = float(best_lam)
        self.best_score_ = float(best_loss)
        self._tlog("best", badge="✅ finalized best", ns=self.best_ns_, lam=self.best_lam_, lam_p=best_lam_p, loss=self.best_score_)
        return self

 

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
    Fits three expectile GAMs (q=0.025, 0.50, 0.975) on price -> units.
    - Non-negativity via softplus at prediction time (matches tuning loss).
    - Optional monotonic constraint (units non-increasing in price).
    """

    def __init__(
        self,
        feature_cols,
        base_gam_kwargs=None,
        logger_print=None,
        # NEW: light-touch controls (no hand-tuned alphas)
        monotone_default: str | None = "down",  # "down", "up", or None
        median_filter_window: int | None = None,  # odd int; None disables
    ):
        self.feature_cols = list(feature_cols)
        self.base_gam_kwargs = dict(base_gam_kwargs or {})
        self._logger = logger_print or (lambda s: None)

        self.models_ = {}  # q -> fitted GAM
        self.fitted_ = False

        # NEW
        self.monotone_default = monotone_default
        self.median_filter_window = median_filter_window

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.where(x > 20, x, np.log1p(np.exp(x)))

    def _build_gam(self, expectile: float):
        # Use a constrained spline if requested (reduces spikes in sparse zones).

        kwargs = dict(self.base_gam_kwargs)
        terms = s(0, constraints=("monotonic_dec" if self.enforce_monotonic else None))
        return ExpectileGAM(expectile=expectile, terms=terms, **kwargs)

    def _project_monotone(
        self, x: np.ndarray, y: np.ndarray, enforce: bool
    ) -> np.ndarray:
        """
        Project y onto a monotone-nonincreasing function of x using isotonic regression.
        If `enforce` is False or data are too short, returns y unchanged.
        """
        if not enforce:
            return y

        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        # not enough points or degenerate x → skip
        if x.size < 3 or np.allclose(x.min(), x.max(), equal_nan=True):
            return y

        # sort by x, fit isotonic (decreasing) on the sorted grid, then invert the permutation
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        y_fit_sorted = ir.fit_transform(x_sorted, y_sorted)  # <-- key change

        # map back to original order
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        y_fit = y_fit_sorted[inv]

        return y_fit

    def _maybe_median_filter(self, y: np.ndarray, window: int | None) -> np.ndarray:
        """
        Centered rolling median; if window is None or <3, passthrough.
        """
        if window is None or window < 3 or window % 2 == 0 or y.size < window:
            return y
        # pandas rolling median (centered), then fill NaNs with original values
        s = pd.Series(y, copy=False)
        m = s.rolling(window=window, center=True, min_periods=1).median().to_numpy()
        return np.where(np.isnan(m), y, m)

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train: np.ndarray,
        weights: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "GAMModeler":
        X = (
            pd.to_numeric(train_df[self.feature_cols[0]], errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1, 1)
        )
        y = np.asarray(y_train, dtype=float)
        w = None if weights is None else np.asarray(weights, dtype=float)

        # Decide which quantiles to fit (defaults to 0.025/0.5/0.975)
        qs = getattr(self, "quantiles_", (0.025, 0.5, 0.975))
        self.models_ = {}

        for q in qs:
            gam = ExpectileGAM(
                expectile=float(q), **self.base_gam_kwargs, max_iter=5000
            )
            gam.fit(X, y, weights=w)
            self.models_[float(q)] = gam
            if verbose:
                self._logger(
                    f"[GAMModeler] Fitted ExpectileGAM(q={q}) "
                    f"with kwargs={{'lam': {gam.lam}, 'n_splines': {gam.n_splines}, 'expectile': {q}, 'max_iter': 5000}}"
                )

        self.fitted_ = True
        return self

    def _predict_units(
        self,
        prices: np.ndarray,
        *,
        q: float,
        monotone: str | None = None,  # <- NEW
        median_filter_window: int | None = None,  # <- NEW
        group_idx: np.ndarray | None = None,  # optional labels (e.g., product)
    ) -> np.ndarray:
        """
        Predict units at quantile q for given prices, then optionally:
          - enforce monotonicity per group via isotonic regression
          - apply a light median filter per group
        """
        if not self.fitted_ or q not in self.models_:
            raise RuntimeError(f"GAM for q={q} is not fitted")

        model = self.models_[q]
        # raw prediction (expectile GAM approximates conditional expectile of units)
        units = model.predict(prices.reshape(-1, 1)).astype(float)
        # clamp at 0 to avoid tiny negatives
        units = np.maximum(units, 0.0)

        # If no grouping, treat whole vector as one group
        if group_idx is None:
            if monotone in ("down", "up"):
                units = self._project_monotone(prices, units, monotone)
            units = self._maybe_median_filter(units, median_filter_window)
            return units

        # Group-aware projection & smoothing (e.g., per product)
        units_out = units.copy()
        # Ensure numpy array of labels
        g = np.asarray(group_idx)
        # Iterate each group label once
        for label in np.unique(g):
            mask = g == label
            if not np.any(mask):
                continue
            u = units[mask]
            p = prices[mask]

            if monotone in ("down", "up"):
                u = self._project_monotone(p, u, monotone)

            u = self._maybe_median_filter(u, median_filter_window)
            units_out[mask] = u

        return units_out

    def add_predictions_to_df(
        self,
        df: pd.DataFrame,
        *,
        write_units=True,
        write_revenue=True,
        price_col="price",
        inplace=False,
    ):
        if not inplace:
            df = df.copy()

        P = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
        group_idx = df["product"].to_numpy() if "product" in df.columns else None

        # Prefer whatever was actually fitted; if not present, fall back to the usual trio
        fitted_qs = (
            tuple(sorted(self.models_.keys()))
            if getattr(self, "models_", None)
            else (0.025, 0.5, 0.975)
        )

        # 1) write UNITS first (no revenue yet)
        for q in (0.025, 0.5, 0.975):
            if q not in self.models_:
                self._logger(
                    f"[GAMModeler] Skip q={q} — not fitted; fitted_qs={fitted_qs}"
                )
                continue
            units = self._predict_units(
                P,
                q=q,
                monotone=self.monotone_default,
                median_filter_window=self.median_filter_window,
                group_idx=group_idx,
            )
            if write_units:
                df[f"units_pred_{q}"] = units

        # 2) support-aware shrinkage (this now affects revenue too)
        if "support_count" in df.columns:
            supp = (
                pd.to_numeric(df["support_count"], errors="coerce")
                .fillna(0.0)
                .to_numpy()
            )
            soft_min = 12.0
            floor = 0.25
            shrink = np.clip(supp / soft_min, floor, 1.0)
            for q in (0.025, 0.5, 0.975):
                col = f"units_pred_{q}"
                if col in df:
                    df[col] = np.maximum(
                        0.0,
                        pd.to_numeric(df[col], errors="coerce").to_numpy() * shrink,
                    )

        # 3) now write (or recompute) REVENUE from the (possibly shrunk) units
        if write_revenue:
            for q in (0.025, 0.5, 0.975):
                ucol = f"units_pred_{q}"
                if ucol in df:
                    df[f"revenue_pred_{q}"] = (
                        P * pd.to_numeric(df[ucol], errors="coerce").to_numpy()
                    )

        return df

    @staticmethod
    def sanitize_results_for_downstream(df: pd.DataFrame) -> pd.DataFrame:
        # Any light-touch cleaning can live here
        cols = [c for c in df.columns if c.startswith("units_pred_")]
        for c in cols:
            df[c] = np.maximum(df[c].to_numpy(), 0.0)
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


class PipelineCore:
    """
    Stateless-ish helpers extracted from PricingPipeline._build_core_frames.
    Holds references to collaborators and constants; all steps are small and testable.
    """

    def __init__(
        self,
        *,
        engineer,
        feat_cols,
        target_col,
        weight_col,
        logger_print,
        Weighting,
        ElasticityAnalyzer,
        GAMModeler,
        ParamSearchCV,
        support_weighting: str = "kde",  # <— NEW
    ):
        self.engineer = engineer
        self.FEAT_COLS = list(feat_cols)
        self.TARGET_COL = target_col
        self.WEIGHT_COL = weight_col
        self._log = logger_print if logger_print is not None else (lambda s: None)

        # collaborators
        self.Weighting = Weighting
        self.ElasticityAnalyzer = ElasticityAnalyzer
        self.GAMModeler = GAMModeler
        self.ParamSearchCV = ParamSearchCV

        # NEW
        self.support_weighting = support_weighting.lower()

    # convenient setter if you prefer to construct engineer in PricingPipeline
    def set_engineer(self, engineer) -> None:
        self.engineer = engineer

    def _loss(self, y_true, y_pred, sample_weight=None) -> float:

        err = y_true - y_pred
        if sample_weight is not None:
            return float(np.average(err * err, weights=sample_weight))  # weighted MSE
        return float(np.mean(err * err))

    def _bin_and_aggregate(
        self,
        df: pd.DataFrame,
        *,
        price_col: str,
        target_col: str,
        weight_col: str,
        bins: int = 20,
        group_cols: (
            list[str] | None
        ) = None,  # accepted for compatibility; ignored in univariate agg
    ) -> pd.DataFrame:
        """
        Aggregate raw points into price bins for *training only* (univariate fit).
        Returns a small DataFrame with columns: ["price", target_col, weight_col].

        Strategy:
        - Bin prices using quantiles (fallback to uniform bins if quantiles collapse)
        - Within each non-empty bin, compute:
            price  := weight-avg price within the bin
            target := weight-avg target within the bin
            weight := sum of weights within the bin
        """
        # Extract arrays
        p = df[price_col].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        w = df[weight_col].to_numpy(dtype=float)

        # Basic sanitation
        m = np.isfinite(p) & np.isfinite(y) & np.isfinite(w)
        if not np.all(m):
            p, y, w = p[m], y[m], w[m]

        w = np.nan_to_num(w, nan=1.0, posinf=3.0, neginf=1e-6)
        w[w <= 0] = 1e-6

        if p.size == 0:
            # Degenerate fallback: return empty frame with expected columns
            return pd.DataFrame({"price": [], target_col: [], weight_col: []})

        # Make bin edges from quantiles; if too few unique edges, fall back to uniform bins
        q_edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        q_edges = np.unique(q_edges)
        if q_edges.size < 3:  # quantiles collapsed
            pmin, pmax = float(np.min(p)), float(np.max(p))
            if pmax == pmin:
                # All prices identical: single-bin aggregate
                return pd.DataFrame(
                    {
                        "price": [float(pmin)],
                        target_col: [float(np.average(y, weights=w))],
                        weight_col: [float(np.sum(w))],
                    }
                )
            edges = np.linspace(pmin, pmax, bins + 1)
        else:
            edges = q_edges

        # Digitize into bins [0 .. nbins-1]
        idx = np.digitize(p, edges[1:-1], right=False)
        nb = edges.size - 1

        # Aggregate per bin
        rows = []
        for b in range(nb):
            sel = idx == b
            if not np.any(sel):
                continue
            w_b = w[sel]
            p_b = p[sel]
            y_b = y[sel]

            wsum = float(np.sum(w_b))
            # Weight-averaged representatives
            p_bar = float(np.sum(p_b * w_b) / wsum)
            y_bar = float(np.sum(y_b * w_b) / wsum)

            rows.append((p_bar, y_bar, wsum))

        if not rows:
            return pd.DataFrame({"price": [], target_col: [], weight_col: []})

        out = pd.DataFrame(rows, columns=["price", target_col, weight_col]).sort_values(
            "price"
        )
        out = out.reset_index(drop=True)
        return out

    def _support_prior(
        self, prices: np.ndarray, bandwidth: float | None = None
    ) -> np.ndarray:
        """
        Smooth density prior over price using a Gaussian kernel (1D KDE).
        Returns weights in [0.2, 1.0].
        """
        p = np.asarray(prices, float).ravel()
        if p.size < 5:
            return np.ones_like(p)

        std = float(np.nanstd(p)) or 1.0
        bw = (
            bandwidth
            if (bandwidth and bandwidth > 0)
            else 1.06 * std * p.size ** (-1 / 5)
        )

        # KDE against itself
        z = (p[:, None] - p[None, :]) / (bw if bw > 1e-12 else 1.0)
        dens = np.exp(-0.5 * z**2).mean(axis=1) / np.sqrt(2 * np.pi)
        dens = np.nan_to_num(dens, nan=np.nanmedian(dens) or 1.0)

        dmin = float(np.nanmin(dens))
        dmax = float(np.nanmax(dens))
        rng = dmax - dmin
        if not np.isfinite(rng) or rng < 1e-12:
            return np.ones_like(p)

        dens = (dens - dmin) / rng  # in [0,1]
        return 0.2 + 0.8 * dens  # in [0.2,1.0]

    def _loss(self, y_true, y_pred, sample_weight=None) -> float:
        err = y_true - y_pred
        if sample_weight is not None:
            return float(np.average(err * err, weights=sample_weight))  # weighted MSE
        return float(np.mean(err * err))

    # ---------------------------- Prep & weights ----------------------------

    def prepare_topsellers(self) -> pd.DataFrame:
        if self.engineer is None:
            raise RuntimeError(
                "PipelineCore.engineer is not set. Call set_engineer(engineer)."
            )
        # Guard: if someone accidentally passed a DataFrame here again, just return it
        if isinstance(self.engineer, pd.DataFrame):
            return self.engineer.copy()
        # Normal path
        if not hasattr(self.engineer, "prepare"):
            raise AttributeError("engineer object must have a .prepare() method")
        return self.engineer.prepare()

    @staticmethod
    def ensure_columns(topsellers: pd.DataFrame) -> None:
        if "asp" not in topsellers.columns and "price" in topsellers.columns:
            topsellers["asp"] = pd.to_numeric(topsellers["price"], errors="coerce")
        if "__intercept__" not in topsellers.columns:
            topsellers["__intercept__"] = 1.0

    def compute_weights(self, topsellers: pd.DataFrame) -> np.ndarray:
        W = self.Weighting()
        return W.build(topsellers)

    # ---------------------------- Features / numeric ----------------------------
    def assemble_numeric(self, topsellers: pd.DataFrame):
        # minimal, no DataEngineer dependency
        need_cols = self.FEAT_COLS + [self.TARGET_COL, self.WEIGHT_COL]
        ts = topsellers[need_cols].copy()

        # coerce to numeric
        for c in need_cols:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

        ts = ts.dropna(subset=need_cols).reset_index(drop=True)

        # matrices
        X = ts[self.FEAT_COLS].to_numpy(dtype=float)
        y = ts[self.TARGET_COL].to_numpy(dtype=float)
        w = ts[self.WEIGHT_COL].to_numpy(dtype=float)

        # sanitize weights
        w = np.nan_to_num(w, nan=1.0, posinf=3.0, neginf=1e-6)
        w[w <= 0] = 1e-6

        # remember which column is price (needed by make_objective)
        if hasattr(self, "FEAT_COLS") and "price" in self.FEAT_COLS:
            self._price_col = self.FEAT_COLS.index("price")
        else:
            self._price_col = 0  # fallback

        return ts, X, y, w

    def elasticity_best_effort(self, topsellers: pd.DataFrame) -> pd.DataFrame:
        try:
            return self.ElasticityAnalyzer.compute(topsellers)
        except Exception:
            return pd.DataFrame(columns=["product", "ratio", "elasticity_score"])

    def standardize_continuous_inplace(self, X: np.ndarray, cont_idx=(0,)):
        """
        Z-score selected continuous columns of X in place.
        Default assumes column 0 is 'price'. Categorical columns are excluded.
        Saves params for reuse at predict-time.
        """
        if X.size == 0 or not cont_idx:
            return

        cont_idx = tuple(cont_idx)  # ensure indexable
        mu = np.nanmean(X[:, cont_idx], axis=0)
        sd = np.nanstd(X[:, cont_idx], axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)  # guard

        # in-place standardization
        X[:, cont_idx] = (X[:, cont_idx] - mu) / sd

        # store for downstream (prediction grids etc.)
        self._zparams = {"idx": np.array(cont_idx), "mu": mu, "sd": sd}

    # ---------------------------- Split & objective ----------------------------
    def train_val_split(self, ts: pd.DataFrame, y_all: np.ndarray, w_all: np.ndarray):
        """
        Build a simple train/val split for the univariate price GAM:
        - TRAIN: binned & aggregated by price (reduces noise; uses weights)
        - VAL:   original point-wise (raw price column)
        No dependency on cached X/feature columns.
        """
        # --- validation on raw points ---
        X_price_val = ts[["price"]].to_numpy(dtype=float)
        y_val = np.asarray(y_all, dtype=float).copy()
        w_val = np.asarray(w_all, dtype=float).copy()

        # --- aggregate for training (by price only) ---
        train_agg = self._bin_and_aggregate(
            ts[["price", self.TARGET_COL, self.WEIGHT_COL]].copy(),
            price_col="price",
            target_col=self.TARGET_COL,
            weight_col=self.WEIGHT_COL,
        )
        X_price_tr = train_agg[["price"]].to_numpy(dtype=float)
        y_tr = train_agg[self.TARGET_COL].to_numpy(dtype=float)
        w_tr = train_agg[self.WEIGHT_COL].to_numpy(dtype=float)

        # --- sanitize weights ---
        w_tr = np.clip(
            np.nan_to_num(w_tr, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None
        )
        w_val = np.clip(
            np.nan_to_num(w_val, nan=1.0, posinf=3.0, neginf=1e-6), 1e-6, None
        )

        # remember which column is price for objective() (assemble_numeric already sets this,
        # but keep a defensive fallback)
        if not hasattr(self, "_price_col"):
            self._price_col = 0

        return {
            "X_tr": X_price_tr,
            "X_val": X_price_val,
            "y_tr": y_tr,
            "y_val": y_val,
            "w_tr": w_tr,
            "w_val": w_val,
            "price_val": X_price_val[:, 0],
            "w_all": w_all,
            "y_all": y_all,
        }

    def kde_support_weights(
        self, price_val: np.ndarray, obs_prices: np.ndarray
    ) -> np.ndarray:
        """
        Estimates a smooth density at each query price via Kernel Density Estimation on the training prices
        and uses that density (or a monotone transform) as a support weight.

        CV'd KDE density at each validation price → weight in [0.25, 1.0].
        Safe if obs_prices is tiny / degenerate.
        """
        x = np.asarray(obs_prices, float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.ones_like(price_val, float)

        std = np.std(x)
        lo = max(1e-3, 0.05 * std)
        hi = max(lo * 1.01, 2.0 * std)
        params = {"bandwidth": np.logspace(np.log10(lo), np.log10(hi), 15)}
        gs = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=5, n_jobs=1)
        gs.fit(x.reshape(-1, 1))
        kde = gs.best_estimator_

        log_d = kde.score_samples(price_val.reshape(-1, 1))
        d = np.exp(log_d)
        d = (d - d.min()) / (d.ptp() or 1.0)
        return 0.25 + 0.75 * d  # map to [0.25, 1.0]

    def knn_support_weights(
        self, price_val: np.ndarray, obs_prices: np.ndarray, k: int = 15
    ) -> np.ndarray:
        """
        Scores how well each query price is supported by nearby observed prices
        in the training set using k-nearest neighbors.

        kNN radius (inverse local density) → weight in [0.25, 1.0].
        Use if you prefer a param-free option over KDE.
        """
        x = np.asarray(obs_prices, float)
        x = x[np.isfinite(x)]
        if x.size < max(3, k + 1):
            return np.ones_like(price_val, float)

        k = min(k, x.size - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(x.reshape(-1, 1))
        dists, _ = nn.kneighbors(price_val.reshape(-1, 1), n_neighbors=k)
        rad = dists[:, -1]  # kth neighbor radius
        dens = 1.0 / np.maximum(rad, np.finfo(float).eps)
        dens = (dens - dens.min()) / (dens.ptp() or 1.0)
        return 0.25 + 0.75 * dens

    def isotonic_nonneg(
        self, prices: np.ndarray, units_hat: np.ndarray, increasing: bool | None
    ) -> np.ndarray:
        """
        Post-processes a predicted units vs. price curve to enforce monotonic non-increasing shape (and non-negativity). In short: it makes demand go down or flat as price increases, never up, and clips below 0.

        Optional, zero-parameter shape enforcement over a curve.
        Set increasing=True/False or pass None to skip.
        """
        if increasing is None or units_hat.size < 3:
            return np.maximum(units_hat, 0.0)
        iso = IsotonicRegression(increasing=increasing, y_min=0.0)
        return iso.fit_transform(prices, np.maximum(units_hat, 0.0))

    def make_objective(
        self,
        split: dict,
        *,
        price_col: int | None = None,
        use_isotonic: bool = False,  # kept for signature compatibility; ignored here
    ):
        # which column is price?
        price_j = self._price_col if price_col is None else price_col

        Xtr, ytr, wtr = split["X_tr"], split["y_tr"], split["w_tr"]
        Xva, yva, wva = split["X_val"], split["y_val"], split["w_val"]

        # indices for linear (non-price) features
        p = Xtr.shape[1]
        other_idx = [j for j in range(p) if j != price_j]

        # safety on targets
        ytr = np.asarray(ytr, dtype=float)
        yva = np.asarray(yva, dtype=float)
        eps = 1e-9
        ytr = np.clip(ytr, eps, None)
        yva = np.clip(yva, eps, None)

        # build a terms object: spline on price + linear for others
        from pygam import GammaGAM, s, l

        base_terms = s(price_j)  # n_splines & lam will be set per-eval
        if other_idx:
            for j in other_idx:
                base_terms = base_terms + l(j)

        def fit_and_score(n_splines: int, lam: float) -> float:
            # fresh model each time
            gam = GammaGAM(terms=base_terms, fit_intercept=True)

            # set per-term λ: spline gets "lam", linear terms get 0 (no smoothing)
            # term order matches construction: [s(price)] + [l(...), l(...), ...]
            n_terms = 1 + len(other_idx)
            lam_vec = np.zeros(n_terms, dtype=float)
            lam_vec[0] = float(lam)

            # tell the spline how many basis functions
            gam.n_splines = [n_splines] + [1] * len(
                other_idx
            )  # linear terms effectively 1-dof
            gam.lam = lam_vec

            # fit
            gam.fit(Xtr, ytr, weights=wtr)

            # predict & deviance on val
            mu = gam.predict(Xva)
            mu = np.clip(mu, 1e-9, None)
            dev = gam.distribution.deviance(yva, mu)
            loss = float(np.average(dev, weights=wva))

            return loss

        # expose for ParamSearchCV
        def objective(ns: int, lam: float) -> float:
            return fit_and_score(int(ns), float(lam))

        return objective

    # ---------------------------- Tuning & fit ----------------------------
    def tune_auto(
        self,
        split: dict,
        *,
        price_col: int = 0,  # used only as a fallback if p_tr/p_val missing
        weighting: str | None = "knn",  # {"knn","kde",None}
        k: int = 15,
        kde_bandwidth: str | float = "scott",
        use_isotonic: bool = False,
    ) -> dict:
        import numpy as np

        # --- pull arrays ---
        X_tr, X_val = split["X_tr"], split["X_val"]
        # Prefer RAW prices carried in the split; fallback to X[:, price_col]
        p_tr = split.get("p_tr")
        p_val = split.get("p_val")
        if p_tr is None:
            p_tr = X_tr[:, price_col]
        if p_val is None:
            p_val = X_val[:, price_col]

        # tiny debug (helps catch accidental standardization)
        log = getattr(self, "_log", lambda *a, **k: None)
        try:
            log(
                f"[DBG] price(raw) train range = [{float(np.min(p_tr)):.2f}, {float(np.max(p_tr)):.2f}] | "
                f"val range = [{float(np.min(p_val)):.2f}, {float(np.max(p_val)):.2f}]"
            )
        except Exception:
            pass

        # --- add support weights only if missing ---
        if weighting in {"knn", "kde"} and (
            split.get("w_tr") is None or split.get("w_val") is None
        ):
            if weighting == "knn":
                w_tr = self.knn_support_weights(p_tr, p_tr, k=k)
                w_val = self.knn_support_weights(p_val, p_tr, k=k)
            else:  # "kde"
                w_tr = self.kde_support_weights(p_tr, bandwidth=kde_bandwidth)
                w_val = self.kde_support_weights(
                    p_val, fit_on=p_tr, bandwidth=kde_bandwidth
                )
            split = dict(split)  # avoid mutating caller's dict
            split["w_tr"], split["w_val"] = w_tr, w_val

        # --- objective closure (uses p_tr/p_val inside make_objective) ---
        objective = self.make_objective(split, use_isotonic=False)

        # --- coarse search (ParamSearchCV has sane defaults) ---
        ps = self.ParamSearchCV(
            objective=objective, logger_print=getattr(self, "_log", None)
        ).fit()
        best_ns, best_lam = ps.best_ns_, ps.best_lam_
        log(
            f"[TUNING] coarse best ns={best_ns} λ={best_lam:g} loss={ps.best_score_:.6f}"
        )

        # --- small refine around the winner ---
        ns_ref = sorted({max(8, best_ns - 2), best_ns, min(32, best_ns + 2)})
        lam_ref = np.geomspace(max(best_lam / 4.0, 1e-5), best_lam * 4.0, 8)

        ps2 = self.ParamSearchCV(
            n_splines_grid=ns_ref,
            n_lam=len(lam_ref),
            lam_floor_init=lam_ref[0],
            lam_span_decades=np.log10(lam_ref[-1] / lam_ref[0]),
            objective=objective,
            logger_print=getattr(self, "_log", None),
        )
        ps2.lam_grid = lam_ref  # use exact vector
        ps2.fit()

        return {
            "n_splines": ps2.best_ns_,
            "lam": ps2.best_lam_,
            "loss": ps2.best_score_,
        }

    def fit_full(self, ts, y_all, w_all, ns_star: int, lam_star: float):
        modeler = self.GAMModeler(
            feature_cols=["price"],
            base_gam_kwargs={"lam": float(lam_star), "n_splines": int(ns_star)},
            monotone_default="down",  # enforce non-increasing units vs price
            median_filter_window=None,  # keep off by default
        )
        modeler.fit(
            train_df=ts[modeler.feature_cols],
            y_train=y_all,
            weights=w_all,
            verbose=True,
        )
        return modeler

    # ---------------------------- Results assembly ----------------------------

    @staticmethod
    def assemble_results(topsellers: pd.DataFrame, modeler) -> pd.DataFrame:
        return (
            topsellers[["product", "price", "asin", "asp"]]
            .copy()
            .reset_index(drop=True)
        )

    def add_support_counts(
        self, topsellers: pd.DataFrame, all_gam_results: pd.DataFrame
    ) -> pd.DataFrame:
        base_df = getattr(self.engineer, "pricing_df", None)
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
        return all_gam_results

    def add_predictions(self, modeler, df: pd.DataFrame) -> pd.DataFrame:
        return modeler.add_predictions_to_df(
            df,
            write_units=True,
            write_revenue=True,
            price_col="price",
            inplace=False,
        )

    def passthrough_actuals(
        self, topsellers: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        df["deal_discount_percent"] = (
            topsellers["deal_discount_percent"]
            .fillna(0)
            .clip(lower=0)
            .reset_index(drop=True)
        )
        df["revenue_actual"] = topsellers["price"].to_numpy() * np.asarray(
            topsellers["shipped_units"], dtype=float
        )
        df["daily_rev"] = df["revenue_actual"]
        df["actual_revenue_scaled"] = (
            topsellers["price"].to_numpy()
            * (1 - df["deal_discount_percent"].to_numpy() / 100.0)
            * np.asarray(topsellers["shipped_units"], dtype=float)
        )
        return df


class PricingPipeline:
    def __init__(
        self,
        pricing_df: pd.DataFrame,
        product_df: pd.DataFrame,
        top_n: int = 10,
        param_search_kwargs=None,  # kept for backward-compat; ignored
        logger_print=None,  # optional explicit logger
        *,
        # DI hooks (defaults to your concrete types)
        Weighting=Weighting,
        ElasticityAnalyzer=ElasticityAnalyzer,
        GAMModeler=GAMModeler,
        ParamSearchCV=ParamSearchCV,
    ):
        """
        Classic constructor: no 'engineer' arg. We build it internally.
        - pricing_df, product_df: raw inputs
        - top_n: passed into DataEngineer
        - param_search_kwargs: accepted (legacy) but ignored; tuning is controlled in ParamSearchCV.__init__ defaults
        """
        # logger
        self._log = logger_print if logger_print is not None else print

        # store raw inputs (handy for downstream/reporting)
        self.pricing_df = pricing_df
        self.product_df = product_df

        # build the engineer internally (as before)
        self.engineer = DataEngineer(pricing_df, product_df, top_n)

        # wire the extracted core with collaborators/constants
        self.core = PipelineCore(
            engineer=self.engineer,
            feat_cols=FEAT_COLS,  # e.g. ["price"] or your configured list
            target_col=TARGET_COL,  # e.g. "shipped_units"
            weight_col=WEIGHT_COL,  # your constant already used elsewhere
            logger_print=self._log,
            Weighting=Weighting,
            ElasticityAnalyzer=ElasticityAnalyzer,
            GAMModeler=GAMModeler,
            ParamSearchCV=ParamSearchCV,
        )
        # self.core.set_engineer(self.engineer)  # <-- one line and done

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
        Orchestrates:
        1) data prep → weights → numeric frames
        2) univariate price-GAM tuning (one-call ParamSearchCV)
        3) full refit & prediction assembly
        Returns:
        topsellers, elasticity_df, all_gam_results
        """
        print(
            "\n─────────────────────────────── 🧮 Starting Data Engineering ───────────────────────────────"
        )

        # 1) Prep
        topsellers = self.core.prepare_topsellers()
        if topsellers.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        self.core.ensure_columns(topsellers)
        print("✅ Data loaded & preprocessed. Proceeding to weight computation...")

        w_stable = self.core.compute_weights(topsellers)
        topsellers[WEIGHT_COL] = w_stable
        print(
            f"⚖️  Weights computed | median={np.nanmedian(w_stable):.3f} | p95={np.nanpercentile(w_stable,95):.3f}"
        )

        ts, X, y, w = self.core.assemble_numeric(topsellers)
        print(
            f"⚖️  Weights ready | median={float(np.nanmedian(w)):.3f} | p95={float(np.nanpercentile(w,95)):.3f} | n={len(w):,}"
        )

        elasticity_df = self.core.elasticity_best_effort(topsellers)

        # standardize continuous features in place (no need to keep Z/mu/sd around here)
        p_raw = ts["price"].to_numpy().astype(float)  # keep this safe
        self.core.standardize_continuous_inplace(X)  # may standardize price

        # 2) Tuning
        print("\n\n" + 35 * "- " + " 🤖 Modeling, Tuning & Prediction " + "- " * 35)
        if "price" not in ts.columns:
            raise ValueError(
                "Expected 'price' column to exist for univariate GAM tuning."
            )

        ts, X, y, w = self.core.assemble_numeric(topsellers)

        # If DE already standardizes, skip this; otherwise keep but limit to continuous cols:
        # self.core.standardize_continuous_inplace(X, cols=("price","discount_pct"))

        split = self.core.train_val_split(ts, y, w)

        objective = self.core.make_objective(
            split, price_col=None, use_isotonic=False
        )  # None → uses cached _price_col
        best = self.core.tune_auto(
            split, price_col=None, weighting=None, use_isotonic=False
        )

        # # split already built by your train_val_split(ts, y, w)
        # split = self.core.train_val_split(ts, y, w)

        # # one-call tuning (no manual grids, no isotonic)
        # best = self.core.tune_auto(
        #     split, price_col=0, weighting="knn", use_isotonic=False
        # )

        ns_star, lam_star = int(best["n_splines"]), float(best["lam"])
        self._log(f"[TUNING] best ns={ns_star} | best λ={lam_star:g}")

        # 3) Full fit & predictions (GAMModeler has smooth-positive outputs via softplus)
        modeler = self.core.fit_full(ts, y, w, ns_star, lam_star)

        all_gam_results = self.core.assemble_results(topsellers, modeler)
        all_gam_results = self.core.add_support_counts(topsellers, all_gam_results)
        all_gam_results = self.core.add_predictions(modeler, all_gam_results)
        all_gam_results = self.core.passthrough_actuals(topsellers, all_gam_results)
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

        df = all_gam_results.copy()

        # ---- NEW: normalize schema so the plotting code works with either old or new names ----
        rename = {}

        # X-axis
        if "asp" not in df.columns and "price" in df.columns:
            rename["price"] = "asp"

        # Predictions
        has_new_preds = {"rev_p50", "rev_p025", "rev_p975"} <= set(df.columns)
        if has_new_preds:
            rename.update(
                {
                    "rev_p50": "revenue_pred_0.5",
                    "rev_p025": "revenue_pred_0.025",
                    "rev_p975": "revenue_pred_0.975",
                }
            )
        # Some variants I’ve seen in older runs
        if "revenue_pred_p50" in df.columns:
            rename["revenue_pred_p50"] = "revenue_pred_0.5"
        if "revenue_pred_p025" in df.columns:
            rename["revenue_pred_p025"] = "revenue_pred_0.025"
        if "revenue_pred_p975" in df.columns:
            rename["revenue_pred_p975"] = "revenue_pred_0.975"

        # Actuals
        if "revenue_actual" not in df.columns:
            if "actual_revenue" in df.columns:
                rename["actual_revenue"] = "revenue_actual"
            elif "revenue" in df.columns:
                rename["revenue"] = "revenue_actual"

        # the rest of your current function can stay the same
        need_cols = [
            "product",
            "asp",
            "revenue_actual",
            "revenue_pred_0.025",
            "revenue_pred_0.5",
            "revenue_pred_0.975",
        ]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"gam_results: missing columns after normalization: {missing}"
            )

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
