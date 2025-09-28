# --------- built_in_logic.py  ---------
# (RMSE-focused; Top-N only; adds annualized opps & data range)
from __future__ import annotations
import random
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Sequence, Tuple, Dict, Any, Iterable

# viz
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

# ML
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from pygam import ExpectileGAM,s,l
from sklearn.ensemble import GradientBoostingRegressor

# local import
from helpers import DataEng

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


EXPECTED_COLS = ["price", "deal_discount_percent", "event_encoded", "product_encoded", "asp", "__intercept__",]
CAT_COLS      = ["event_encoded", "product_encoded"]
NUM_COLS      = ["price", "deal_discount_percent"]


def _v_best(obj, msg: str):
    """
    Logging func -
    Print only when obj._verbose is True. Use for 'best only' logs.
    """
    if getattr(obj, "_verbose", False):
        from datetime import datetime

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] {obj.__class__.__name__} {msg}",
            flush=True,
        )


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

    def __init__(self, pricing_df, product_df, top_n=10):
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n

    @staticmethod
    def _nonneg(s: pd.Series) -> pd.Series:
        """Coerce to numeric and clip at 0."""
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        return s.clip(lower=0)

    def _days_at_price(self, df) -> pd.DataFrame:
        """
        add a column for the number of days where an ASP was sold at

        param
            df

        return
            df [with a new "number of days for a price" column]
        """

        days_at_asp = (
            df[["asin", "order_date", "price"]]
            .groupby(["asin", "price"])
            .nunique()
            .reset_index()
        )
        days_at_asp.rename(columns={"order_date": "days_sold"}, inplace=True)

        res = df.merge(
            days_at_asp, left_on=["asin", "price"], right_on=["asin", "price"]
        )

        return res

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

    def prepare(self) -> pd.DataFrame:
        """
        Prepare merged & cleaned Top-N product rows for modeling.
        - normalizes columns
        - merges pricing/product
        - coerces types & nonneg
        - (optional) synthesize order_date (commented)
        - expands days_sold, computes daily revenue/units
        - filters Top-N products by total revenue
        - encodes categories
        Returns: DataFrame ready for downstream modeling.
        """
        # 1) normalize column names
        self.pricing_df = DataEng.clean_cols(self.pricing_df)
        self.product_df = DataEng.clean_cols(self.product_df)

        # 2) merge on asin
        df = self.pricing_df.merge(self.product_df, how="left", on="asin")

        # 3) product label for grouping
        df["product"] = DataEng.compute_product_series(df)

        # 4) drop unused if present
        for c in ("tag", "variation"):
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        # 5) types & non-neg
        df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")
        if "shipped_units" in df.columns:
            df["shipped_units"] = self._nonneg(df["shipped_units"])
        if "price" in df.columns:
            df["price"] = self._nonneg(df["price"])

        # 6) (optional) synthesize order_date across a window (keep commented unless you need it)
        # df = self._synthesize_order_dates(
        #     df, start="2023-09-17", end="2025-09-17", seed=42, col="order_date"
        # )

        # persist back for range/meta to use identical dates
        self.pricing_df = df.copy()

        # 7) days at ASP & actual revenue
        df_days_asp = self._days_at_price(df)
        df_days_asp["revenue"] = (
            df_days_asp["shipped_units"] * df_days_asp["price"]
        ).clip(lower=0)

        # 8) daily aggregation by asin-event-date
        df_agg = (
            df_days_asp[
                ["asin", "event_name", "order_date", "shipped_units", "revenue"]
            ]
            .groupby(["asin", "event_name", "order_date"])[["revenue", "shipped_units"]]
            .sum()
            .reset_index()
        )
        df_agg["price"] = (df_agg["revenue"] / df_agg["shipped_units"]).replace(
            [np.inf, -np.inf], np.nan
        )
        df_agg["price"] = (
            pd.to_numeric(df_agg["price"], errors="coerce").fillna(0).round(2)
        )

        df_agg_event = df_agg.merge(
            df_days_asp[
                [
                    "asin",
                    "product",
                    "order_date",
                    "deal_discount_percent",
                    "current_price",
                    "days_sold",
                ]
            ],
            on=["asin", "order_date"],
            how="left",
        )

        # 9) derive daily revenue/units (guard days_sold==0)
        den = pd.to_numeric(df_agg_event["days_sold"], errors="coerce").replace(
            0, np.nan
        )
        df_agg_event["daily_rev"] = (
            pd.to_numeric(df_agg_event["revenue"], errors="coerce") / den
        ).clip(lower=0)
        df_agg_event["daily_units"] = (
            pd.to_numeric(df_agg_event["shipped_units"], errors="coerce") / den
        ).clip(lower=0)
        df_agg_event.drop(columns=["revenue", "shipped_units"], inplace=True)
        df_agg_event.rename(
            columns={"daily_rev": "revenue", "daily_units": "shipped_units"},
            inplace=True,
        )

        # 10) Top-N products by total revenue
        top_n_products = (
            df_agg_event.groupby("product")["revenue"]
            .sum()
            .reset_index()
            .sort_values("revenue", ascending=False)["product"]
            .head(self.top_n)
            .tolist()
        )

        # 11) filter to Top-N; normalize dtypes; encode categories
        filtered = df_days_asp[df_days_asp["product"].isin(top_n_products)].copy()
        filtered["asin"] = filtered["asin"].astype(str)
        filtered.rename(columns={"revenue": "revenue_share_amt"}, inplace=True)
        filtered["revenue_share_amt"] = self._nonneg(filtered["revenue_share_amt"])

        res = self._label_encoder(filtered)

        return res


class Weighting:
    """
    Encapsulates recency (time-decay) + rarity weighting.
    All methods accept/return pandas objects or numpy arrays and are robust
    to missing columns.
    """
    def __init__(self, decay_rate: float = -0.01,
                 rarity_smooth: int = 5,
                 rarity_cap: float = 1.25,
                 rarity_beta: float = 0.35,
                 rarity_tails_only: bool = True,
                 rarity_tail_q: float = 0.10,
                 clip_min: float = 0.25, clip_max: float = 2.5):
        self.decay_rate = float(decay_rate)
        self.rarity_smooth = int(rarity_smooth)
        self.rarity_cap = float(rarity_cap)
        self.rarity_beta = float(rarity_beta)
        self.rarity_tails_only = bool(rarity_tails_only)
        self.rarity_tail_q = float(rarity_tail_q)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    # ----- helpers -----
    def _time_decay(self, df: pd.DataFrame) -> np.ndarray:
        """
        Exponential decay based on 'order_date' (if present). Missing dates → weight 1.0.
        """
        d = df.copy()
        if "order_date" in d.columns:
            odt = pd.to_datetime(d["order_date"], errors="coerce")
        else:
            odt = pd.Series(pd.NaT, index=d.index)
        today_ref = pd.Timestamp.now().normalize()
        days = (today_ref - odt).dt.days.astype("float64")
        days = np.where(np.isfinite(days), days, 0.0)
        return np.exp(self.decay_rate * days)

    def _rarity_multiplier(self, prices: np.ndarray,
                           bin_width: float | None = None) -> np.ndarray:
        """
        Assigns higher weights to low-density price regions.
        Robust to constant arrays and NaNs.
        """
        p = np.asarray(prices, dtype=float).ravel()
        mask = np.isfinite(p)
        if p.size == 0 or np.sum(mask) < 3 or np.allclose(p[mask], p[mask][0]):
            return np.ones_like(p, dtype=float)

        p = np.where(np.isfinite(p), p, np.nan)
        p_valid = p[np.isfinite(p)]
        n = p_valid.size

        # optional "tails-only" band
        in_core = np.zeros_like(p, dtype=bool)
        if self.rarity_tails_only:
            lo, hi = np.nanquantile(p, [self.rarity_tail_q, 1.0 - self.rarity_tail_q])
            in_core = (p >= lo) & (p <= hi)

        # Freedman–Diaconis bin width with guardrails
        if bin_width is None:
            q25, q75 = np.nanpercentile(p_valid, [25, 75])
            iqr = max(q75 - q25, 1e-9)
            bw = 2.0 * iqr / np.cbrt(n)
            span = max(np.nanmax(p_valid) - np.nanmin(p_valid), 1e-6)
            bw = np.clip(bw, span / 80.0, span / 25.0)
        else:
            bw = float(bin_width)

        edges = np.arange(np.nanmin(p_valid) - 1e-9, np.nanmax(p_valid) + bw + 1e-9, bw)
        counts, _ = np.histogram(p_valid, bins=edges)

        # smooth counts (moving average) to avoid single-bin spikes
        if self.rarity_smooth and self.rarity_smooth > 1:
            k = self.rarity_smooth + (self.rarity_smooth % 2 == 0)  # force odd
            kernel = np.ones(k, dtype=float) / k
            counts = np.convolve(counts, kernel, mode="same")

        counts = np.maximum(counts, 1e-12)
        idx = np.clip(np.digitize(np.nan_to_num(p_valid, nan=edges[0]), edges) - 1, 0, len(counts) - 1)

        # map counts back to full vector
        c_full = np.full_like(p, np.nan, dtype=float)
        c_full[np.isfinite(p)] = counts[idx]

        raw = (1.0 / c_full) ** self.rarity_beta
        raw = np.where(np.isfinite(raw), raw, 1.0)
        raw /= np.nanmean(raw) if np.isfinite(np.nanmean(raw)) else 1.0
        mult = np.clip(raw, 1.0 / self.rarity_cap, self.rarity_cap)

        # disable rarity inside the core band if requested
        if self.rarity_tails_only:
            mult[in_core] = 1.0

        mult = np.where(np.isfinite(mult), mult, 1.0)
        return mult

    # ----- main entry -----
    def _make_weights(self, df: pd.DataFrame,
                      base_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Weighted combination: time decay * rarity (on 'price'), then clipped & normalized.
        """
        decay = self._time_decay(df)
        prices = pd.to_numeric(df.get("price", np.nan), errors="coerce").to_numpy()
        rarity = self._rarity_multiplier(prices)

        w = decay * rarity
        if base_weights is not None:
            w = w * np.asarray(base_weights, dtype=float)

        # clip & normalize
        w = np.nan_to_num(w, nan=1.0, posinf=self.clip_max, neginf=self.clip_min)
        w = np.clip(w, self.clip_min, self.clip_max)
        mean = w.mean() if w.size else 1.0
        w = w / (mean if mean else 1.0)
        return w.astype(float)


class ParamSearchCV:
    """
    Single-expectile tuner:
    - preprocessing (ColumnTransformer: scale numerics, OHE categoricals)
    - per-fold weights via Weighting
    - CV over (lam, n_splines)
    - fits ONE ExpectileGAM (for self.expectile) and returns it
    """
    def __init__(
        self,
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        n_splits: int = 3,
        n_splines_grid: Tuple[int, ...] = (12, 16, 20, 24),
        loglam_range: Tuple[float, float] = (np.log(30.0), np.log(5000.0)),
        lam_iters: int = 8,
        expectile: float = 0.50,
        random_state: int = 42,
        verbose: bool = False,
        weighting: Optional[Weighting] = None,
    ):
        self.numeric_cols = list(numeric_cols or NUM_COLS)
        self.categorical_cols = list(categorical_cols or CAT_COLS)

        # preprocessors
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.scaler = StandardScaler()
        self.ct = ColumnTransformer(
            transformers=[
                ("num", self.scaler, self.numeric_cols),
                ("cat", self.ohe, self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._fitted = False

        # search params
        self.n_splits = int(n_splits)
        self.initial_n_splines = tuple(int(x) for x in n_splines_grid)
        self.initial_loglam_range = (float(loglam_range[0]), float(loglam_range[1]))
        self.lam_iters = int(lam_iters)
        self.expectile = float(expectile)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        # weighting
        self.weighting = weighting or Weighting()

        # artifacts
        self.final_model_ = None
        self.final_cfg_: Optional[Dict[str, Any]] = None

        # optional: external elasticity hints
        self.elasticity_scores: Dict[Any, float] = {}

    # ----- dataframe & transform helpers -----
    def _ensure_dataframe(self, X, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            assert feature_names is not None, "feature_names required for ndarray X"
            df = pd.DataFrame(np.asarray(X), columns=list(feature_names))
        for c in self.categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype("string")
        return df

    def _preprocess_features(self, X, fit: bool, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        df = self._ensure_dataframe(X, feature_names or EXPECTED_COLS)
        Xt = self.ct.fit_transform(df) if fit else self.ct.transform(df)
        self._fitted = self._fitted or fit
        return Xt

    def transform(self, X_pred) -> np.ndarray:
        """Use already-fitted transformers; do NOT refit."""
        return self._preprocess_features(X_pred, fit=False, feature_names=EXPECTED_COLS)

    # ----- core one-expectile fit (stable) -----
    def _fit_one_expectile_core(
        self,
        X_tr, y_tr, w_tr, X_va,
        *,
        expectile: float,
        lam: float = 300.0,
        n_splines: int = 20,
        max_iter: int = 4000,
        tol: float = 1e-3,
    ) -> Tuple[np.ndarray, Any]:
        """
        Fit ONE ExpectileGAM and predict on X_va.
        Stabilized with y z-scoring, weight hygiene, lam-retry, const-column drop.
        Always returns (y_hat_va, model).
        """
        # arrays
        X_tr = np.asarray(X_tr, dtype=float); X_va = np.asarray(X_va, dtype=float)
        y_tr = np.asarray(y_tr, dtype=float)

        # remove constant columns (OHE on tiny folds)
        std = X_tr.std(axis=0)
        keep = std > 0
        if not np.all(keep):
            X_tr = X_tr[:, keep]
            X_va = X_va[:, keep]

        # weights
        if w_tr is None:
            w_tr = np.ones(X_tr.shape[0], dtype=float)
        else:
            w_tr = np.asarray(w_tr, dtype=float)
            lo, hi = np.quantile(w_tr, [0.01, 0.99])
            w_tr = np.clip(w_tr, lo, hi)
            w_tr = w_tr / (w_tr.mean() or 1.0)

        # scale y
        y_mu = float(y_tr.mean()); y_sd = float(y_tr.std()) or 1.0
        y_tr_s = (y_tr - y_mu) / y_sd

        # terms: smooth for first 2 cols (scaled price/discount), linear for the rest (OHE)
        n_feats = X_tr.shape[1]
        terms = s(0, n_splines=int(n_splines)) + s(1, n_splines=int(n_splines))
        for j in range(2, n_feats):
            terms += l(j)

        # main attempts
        lam_try = float(lam)
        for _attempt in range(3):
            gam = ExpectileGAM(
                terms=terms,
                expectile=float(expectile),
                lam=lam_try,
                max_iter=int(max_iter),
                tol=float(tol),
            )
            try:
                gam.fit(X_tr, y_tr_s, weights=w_tr)
                y_hat_va = gam.predict(X_va) * y_sd + y_mu
                return y_hat_va, gam
            except Exception:
                lam_try *= 10.0  # smooth more & retry

        # last resort: linear-only
        try:
            t_lin = l(0)
            for j in range(1, n_feats):
                t_lin += l(j)
            gam_lin = ExpectileGAM(
                terms=t_lin, expectile=float(expectile),
                lam=1e6, max_iter=8000, tol=5e-3
            )
            gam_lin.fit(X_tr, y_tr_s, weights=w_tr)
            y_hat_va = gam_lin.predict(X_va) * y_sd + y_mu
            return y_hat_va, gam_lin
        except Exception:
            pass

        # constant predictor
        class _Const:
            def __init__(self, c, q): self._c=float(c); self.expectile=float(q)
            def predict(self, X): X=np.asarray(X); return np.full(X.shape[0], self._c, float)
        const = _Const(y_mu, expectile)
        return np.full(X_va.shape[0], y_mu, float), const

    # ----- CV scorer (RMSE on the single expectile) -----
    def _cv_score(self, X, y, w=None, cfg=None,
                  n_splits: Optional[int] = None,
                  random_state: Optional[int] = None) -> float:
        cfg = cfg or {}
        lam = float(cfg.get("lam", 300.0))
        n_splines = int(cfg.get("n_splines", 20))
        q = float(cfg.get("expectile", self.expectile))

        X_df = self._ensure_dataframe(X, EXPECTED_COLS)
        y = np.asarray(y)
        base_w = None if w is None else np.asarray(w)

        kf = KFold(
            n_splits=n_splits or self.n_splits,
            shuffle=True,
            random_state=self.random_state if random_state is None else random_state,
        )
        rmses = []
        for tr_idx, va_idx in kf.split(X_df):
            X_tr_df, X_va_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # per-fold weights from train fold only
            w_tr = self.weighting._make_weights(X_tr_df, base_w[tr_idx] if base_w is not None else None)
            w_va = None if base_w is None else base_w[va_idx]

            X_tr = self._preprocess_features(X_tr_df, fit=True)
            X_va = self._preprocess_features(X_va_df, fit=False)

            y_hat, _ = self._fit_one_expectile_core(
                X_tr, y_tr, w_tr, X_va,
                expectile=q, lam=lam, n_splines=n_splines
            )
            mse = mean_squared_error(y_va, y_hat, sample_weight=w_va)
            rmses.append(float(np.sqrt(mse)))
        return float(np.mean(rmses)) if rmses else float("inf")

    # ----- golden-section search over log(lam) -----
    def _golden_search_loglam(self, X, y, w, *, base_cfg: Dict[str, Any],
                              loglam_range: Tuple[float, float] | None = None,
                              max_iters: Optional[int] = None) -> Tuple[Dict[str, Any], float]:
        if loglam_range is None:
            loglam_range = self.initial_loglam_range
        if max_iters is None:
            max_iters = self.lam_iters

        # guardrails
        lo = max(np.log(10.0), float(loglam_range[0]))
        hi = min(np.log(1e5), float(loglam_range[1]))

        phi = (1 + 5**0.5) / 2
        invphi = 1 / phi
        a, b = lo, hi
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi

        cfg_c = dict(base_cfg, lam=float(np.exp(c)))
        sc_c = self._cv_score(X, y, w, cfg=cfg_c)
        cfg_d = dict(base_cfg, lam=float(np.exp(d)))
        sc_d = self._cv_score(X, y, w, cfg=cfg_d)

        for _ in range(max(max_iters - 1, 0)):
            if sc_c <= sc_d:
                b, d, sc_d = d, c, sc_c
                c = b - (b - a) * invphi
                cfg_c = dict(base_cfg, lam=float(np.exp(c)))
                sc_c = self._cv_score(X, y, w, cfg=cfg_c)
            else:
                a, c, sc_c = c, d, sc_d
                d = a + (b - a) * invphi
                cfg_d = dict(base_cfg, lam=float(np.exp(d)))
                sc_d = self._cv_score(X, y, w, cfg=cfg_d)

        return (cfg_c, sc_c) if sc_c <= sc_d else (cfg_d, sc_d)

    # ----- adaptive grid over n_splines with elasticity hint (optional) -----
    def _adaptive_grid_search(self, X, y, sample_weight=None) -> Tuple[Optional[Dict[str, Any]], float]:
        X_df = self._ensure_dataframe(X, EXPECTED_COLS)
        unique_products = X_df["product_encoded"].unique() if "product_encoded" in X_df else []
        es = self.elasticity_scores or {}
        mean_elasticity = float(np.mean([es.get(p, 0.5) for p in unique_products])) if len(unique_products) else 0.5

        # adjust search ranges
        if mean_elasticity > 0.7:
            n_splines_grid = (24, 28, 32)
            loglam_range = (np.log(20.0), np.log(1000.0))
        elif mean_elasticity < 0.3:
            n_splines_grid = (10, 14, 18)
            loglam_range = (np.log(200.0), np.log(5000.0))
        else:
            n_splines_grid = self.initial_n_splines
            loglam_range = self.initial_loglam_range

        current_best_cfg = None
        current_best_score = np.inf
        initial_lam_iters = max(4, self.lam_iters // 2)

        # phase 1
        for ns in n_splines_grid:
            base_cfg = {"n_splines": int(ns), "expectile": self.expectile}
            cfg, score = self._golden_search_loglam(
                X, y, sample_weight, base_cfg=base_cfg,
                loglam_range=loglam_range, max_iters=initial_lam_iters
            )
            if score < current_best_score:
                current_best_score = score
                current_best_cfg = cfg
                if self.verbose:
                    print(f"Improvement: New score {score:.4f} @ {cfg}")

        # phase 2 (refine around best)
        if current_best_cfg is not None:
            best_ns = int(current_best_cfg["n_splines"])
            best_lam_log = float(np.log(current_best_cfg["lam"]))
            ns_step = 5
            lam_window = 0.5
            refined_ns = [max(8, best_ns - ns_step), best_ns, min(36, best_ns + ns_step)]
            refined_loglam = (best_lam_log - lam_window, best_lam_log + lam_window)

            for ns in refined_ns:
                base_cfg = {"n_splines": int(ns), "expectile": self.expectile}
                cfg, score = self._golden_search_loglam(
                    X, y, sample_weight,
                    base_cfg=base_cfg, loglam_range=refined_loglam, max_iters=self.lam_iters
                )
                if score < current_best_score:
                    current_best_score = score
                    current_best_cfg = cfg
                    if self.verbose:
                        print(f"Refine: New score {score:.4f} @ {cfg}")

        return current_best_cfg, current_best_score

    # ----- public fit -----
    def fit(self, X, y, sample_weight=None, verbose=False):
        self.verbose = verbose
        best_cfg, _ = self._adaptive_grid_search(X, y, sample_weight)
        if best_cfg is None:
            return None, None

        X_df = self._ensure_dataframe(X, EXPECTED_COLS)
        full_w = self.weighting._make_weights(X_df, sample_weight)

        X_full = self._preprocess_features(X_df, fit=True)
        y_full = np.asarray(y)

        _, model = self._fit_one_expectile_core(
            X_full, y_full, full_w, X_full,
            expectile=float(best_cfg.get("expectile", self.expectile)),
            lam=float(best_cfg["lam"]),
            n_splines=int(best_cfg["n_splines"]),
        )

        self.final_model_ = model
        self.final_cfg_ = {
            "lam": float(best_cfg["lam"]),
            "n_splines": int(best_cfg["n_splines"]),
            "expectile": float(best_cfg.get("expectile", self.expectile)),
        }
        return model, self.final_cfg_


class GAMModeler:
    """
    Thin orchestrator:
    - ensures 4-column schema
    - loops over expectiles and delegates to ParamSearchCV
    - transforms X_pred via param_search.transform() and predicts
    """
    def __init__(self, param_search: ParamSearchCV):
        self.param_search = param_search

    def _to_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X[EXPECTED_COLS].copy()
        else:
            df = pd.DataFrame(np.asarray(X), columns=EXPECTED_COLS)
        for c in CAT_COLS:
            df[c] = df[c].astype("string")
        return df

    def fit_predict_expectiles(
        self,
        X_train,
        y_train,
        X_pred,
        w_train=None,
        expectiles=(0.025, 0.50, 0.975),
    ):
        """
        Fit quantile regressors (one per expectile) and predict units for X_pred.
        Returns:
            {"predictions": {f"units_pred_{q}": np.array shape (n_pred,) }}
        """
        # ---- 1) Prepare data (numeric-only, stable scaling) ----
        X_tr = np.asarray(X_train, dtype=float)
        X_pr = np.asarray(X_pred, dtype=float)
        y_tr = np.asarray(y_train, dtype=float)

        # Guard against NaNs/Infs
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_pr = np.nan_to_num(X_pr, nan=0.0, posinf=0.0, neginf=0.0)
        y_tr = np.nan_to_num(y_tr, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional sample weights
        sw = None
        if w_train is not None:
            sw = np.asarray(w_train, dtype=float)
            sw = np.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0)
            # floor tiny weights so the model doesn't learn a trivial zero solution
            if np.isfinite(sw).any():
                q20 = np.nanquantile(sw[np.isfinite(sw)], 0.20) if np.isfinite(sw).sum() else 0.0
                floor = 0.1 if (not np.isfinite(q20) or q20 <= 1e-6) else float(q20)
                sw = np.clip(sw, floor, None)
            else:
                sw = np.ones_like(y_tr, dtype=float)

        # Standardize features to a comparable scale (helps trees split sensibly on price)
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_trs = scaler.fit_transform(X_tr)
        X_prs = scaler.transform(X_pr)

        # ---- 2) Train one quantile model per expectile ----
        preds = {}
        # Keep reasonably small trees to avoid overfit; you can tune these defaults
        base_kwargs = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=10,
            subsample=0.9,
            random_state=42,
        )

        for q in expectiles:
            # Map expectile to quantile (same numeric value here)
            alpha = float(q)
            model = GradientBoostingRegressor(loss="quantile", alpha=alpha, **base_kwargs)
            model.fit(X_trs, y_tr, sample_weight=sw)
            yhat = model.predict(X_prs)
            # units must be non-negative
            yhat = np.clip(yhat, 0.0, None)
            preds[f"units_pred_{q}"] = yhat

        return {"predictions": preds}


class Optimizer:
    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        """Calculate weighted predictions based on ratio"""
        if "ratio" in all_gam_results.columns:
            max_ratio = all_gam_results["ratio"].max()
            confidence_weight = 1 - (all_gam_results["ratio"] / max_ratio)
        else:
            confidence_weight = 0.5

        # Calculate weighted prediction
        all_gam_results["weighted_pred"] = all_gam_results[
            "units_pred_0.5"
        ] * confidence_weight + all_gam_results["units_pred_avg"] * (
            1 - confidence_weight
        )

        return {
            "best_weighted": DataEng.pick_best_by_group(
                all_gam_results, "product", "weighted_pred"
            ),
            "best_avg": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_avg"
            ),
            "best50": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_0.5"
            ),
            "best975": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_0.975"
            ),
            "best25": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_0.025"
            ),
            "all_gam_results": all_gam_results,
        }


class PricingPipeline:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.engineer = DataEngineer(pricing_df, product_df, top_n)
        self.param_search = ParamSearchCV(
            n_splits=4,
            n_splines_grid=(15, 20, 25, 30),
            loglam_range=(np.log(0.05), np.log(20.0)),
            expectile=0.5,
            random_state=42,
            verbose=False,
        )

    @classmethod
    def from_csv_folder(
        cls,
        base_dir,
        data_folder="data",
        pricing_file="pricing.csv",
        product_file="products.csv",
        top_n=10,
    ):
        pricing_df = pd.read_csv(os.path.join(base_dir, data_folder, pricing_file))
        product_df = pd.read_csv(os.path.join(base_dir, data_folder, product_file))
        return cls(pricing_df, product_df, top_n).assemble_dashboard_frames()

    def _build_curr_price_df(self) -> pd.DataFrame:
        """current price df with product tag"""
        if "current_price" not in self.engineer.product_df.columns:
            return pd.DataFrame(columns=["asin", "product", "current_price"])

        product = self.engineer.product_df.copy()
        product["product"] = DataEng.compute_product_series(product)
        out = product[["asin", "product", "current_price"]].copy()
        out["current_price"] = pd.to_numeric(out["current_price"], errors="coerce")
        return out.reset_index(drop=True)

    def _build_core_frames(self):
        """
        Build core frames: topsellers (engineered rows), elasticity_df, all_gam_results (predictions + viz columns)
        """
        # 1) Base table and required aliases
        topsellers = self.engineer.prepare()
        if 'asp' not in topsellers.columns and 'price' in topsellers.columns:
            topsellers['asp'] = pd.to_numeric(topsellers['price'], errors='coerce')
        if '__intercept__' not in topsellers.columns:
            topsellers['__intercept__'] = 1.0

        # 2) Weights for training / elasticity
        topsellers['time_weight'] = self.param_search.weighting._time_decay(topsellers)
        w_raw = np.asarray(self.param_search.weighting._make_weights(topsellers), dtype=float)
        if not np.isfinite(w_raw).any() or np.nansum(w_raw) <= 0:
            w_stable = np.ones(len(topsellers), dtype=float)
        else:
            finite_w = w_raw[np.isfinite(w_raw)]
            if finite_w.size == 0:
                w_stable = np.ones(len(topsellers), dtype=float)
            else:
                floor = np.nanquantile(finite_w, 0.20)
                if not np.isfinite(floor) or floor <= 1e-6:
                    floor = 0.1
                w_stable = np.clip(w_raw, floor, None)

        # 3) Elasticity (best-effort)
        try:
            elasticity_df = ElasticityAnalyzer.compute(topsellers)
        except Exception:
            elasticity_df = pd.DataFrame(columns=['product', 'ratio', 'elasticity_score'])

        # 4) Design matrix and target strictly from EXPECTED_COLS
        X = topsellers[EXPECTED_COLS].copy()
        y = np.asarray(topsellers['shipped_units'], dtype=float)

        # 5) Fit GAM expectiles
        modeler = GAMModeler(self.param_search)
        res = modeler.fit_predict_expectiles(
            X_train=X,
            y_train=y,
            X_pred=X,
            w_train=w_stable,
            expectiles=(0.025, 0.50, 0.975),
        )

        # 6) Assemble results
        all_gam_results = topsellers[['product']].copy().reset_index(drop=True)
        for k, arr in res.get('predictions', {}).items():
            all_gam_results[k] = np.asarray(arr, dtype=float)

        if all(c in all_gam_results.columns for c in ['units_pred_0.025','units_pred_0.5','units_pred_0.975']):
            all_gam_results['units_pred_avg'] = (
                all_gam_results[['units_pred_0.025','units_pred_0.5','units_pred_0.975']].to_numpy().mean(axis=1)
            )
        elif 'units_pred_0.5' in all_gam_results.columns:
            all_gam_results['units_pred_avg'] = all_gam_results['units_pred_0.5']
        else:
            all_gam_results['units_pred_avg'] = np.nan

        # Carry identifiers & prices
        for col in ['asin','price','asp']:
            if col in topsellers.columns:
                all_gam_results[col] = topsellers[col].reset_index(drop=True)

        # P50 price used by tables
        if 'pred_0.5' not in all_gam_results.columns:
            all_gam_results['pred_0.5'] = (
                all_gam_results['asp'] if 'asp' in all_gam_results.columns else all_gam_results.get('price')
            )

        # Actual daily revenue (price × daily units)
        if 'revenue_actual' not in all_gam_results.columns:
            price_col = 'asp' if 'asp' in topsellers.columns else ('price' if 'price' in topsellers.columns else None)
            if price_col is not None and 'shipped_units' in topsellers.columns:
                pr = pd.to_numeric(topsellers[price_col], errors='coerce').reset_index(drop=True)
                pu = pd.to_numeric(topsellers['shipped_units'], errors='coerce').reset_index(drop=True)
                all_gam_results['revenue_actual'] = pr * pu
            else:
                all_gam_results['revenue_actual'] = pd.Series(np.nan, index=all_gam_results.index)

        if 'daily_rev' not in all_gam_results.columns and 'revenue_actual' in all_gam_results.columns:
            all_gam_results['daily_rev'] = pd.to_numeric(all_gam_results['revenue_actual'], errors='coerce')

        # Revenue predictions from units × price
        price_col = 'asp' if 'asp' in all_gam_results.columns else 'price'
        if price_col in all_gam_results.columns:
            for q in ('0.025','0.5','0.975'):
                up, rp = f'units_pred_{q}', f'revenue_pred_{q}'
                if up in all_gam_results.columns and rp not in all_gam_results.columns:
                    all_gam_results[rp] = pd.to_numeric(all_gam_results[up], errors='coerce') * pd.to_numeric(all_gam_results[price_col], errors='coerce')

        # Merge elasticity if available
        if not elasticity_df.empty and 'product' in elasticity_df.columns:
            keep_cols = ['product'] + [c for c in ['ratio','elasticity_score'] if c in elasticity_df.columns]
            all_gam_results = all_gam_results.merge(elasticity_df[keep_cols], on='product', how='left')

        # Debug: sanity medians
        try:
            med_act = float(np.nanmedian(all_gam_results.get('daily_rev', pd.Series([np.nan]))))
            med_p50 = float(np.nanmedian(all_gam_results.get('revenue_pred_0.5', pd.Series([np.nan]))))
            print(f"[DBG] med(actual_daily_rev)={med_act:.2f}  med(pred50_rev)={med_p50:.2f}", flush=True)
        except Exception:
            pass

        return topsellers, elasticity_df, all_gam_results


    def _compute_best_tables(self, all_gam_results, topsellers):
        """Run Optimizer and ensure required cols are present."""
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
        """Pick best P50 revenue row per product."""
        if {"revenue_pred_0.5", "units_pred_0.5"}.issubset(all_gam_results.columns):
            idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
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
                        "pred_0.5",
                    ],
                ]
                .drop_duplicates(subset=["product"])
                .reset_index(drop=True)
            )
        else:
            best50 = pd.DataFrame(
                columns=[
                    "product",
                    "asin",
                    "price",
                    "asp",
                    "units_pred_0.5",
                    "revenue_pred_0.5",
                    "pred_0.5",
                ]
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
        if "revenue_actual" not in df.columns and {"price", "shipped_units"}.issubset(df.columns):
            df["revenue_actual"] = pd.to_numeric(df["price"], errors="coerce") * pd.to_numeric(df["shipped_units"], errors="coerce")
        # Derive revenue_pred_* from units_pred_* if needed
        if "price" in df.columns:
            for q in ("0.025", "0.5", "0.975"):
                up = f"units_pred_{q}"
                rp = f"revenue_pred_{q}"
                if rp not in df.columns and up in df.columns:
                    df[rp] = pd.to_numeric(df[up], errors="coerce") * pd.to_numeric(df["price"], errors="coerce")

        # Actual daily revenue
                # Ensure denominator is a Series; missing or zero -> NaN
        if "days_sold" in df.columns:
            den = pd.to_numeric(df["days_sold"], errors="coerce")
        else:
            den = pd.Series(np.nan, index=df.index)
        den = den.mask(den == 0)
        act_rev = pd.to_numeric(df["revenue_actual"], errors="coerce") if "revenue_actual" in df.columns else pd.Series(np.nan, index=df.index)
        daily_act = (act_rev / den).where(den.notna(), act_rev)

        out = {}

        # P50 metrics
        pred50_rev = pd.to_numeric(df["revenue_pred_0.5"], errors="coerce") if "revenue_pred_0.5" in df.columns else pd.Series(np.nan, index=df.index)
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

    def gam_results(self, all_gam_results: pd.DataFrame):
        """
        Single y-axis revenue chart with:
        - Revenue band (P2.5 - P97.5)
        - P50 revenue line
        - Actual revenue points
        - Markers for recommended, conservative and optimistic prices
        """
        import plotly.graph_objects as go
        import numpy as np

        # 0) Guards
        if all_gam_results is None or getattr(all_gam_results, "empty", True):
            return self.empty_fig("No model data")

        need_cols = [
            "product", "asp", "revenue_actual",
            "revenue_pred_0.025", "revenue_pred_0.5", "revenue_pred_0.975",
        ]
        missing = [c for c in need_cols if c not in all_gam_results.columns]
        if missing:
            return self.empty_fig(f"Missing column(s): {', '.join(missing)}")

        # 1) Figure
        fig = go.Figure()

        # 2) Per-product plotting, **sorted by ASP**
        for group_name, g in all_gam_results.groupby("product"):
            g = g.dropna(subset=["asp"]).copy()
            if g.empty:
                continue
            g = g.sort_values("asp")

            # Actual revenue points
            fig.add_trace(go.Scatter(
                x=g["asp"], y=g["revenue_actual"], mode="markers",
                opacity=0.55, name=f"{group_name} • Actual Revenue",
                marker=dict(size=8, symbol="circle"),
                legendgroup=group_name,
                hovertemplate="ASP=%{x:$,.2f}<br>Actual Rev=%{y:$,.0f}<extra></extra>",
            ))

            # Revenue band (P2.5–P97.5), drawn as two traces to avoid self-crossing
            fig.add_trace(go.Scatter(
                name=f"{group_name} (Rev band upper)",
                x=g["asp"], y=g["revenue_pred_0.975"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                legendgroup=group_name,
            ))
            fig.add_trace(go.Scatter(
                name=f"{group_name} (Rev band)",
                x=g["asp"], y=g["revenue_pred_0.025"],
                mode="lines", line=dict(width=0), fill="tonexty", opacity=0.25,
                showlegend=False, hoverinfo="skip", legendgroup=group_name,
            ))

            # Diamonds for recommended / conservative / optimistic prices
            best_rows = {
                "Recommended (P50)": ("revenue_pred_0.5", g.loc[g["revenue_pred_0.5"].idxmax()]),
                "Conservative (P2.5)": ("revenue_pred_0.025", g.loc[g["revenue_pred_0.025"].idxmax()]),
                "Optimistic (P97.5)": ("revenue_pred_0.975", g.loc[g["revenue_pred_0.975"].idxmax()]),
            }
            marker_colors = {
                "Conservative (P2.5)": "#1F6FEB",
                "Optimistic (P97.5)": "#238636",
                "Recommended (P50)": "#B82132",
            }
            for label, (pred_col, row) in best_rows.items():
                fig.add_trace(go.Scatter(
                    x=[row["asp"]], y=[row[pred_col]], mode="markers",
                    marker=dict(color=marker_colors[label], size=12, symbol="diamond"),
                    name=f"{group_name} • {label}", legendgroup=group_name,
                    hovertemplate=f"{label}<br>Price=%{{x:$,.2f}}<br>Rev=%{{y:$,.0f}}<extra></extra>",
                ))

            # P50 line (sorted!)
            fig.add_trace(go.Scatter(
                x=g["asp"], y=g["revenue_pred_0.5"], mode="lines",
                name=f"{group_name} • Expected Revenue (P50)",
                line=dict(width=2), legendgroup=group_name,
                hovertemplate="ASP=%{x:$,.2f}<br>Expected Rev=%{y:$,.0f}}<extra></extra>",
            ))

        # 3) Layout (avoid crashing if self.template is not set)
        template = getattr(self, "template", None)
        fig.update_layout(
            template=template if template else None,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            margin=dict(r=8, t=40, b=40, l=60),
        )
        fig.update_yaxes(title="Expected Daily Revenue", tickprefix="$", separatethousands=True)
        fig.update_xaxes(title="Average Selling Price (ASP)", tickprefix="$", separatethousands=True)
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


# if __name__ == "__main__":
#     pricing_df, product_df = pd.read_csv('data/pricing.csv'), pd.read_csv('data/products.csv')

#     PricingPipeline(pricing_df,product_df,top_n=10, use_grid_search=True).assemble_dashboard_frames()


# all_gam_results = GAMModeler(
#     DataEngineer(pricing_df, product_df, top_n=10).prepare()).run()
# # Create a viz instance first, then call the method
# viz_instance = viz()
# viz_instance.gam_results(all_gam_results)
