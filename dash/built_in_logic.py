# --------- built_in_logic.py  ---------
# (RMSE-focused; Top-N only; adds annualized opps & data range)
import os, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict

# viz
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

# ML
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pygam import ExpectileGAM, s, f, te

# local import
from helpers import DataEng


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

    def _time_decay(self, prepared_df, decay_rate=-0.01) -> pd.DataFrame:
        """
        Calculates an exponential decay weight based on time difference.
        Assumes order_date is already synthesized/valid.

        param
            prepared_df (assuming that DataEngineer.prepare() has been run through)
            decay_rate (default: -0.01)

        return
            df
        """
        df = prepared_df.copy()
        df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")

        today_ref = pd.Timestamp("today")
        df["days_apart"] = (today_ref - df["order_date"]).dt.days
        df["time_decay_weight"] = np.exp(decay_rate * df["days_apart"].fillna(0))
        return df

    def _make_weights(self, sub: pd.DataFrame, **kwargs) -> np.ndarray:
        """Simplified weight calculation using only time decay and rarity"""
        # 1) Time decay
        decayed_df = self._time_decay(sub, decay_rate=-0.01)  # Fixed decay rate
        w = decayed_df["time_decay_weight"].astype(float).to_numpy()

        # 2) Rarity multiplier
        rarity_mult = self._rarity_multiplier(
            sub["price"].to_numpy(float),
            smooth=5,
            cap=1.25,
            beta=0.35,
            tails_only=True,
            tail_q=0.10,
        )
        w *= rarity_mult

        # Final normalization
        w = np.clip(w, 0.25, 2.5)
        w = w / w.mean() if w.size else w
        return w

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

    def _label_encoder(self, df) -> pd.DataFrame:
        """label encoding categorical variable"""
        le = LabelEncoder()
        res = df.copy()

        res["event_encoded"] = le.fit_transform(res["event_name"])
        res["product_encoded"] = le.fit_transform(res["product"])

        return res

    @staticmethod
    def _rarity_multiplier(
        prices: np.ndarray,
        bin_width: float | None = None,  # None = auto (Freedman–Diaconis)
        smooth: int = 5,  # stronger smoothing
        cap: float = 1.25,  # tighter cap than 1.35
        beta: float = 0.35,  # softer curvature
        tails_only: bool = True,  # <— NEW: only weight the tails
        tail_q: float = 0.10,  # 10% tails by default
    ) -> np.ndarray:
        """
        find out distribution density - assign bigger bonus points to rare points

            -> (1/density)^beta -> normalize -> clip.
            - Auto bin width via Freedman-Diaconis unless provided.
            - Capping via light smoothing of bin counts to avoid single-bin spikes.
            - Normalized to mean=1 so it only redistributes mass.

        Returns
            a multiplicative weight per observation that is higher in
            low-density (rare) price regions and lower in dense regions,
            but bounded to avoid domination by singletons.
        """
        p = np.asarray(prices, dtype=float).ravel()
        n = p.size
        if n == 0 or np.allclose(p, p[0]):
            return np.ones_like(p)

        # define safe interior (no rarity bonus there)
        if tails_only:
            lo, hi = np.quantile(p, [tail_q, 1.0 - tail_q])
            in_core = (p >= lo) & (p <= hi)
        else:
            in_core = np.zeros_like(p, dtype=bool)

        # Freedman–Diaconis bin width with guardrails
        if bin_width is None:
            q25, q75 = np.percentile(p, [25, 75])
            iqr = max(q75 - q25, 1e-9)
            bw = 2.0 * iqr / np.cbrt(n)
            span = max(p.max() - p.min(), 1e-6)
            bw = np.clip(bw, span / 80, span / 25)
        else:
            bw = float(bin_width)

        edges = np.arange(p.min() - 1e-9, p.max() + bw + 1e-9, bw)
        counts, edges = np.histogram(p, bins=edges)

        if smooth and smooth > 1:
            k = smooth + (smooth % 2 == 0)  # force odd
            kernel = np.ones(k, dtype=float) / k
            counts = np.convolve(counts, kernel, mode="same")

        counts = np.maximum(counts, 1e-12)
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(counts) - 1)
        c = counts[idx]

        raw = (1.0 / c) ** float(beta)
        raw /= np.mean(raw)

        mult = np.clip(raw, 1.0 / float(cap), float(cap))
        # disable rarity inside the core band
        mult[in_core] = 1.0
        return mult

    @staticmethod
    def _cap_local_leverage(
        w: np.ndarray, prices: np.ndarray, window: int = 7, lcap: float = 1.4
    ) -> np.ndarray:
        """Limit each point's weight to <= lcap × local-mean(weight) in price order."""
        w = np.asarray(w, dtype=float)
        x = np.asarray(prices, dtype=float)
        order = np.argsort(x)
        ws = w[order].copy()
        m = max(1, window // 2)
        n = len(ws)
        for i in range(n):
            lo, hi = max(0, i - m), min(n, i + m + 1)
            mu = float(ws[lo:hi].mean())
            if mu > 0:
                ws[i] = min(ws[i], lcap * mu)
        out = np.empty_like(ws)
        out[order] = ws
        return out

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


class ParamSearchCV:
    """
    Cross-validation based parameter search for price curve modeling
        - Tuning for (1) n_splines, (2) lambda;
        - Uses price-based stratification and golden-section search for efficient tuning.
    """

    def __init__(
        self,
        n_splits: int = 4,
        n_splines_grid: tuple = (15, 20, 25, 30),
        loglam_range: tuple = (np.log(50.0), np.log(1000.0)),
        lam_iters: int = 6,
        expectile: float = 0.5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.n_splines_grid = n_splines_grid
        self.loglam_range = loglam_range
        self.lam_iters = lam_iters
        self.expectile = expectile
        self.random_state = random_state
        # Track best solution ever found
        self.best_score_ever = np.inf
        self.best_config_ever = None

    def _fit_price_curve_with_anchors(
        self,
        X: np.ndarray,  # shape (n, 1) price in col 0
        y: np.ndarray,  # target (e.g., daily revenue or units)
        w: Optional[np.ndarray] = None,  # sample weights (already clipped/normalized)
        *,
        expectile: float = 0.5,
        n_splines: int = 16,
        lam: float = 500.0,
        max_iter: int = 6000,
        tol: float = 2e-4,
    ):
        """
        fits GAM with specific params
        - take params (n_splines, lambda, etc.)
        - adds anchor points to suppress boundary spikes
        - fits a single GAM model with those exact specifications


        Returns
            a fitted ExpectileGAM.
        """
        if ExpectileGAM is None:
            raise RuntimeError("pygam is required for fit_price_curve_with_anchors")

        X = np.asarray(X, float)
        y = np.asarray(y, float)
        p = X[:, 0]

        # local medians on edges
        try:
            q10, q90 = np.quantile(p, [0.10, 0.90])
        except Exception:
            q10, q90 = (np.min(p), np.max(p))
        y_lo = (
            float(np.median(y[p <= q10])) if np.any(p <= q10) else float(np.median(y))
        )
        y_hi = (
            float(np.median(y[p >= q90])) if np.any(p >= q90) else float(np.median(y))
        )

        # anchors just outside the observed range
        span = (p.max() - p.min()) or 1.0
        eps = 1e-3 * span
        X_anchor = np.array([[p.min() - eps], [p.max() + eps]], dtype=float)
        y_anchor = np.array([y_lo, y_hi], dtype=float)

        if w is None:
            w_anchor = np.array([0.05, 0.05], dtype=float)
            w_aug = None
        else:
            w = np.asarray(w, float)
            w_anchor = np.full(2, 0.05 * float(np.mean(w)), dtype=float)
            w_aug = np.concatenate([w, w_anchor])

        X_aug = np.vstack([X, X_anchor])
        y_aug = np.concatenate([y, y_anchor])

        terms = s(0, n_splines=int(n_splines), spline_order=3)
        gam = ExpectileGAM(
            terms,
            lam=float(lam),
            expectile=float(expectile),
            max_iter=int(max_iter),
            tol=float(tol),
        )
        gam.fit(X_aug, y_aug, weights=w_aug)

        return gam

    def _price_bins_for_cv(self, prices: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """
        Create stratified bins based on price quantiles for balanced CV splits.

        Args:
            prices: Array of prices
            n_bins: Number of bins to create

        Returns:
            Array of bin labels (0 to n_bins-1) for each price
        """
        qs = np.quantile(prices, np.linspace(0, 1, n_bins + 1))
        # Make edges strictly monotone to handle ties
        for i in range(1, len(qs)):
            if qs[i] <= qs[i - 1]:
                qs[i] = qs[i - 1] + 1e-9
        return np.clip(np.searchsorted(qs, prices, side="right") - 1, 0, n_bins - 1)

    def _cv_score_anchored(self, X, y, w, *, cfg) -> float:
        """
        Score a parameter configuration using stratified K-fold CV.

        Args:
            X: Features (price in first column)
            y: Target values
            w: Sample weights
            cfg: Parameter configuration dictionary

        Returns:
            Mean CV score (lower is better)
        """
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        p = X[:, 0]  # prices
        w = None if w is None else np.asarray(w, float)

        # Stratify by price bins
        bins = self._price_bins_for_cv(p, n_bins=5)
        idx = np.arange(len(p))
        rng.shuffle(idx)

        scores = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(idx):
            tr = idx[train_idx]
            va = idx[val_idx]

            # Fit on train fold
            gam = self._fit_price_curve_with_anchors(
                X[tr],
                y[tr],
                None if w is None else w[tr],
                expectile=cfg.get("expectile", self.expectile),
                n_splines=int(cfg["n_splines"]),
                lam=float(cfg["lam"]),
                max_iter=int(cfg.get("max_iter", 6000)),
                tol=float(cfg.get("tol", 2e-4)),
            )

            # Score on validation fold
            yhat = gam.predict(X[va]).astype(float)
            resid2 = (y[va] - yhat) ** 2

            # Weighted or unweighted RMSE
            if w is None:
                rmse = float(np.sqrt(resid2.mean()))
            else:
                ww = w[va] / max(w[va].sum(), 1e-12)
                rmse = float(np.sqrt((resid2 * ww).sum()))

            # Add curvature penalty
            order = np.argsort(X[va, 0])
            yv = yhat[order]
            if yv.size >= 5:
                dd = np.abs(np.diff(yv, n=2))
                denom = max(np.mean(np.abs(yv)), 1e-9)
                curvature = float(np.mean(dd) / denom)
            else:
                curvature = 0.0

            scores.append(rmse * (1.0 + 0.7 * curvature))

        return float(np.mean(scores))

    def _golden_search_loglam(self, X, y, w, *, base_cfg) -> tuple:
        """
        Use golden-section search to efficiently find best lambda value.

        Args:
            X, y, w: Data and weights
            base_cfg: Base configuration to optimize lambda for

        Returns:
            (best_config, best_score) tuple
        """
        phi = (1 + 5**0.5) / 2
        invphi = 1 / phi

        a, b = float(self.loglam_range[0]), float(self.loglam_range[1])

        # Initialize interior points
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi

        cfg_c = dict(base_cfg, lam=float(np.exp(c)))
        sc_c = self._cv_score_anchored(X, y, w, cfg=cfg_c)

        cfg_d = dict(base_cfg, lam=float(np.exp(d)))
        sc_d = self._cv_score_anchored(X, y, w, cfg=cfg_d)

        for _ in range(max(self.lam_iters - 1, 0)):
            if sc_c <= sc_d:
                b, d = d, c
                sc_d = sc_c
                c = b - (b - a) * invphi
                cfg_c = dict(base_cfg, lam=float(np.exp(c)))
                sc_c = self._cv_score_anchored(X, y, w, cfg=cfg_c)
            else:
                a, c = c, d
                sc_c = sc_d
                d = a + (b - a) * invphi
                cfg_d = dict(base_cfg, lam=float(np.exp(d)))
                sc_d = self._cv_score_anchored(X, y, w, cfg=cfg_d)

        if sc_c <= sc_d:
            return cfg_c, sc_c
        return cfg_d, sc_d

    def _update_search_range(self, current_score, current_cfg):
        """Update search ranges but always relative to best solution ever found"""
        # If this is a new best score ever, update it
        if current_score < self.best_score_ever:
            self.best_score_ever = current_score
            self.best_config_ever = current_cfg.copy()
            self._gam_best_line = (
                f"New best score {current_score:.4f} with config {current_cfg}"
            )

            # Use best-ever solution's parameters as center of search
            current_loglam = np.log(self.best_config_ever["lam"])

            # More conservative range updates
            if self.best_score_ever < 1.0:
                range_width = 0.5  # Narrow search around excellent solution
            elif self.best_score_ever < 2.0:
                range_width = 0.7
            else:
                range_width = 1.0

            # self.loglam_range = (
            #     max(np.log(50.0), current_loglam - range_width),
            #     min(np.log(1000.0), current_loglam + range_width),
            # )

    def fit(self, X, y, sample_weight=None, verbose=False):
        """ """
        current_best_cfg, current_best_sc = None, np.inf

        # Regular grid search
        for ns in self.n_splines_grid:
            try:
                cfg, sc = self._golden_search_loglam(
                    X[:, [0]],
                    y,
                    sample_weight,
                    base_cfg=dict(
                        expectile=self.expectile,
                        n_splines=int(ns),
                        max_iter=6000,
                        tol=2e-4,
                    ),
                )

                if sc < current_best_sc:
                    current_best_sc = sc
                    current_best_cfg = cfg
                    self._update_search_range(current_best_sc, current_best_cfg)

            except Exception:
                continue

        # Always use the best configuration we've ever found
        if self.best_config_ever is not None:
            return self._fit_price_curve_with_anchors(
                X[:, [0]],
                y,
                sample_weight,
                expectile=self.expectile,
                n_splines=self.best_config_ever["n_splines"],
                lam=self.best_config_ever["lam"],
                max_iter=6000,
                tol=2e-4,
            )


class GAMTuner:
    """
    Silent tuner for ExpectileGAM.
    - Uses coarse + fine lambda search over multiple n_splines for price.
    - No logging here; caller can read best info from returned model:
        model._tuning_n_price, model._tuning_lam, model._tuning_score
    """

    def __init__(
        self, expectile=0.5, lam_grid=None, n_splines_grid=None, use_interaction=False
    ):
        self.expectile = float(expectile)
        self.lam_grid = (
            np.logspace(-6, 9, 18) if lam_grid is None else np.asarray(lam_grid, float)
        )
        self.n_splines_grid = (
            [15, 25, 35, 45] if n_splines_grid is None else list(n_splines_grid)
        )
        self.use_interaction = bool(use_interaction)
        # external toggles (not used for printing here; kept for compatibility)
        self._verbose = getattr(self, "_verbose", False)

    # ---------- public API ----------
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        Xs, y, sw, scaler, num_idx, fac_idx = self._prepare_data(X, y, sample_weight)

        best_gam, best_ns, coarse_lam, best_score = self._coarse_search(
            Xs, y, sw, sample_weight
        )

        best_gam, best_score, lam_mean = self._fine_search(
            Xs, y, sw, sample_weight, best_ns, coarse_lam, best_gam, best_score
        )

        if best_gam is None:
            # Last-resort single fit with a reasonable terms spec
            terms = self._build_terms(best_ns if best_ns is not None else 25)
            best_gam = ExpectileGAM(
                terms, expectile=self.expectile, fit_intercept=True
            ).fit(Xs, y, weights=sample_weight)
            lam_mean = self._lam_mean(getattr(best_gam, "lam", None))
            best_score = np.nan

        # attach preprocessors + tuning summary
        best_gam._scaler = scaler
        best_gam._num_idx = num_idx
        best_gam._fac_idx = fac_idx
        best_gam._tuning_n_price = best_ns
        best_gam._tuning_lam = lam_mean
        best_gam._tuning_score = (
            float(best_score) if np.isfinite(best_score) else np.nan
        )
        return best_gam

    # ---------- helpers ----------
    def _prepare_data(self, X, y, sample_weight):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        num_idx, fac_idx = [0, 1], [2]

        scaler = StandardScaler()
        X_num = scaler.fit_transform(X[:, num_idx])
        X_fac = X[:, fac_idx]
        Xs = np.column_stack([X_num, X_fac])

        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, float)
            sw[~np.isfinite(sw)] = 0.0
            ssum = sw.sum()
            sw = (sw / ssum) if ssum > 0 else None
        return Xs, y, sw, scaler, num_idx, fac_idx

    def _build_terms(self, n_price: int):
        terms = (
            s(0, basis="ps", n_splines=int(n_price))
            + s(1, basis="ps", n_splines=15)
            + f(2)
        )
        if self.use_interaction:
            terms = terms + te(0, 1)
        return terms

    def _score_model(
        self, gam: ExpectileGAM, Xs: np.ndarray, y: np.ndarray, sw: np.ndarray | None
    ) -> float:
        """RMSE*(1+0.9*smoothness) on the provided (Xs, y)."""
        y_pred = gam.predict(Xs)
        resid2 = (y - y_pred) ** 2
        mse = float((resid2 * sw).sum()) if sw is not None else float(resid2.mean())
        rmse = np.sqrt(mse)
        denom = max(np.mean(np.abs(y_pred)), 1e-9)
        smooth = float(np.mean(np.abs(np.diff(y_pred, 2))) / denom)
        return rmse * (1 + 0.9 * smooth)

    def _coarse_search(self, Xs, y, sw, sample_weight):
        best_gam, best_score, best_ns, coarse_best_lam = None, np.inf, None, None
        for n_price in self.n_splines_grid or [25]:
            terms = self._build_terms(n_price)
            try:
                g = ExpectileGAM(
                    terms, expectile=self.expectile, fit_intercept=True
                ).gridsearch(
                    Xs, y, lam=self.lam_grid, weights=sample_weight, progress=False
                )
                sc = self._score_model(g, Xs, y, sw)
                if sc < best_score:
                    best_score, best_gam, best_ns = sc, g, n_price
                    coarse_best_lam = getattr(g, "lam", None)
            except Exception:
                continue
        return best_gam, best_ns, coarse_best_lam, best_score

    def _fine_search(
        self, Xs, y, sw, sample_weight, best_ns, coarse_lam, best_gam, best_score
    ):
        if coarse_lam is None:
            return best_gam, best_score, self._lam_mean(getattr(best_gam, "lam", None))
        try:
            center = self._lam_mean(coarse_lam)
            fine = np.clip(center * np.logspace(-0.6, 0.6, 8), 1e-8, 1e12)
            terms = self._build_terms(best_ns if best_ns is not None else 25)
            g2 = ExpectileGAM(
                terms, expectile=self.expectile, fit_intercept=True
            ).gridsearch(Xs, y, lam=fine, weights=sample_weight, progress=False)
            sc2 = self._score_model(g2, Xs, y, sw)
            if sc2 < best_score:
                return g2, sc2, self._lam_mean(getattr(g2, "lam", None))
            return best_gam, best_score, self._lam_mean(getattr(best_gam, "lam", None))
        except Exception:
            return best_gam, best_score, self._lam_mean(getattr(best_gam, "lam", None))

    @staticmethod
    def _lam_mean(lam):
        """Return a scalar summary of lam (handles None, scalar, or array)."""
        if lam is None:
            return np.nan
        lam = np.asarray(lam).astype(float)
        return float(lam.mean()) if lam.size > 1 else float(lam.item())


class GAMModeler:

    def __init__(
        self,
        topsellers: pd.DataFrame,
        # split + bootstrap controls
        test_size: float = 0.2,
        split_by_date: bool = True,
        n_bootstrap: int = 10,
        bootstrap_frac: float = 0.8,
        random_state: int | None = 42,
        bootstrap_target_rel_se: float | None = 0.04,
        # log
        verbose: bool = False,
        log_every: int = 5,
    ):
        self.topsellers = topsellers if topsellers is not None else pd.DataFrame()
        self.test_size = float(test_size)
        self.split_by_date = bool(split_by_date)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_frac = float(bootstrap_frac)
        self.bootstrap_target_rel_se = bootstrap_target_rel_se
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.log_every = int(log_every)

        # Initialize DataEngineer for weight calculation
        self.engineer = DataEngineer(None, None)
        self.engineer._verbose = self.verbose

        # Initialize parameter search
        self.param_search = ParamSearchCV(
            n_splits=4,
            n_splines_grid=(15, 20, 25, 30),
            loglam_range=(np.log(50.0), np.log(1000.0)),
            lam_iters=6,
            random_state=random_state,
        )

    # --------- helpers ---------
    def _bootstrap_loops_for_size(self, n_train: int) -> int:
        if self.n_bootstrap == 0:
            return 0
        # auto budget by sample size
        if n_train < 60:
            return min(self.n_bootstrap, 0)
        if n_train < 120:
            return min(self.n_bootstrap, 10)
        if n_train < 300:
            return min(self.n_bootstrap, 20)
        return min(self.n_bootstrap, 30)

    def _required_cols(self) -> list:
        return [
            "price",
            "deal_discount_percent",
            "event_encoded",
            "product_encoded",
            "shipped_units",
        ]

    # ---------- _make_design_matrix ----------
    def _make_design_matrix(self, sub: pd.DataFrame):
        """
        Orchestrates per-product design matrix construction.

        Returns (unchanged signature expected by callers):
            sub_clean, X_tr, y_tr, w_tr, X_all
        """
        if sub is None or sub.empty:
            return None, None, None, None, None

        # sort for stability
        sub = self._mdm_sort(sub)

        # per-row weights
        w = self._mdm_build_weights(sub)
        sub = pd.concat([sub, pd.Series(w, name="wt", index=sub.index)], axis=1)

        # basic checks
        sub_clean = self._mdm_prune_and_check(sub)
        if sub_clean is None:
            return None, None, None, None, None

        # train/test split
        is_train, _ = self._mdm_train_test_split(sub_clean)
        sub_clean["split"] = np.where(is_train, "train", "test")

        # features/targets/weights
        X_all, y_all, w_all = self._mdm_build_features(sub_clean)

        # slice train
        mask = is_train.values
        X_tr = X_all[mask]
        y_tr = y_all[mask]
        w_tr = w_all[mask]

        # Note: we intentionally return ONLY 5 items to match existing callers.
        return sub_clean, X_tr, y_tr, w_tr, X_all

    def _mdm_sort(self, sub: pd.DataFrame) -> pd.DataFrame:
        return sub.sort_values("price").reset_index(drop=True)

    def _mdm_build_weights(self, sub: pd.DataFrame) -> np.ndarray:
        """Build weights using only time decay and price rarity"""
        # Cache key based on dataframe content
        cache_key = hash(str(sub.values.tobytes()))
        if hasattr(self, "_weight_cache") and cache_key in self._weight_cache:
            return self._weight_cache[cache_key]

        # Time decay weights
        dec = (
            self.engineer._time_decay(sub)["time_decay_weight"].astype(float).to_numpy()
        )

        # Rarity weights based on price distribution
        rarity = self.engineer._rarity_multiplier(
            sub["price"].to_numpy(float),
            smooth=5,
            cap=1.25,
            beta=0.35,
            tails_only=True,
            tail_q=0.10,
        )

        # Combine weights and normalize
        w = dec * rarity
        w = w / w.mean()

        # Cache result
        if not hasattr(self, "_weight_cache"):
            self._weight_cache = {}
        self._weight_cache[cache_key] = w

        return w

    def _mdm_prune_and_check(self, sub: pd.DataFrame):
        sub_clean = sub.dropna(subset=["price", "shipped_units"]).copy()
        if sub_clean.shape[0] < 5 or sub_clean["price"].nunique() < 3:
            return None
        return sub_clean

    def _mdm_train_test_split(self, sub_clean: pd.DataFrame):
        is_train = pd.Series(True, index=sub_clean.index)
        used_time_split = False
        if self.split_by_date and "order_date" in sub_clean.columns:
            dates = pd.to_datetime(sub_clean["order_date"], errors="coerce")
            if dates.notna().sum() >= 5:
                cutoff = dates.quantile(1 - self.test_size)
                is_train = dates <= cutoff
                used_time_split = True
        if is_train.sum() < 3 or (~is_train).sum() < 1:
            seed = abs(hash(str(sub_clean.get("product", "?").iloc[0]))) % (2**32 - 1)
            rng = np.random.default_rng(self.random_state or seed)
            is_train = pd.Series(
                rng.random(len(sub_clean)) >= self.test_size, index=sub_clean.index
            )
            if is_train.sum() < 3:
                is_train[:] = True

        return is_train, used_time_split

    def _mdm_build_features(self, sub_clean: pd.DataFrame):
        price = sub_clean["price"].to_numpy(dtype=float)
        if "deal_discount_percent" in sub_clean.columns:
            disc = pd.to_numeric(
                sub_clean["deal_discount_percent"], errors="coerce"
            ).fillna(0.0)
            disc = disc.clip(lower=-100.0, upper=100.0).to_numpy(dtype=float)
        else:
            disc = np.zeros(len(sub_clean), dtype=float)
        if "event_encoded" in sub_clean.columns:
            evt = (
                pd.to_numeric(sub_clean["event_encoded"], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=int)
            )
        else:
            evt = np.zeros(len(sub_clean), dtype=int)
        X_all = np.column_stack([price, disc, evt])
        y_all = sub_clean["shipped_units"].to_numpy(dtype=float)
        w_all = sub_clean["wt"].to_numpy(dtype=float)
        return X_all, y_all, w_all

    def _ensure_factor_domain(
        self, X_train, y_train, w_train, X_pred, fac_col=2, eps=1e-8
    ):
        """
        Make sure all categories present in X_pred[:, fac_col] exist in the
        training design given to pyGAM. If missing, append one pseudo-row per
        missing level with tiny weight (eps).
        """
        xt = np.asarray(X_train)
        xp = np.asarray(X_pred)

        train_lvls = np.unique(xt[:, fac_col])
        pred_lvls = np.unique(xp[:, fac_col])
        missing = np.setdiff1d(pred_lvls, train_lvls)

        if missing.size == 0:
            return X_train, y_train, w_train

        # use medians for numeric columns as placeholders
        price_med = np.median(xt[:, 0].astype(float)) if xt.shape[1] > 0 else 0.0
        disc_med = np.median(xt[:, 1].astype(float)) if xt.shape[1] > 1 else 0.0

        X_extra = np.column_stack(
            [
                np.full(missing.size, price_med, dtype=float),
                np.full(missing.size, disc_med, dtype=float),
                missing.astype(float),  # factor column
            ]
        )

        y_extra = np.full(
            missing.size, np.median(y_train) if len(y_train) else 0.0, dtype=float
        )
        if w_train is None:
            w_train = np.ones(len(y_train), dtype=float)

        w_extra = np.full(missing.size, float(eps), dtype=float)

        X_aug = np.vstack([xt, X_extra])
        y_aug = np.concatenate([y_train, y_extra])
        w_aug = np.concatenate([w_train, w_extra])
        return X_aug, y_aug, w_aug

    @staticmethod
    def _bootstrap_weights(n_rows: int, sample_idx: np.ndarray) -> np.ndarray:
        """
        Return a length-n_rows vector: counts of how many times each row was drawn.
        If you want classical bootstrap behavior, use this as multiplicative
        weights (and multiply by the original sample weights).
        """
        w = np.bincount(sample_idx, minlength=n_rows).astype(float)
        # keep the average weight ~1 for numeric stability
        mean = w.mean()
        if mean > 0:
            w /= mean
        return w

    def _fit_expectiles(
        self, X_train, y_train, w_train, X_pred, qs=(0.025, 0.5, 0.975)
    ) -> dict:
        """
        Fits multiple expectiles using only basic weights.
        Simplified to remove complex weight adjustments and edge tapering.
        """
        # 1) Data preparation - just ensure factor domain
        X_seed, y_seed, w_seed = self._fe_seed_domain(X_train, y_train, w_train, X_pred)

        # 2) Core fitting
        preds = {}
        for q in qs:
            # This is where parameter search happens for each quantile
            fit_result = self._fe_fit_one_expectile(q, X_seed, y_seed, X_pred, w_seed)
            preds.update(fit_result)

            # Add logging here to track progress per quantile
            _v_best(self, f"Fitted expectile {q} with {len(X_seed)} points")

        # 3) Ensure predictions don't cross...
        qkeys = sorted(
            [k for k in preds if k.startswith("units_pred_") and not k.endswith("_sd")],
            key=lambda k: float(k.replace("units_pred_", "")),
        )

        if qkeys:
            P = np.vstack([preds[k] for k in qkeys])
            P_nc = np.maximum.accumulate(P, axis=0)  # Enforce monotonicity
            for i, k in enumerate(qkeys):
                preds[k] = np.maximum(P_nc[i], 0.0)  # Ensure non-negative

                # Handle standard deviation estimates if present
                sd_key = f"{k}_sd"
                if sd_key in preds:
                    preds[sd_key] = np.maximum(preds[sd_key], 0.0)

        # Add summary logging here if needed
        _v_best(
            self, f"Completed all {len(qs)} expectiles with monotonicity enforcement"
        )

        return preds

    # ---- helpers ----
    def _fe_seed_domain(self, X_train, y_train, w_train, X_pred, fac_col=2, eps=1e-8):
        return self._ensure_factor_domain(
            X_train, y_train, w_train, X_pred, fac_col=fac_col, eps=eps
        )

    def _fe_fit_one_expectile(self, q, X_seed, y_seed, X_pred, w_robust):
        def fit_once(q, boot_idx=None):
            try:
                self.param_search.expectile = q
                w_boot = (
                    w_robust
                    if boot_idx is None
                    else (self._bootstrap_weights(len(y_seed), boot_idx) * w_robust)
                )

                best_gam = self.param_search.fit(
                    X_seed[:, [0]], y_seed, w_boot, verbose=self.verbose
                )

                if best_gam is not None and hasattr(best_gam, "_tuning_score"):
                    _v_best(
                        self,
                        f"Expectile {q}: Found best model with score {best_gam._tuning_score:.4f}",
                    )

                if best_gam is None:
                    return None

                X_pred_price = X_pred[:, [0]]
                return np.maximum(best_gam.predict(X_pred_price), 0.0)

            except Exception as e:
                print(f"Error in fit_once: {str(e)}")
                return None

        # Add return dictionary with predictions
        pred = fit_once(q)
        if pred is not None:
            return {
                f"units_pred_{q}": pred,
                f"units_pred_{q}_sd": np.zeros_like(
                    pred
                ),  # or actual SD if bootstrapping
            }
        return {}  # Return empty dict if fit failed

    def _assemble_group_results(
        self, sub_clean: pd.DataFrame, preds: dict
    ) -> pd.DataFrame:
        res_pred = pd.DataFrame(preds, index=sub_clean.index)

        price_nonneg = np.maximum(sub_clean["price"].values.astype(float), 0.0)

        # revenue preds + (optional) SDs from bootstrap
        for k in list(preds.keys()):
            if k.startswith("units_pred_") and not k.endswith("_sd"):
                q = k.replace("units_pred_", "")
                rev_col = f"revenue_pred_{q}"
                res_pred[rev_col] = (res_pred[k].astype(float) * price_nonneg).clip(
                    lower=0.0
                )

                sd_col = f"units_pred_{q}_sd"
                if sd_col in res_pred.columns:
                    res_pred[f"revenue_pred_{q}_sd"] = (
                        res_pred[sd_col].astype(float) * price_nonneg
                    )

        # clamp any unit preds that slipped (paranoia)
        for c in [
            c
            for c in res_pred.columns
            if c.startswith("units_pred_") and not c.endswith("_sd")
        ]:
            res_pred[c] = pd.to_numeric(res_pred[c], errors="coerce").clip(lower=0.0)

        keep_cols = [
            "order_date",
            "wt",
            "asin",
            "price",
            "days_sold",
            "product",
            "event_name",
            "deal_discount_percent",
            "shipped_units",
            "revenue_share_amt",
            "split",  # NEW: carry split downstream
        ]
        keep_cols = [c for c in keep_cols if c in sub_clean.columns]
        return pd.concat(
            [
                sub_clean[keep_cols].reset_index(drop=True),
                res_pred.reset_index(drop=True),
            ],
            axis=1,
        )

    def _postprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """ """
        if df is None or df.empty:
            return df

        df["asp"] = df["price"]
        df["asin"] = df["asin"].astype(str)

        # Actuals (already non-negative elsewhere) – kept for clarity
        df["revenue_actual"] = (df["shipped_units"] * df["price"]).clip(lower=0.0)

        # prepping for model fit (daily rev) KPI card
        if "days_sold" in df.columns:
            den = pd.to_numeric(df["days_sold"], errors="coerce").replace(0, np.nan)
            df["daily_rev"] = (
                pd.to_numeric(df["revenue_actual"], errors="coerce") / den
            ).clip(lower=0.0)
            df["daily_units"] = (
                pd.to_numeric(df["shipped_units"], errors="coerce") / den
            ).clip(lower=0.0)
        else:
            # Fall back to per-row actuals if days_sold is missing
            df["daily_rev"] = pd.to_numeric(df["revenue_actual"], errors="coerce").clip(
                lower=0.0
            )
            df["daily_units"] = pd.to_numeric(
                df["shipped_units"], errors="coerce"
            ).clip(lower=0.0)

        # Average predictions -> clamp units & revenue to >= 0
        units_cols = [c for c in df.columns if c.startswith("units_pred_")]
        if units_cols:
            df["units_pred_avg"] = df[units_cols].mean(axis=1).clip(lower=0.0)
            df["revenue_pred_avg"] = (df["units_pred_avg"] * df["price"]).clip(
                lower=0.0
            )
        else:
            df["units_pred_avg"] = np.nan
            df["revenue_pred_avg"] = np.nan

        # Alias for helpers that expect 'pred_0.5' (units), clamped
        if "units_pred_0.5" in df.columns and "pred_0.5" not in df.columns:
            df["pred_0.5"] = pd.to_numeric(df["units_pred_0.5"], errors="coerce").clip(
                lower=0.0
            )

        # Ensure every revenue_pred_* is non-negative (double safety)
        for c in [c for c in df.columns if c.startswith("revenue_pred_")]:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0.0)

        return df

    def run(self) -> pd.DataFrame:
        """
        Train per product, predict expectiles, assemble results, and postprocess.
        """
        all_results = []

        for product_name, sub in self.topsellers.groupby("product"):
            sub_clean, X_tr, y_tr, w_tr, X_pred = self._make_design_matrix(sub)
            if sub_clean is None:
                continue

            preds = self._fit_expectiles(X_tr, y_tr, w_tr, X_pred)

            group_df = self._assemble_group_results(sub_clean, preds)
            all_results.append(group_df)

            if self.verbose and hasattr(self, "_gam_best_line"):
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] GAMModeler {self._gam_best_line}",
                    flush=True,
                )

                # reset for next product
                del self._gam_best_line
                self._gam_best_score = float("inf")

        if not all_results:
            return pd.DataFrame()

        all_gam_results = pd.concat(all_results, axis=0, ignore_index=True)
        return self._postprocess_all(all_gam_results)


class Optimizer:
    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        """Calculate weighted predictions based on ratio"""

        if "ratio" in all_gam_results.columns:
            # Normalize ratio to 0-1 range for weighting
            max_ratio = all_gam_results["ratio"].max()
            confidence_weight = 1 - (all_gam_results["ratio"] / max_ratio)
        else:
            confidence_weight = 0.5  # Default to equal weighting

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
        """Prepare core data with time decay weights"""
        topsellers = self.engineer.prepare()

        # Add time decay and calculate elasticity
        topsellers_decayed = self.engineer._time_decay(topsellers)
        elasticity_df = ElasticityAnalyzer.compute(topsellers_decayed)

        # Initialize GAMModeler without elasticity
        modeler = GAMModeler(
            topsellers,
            test_size=0.2,
            split_by_date=True,
            n_bootstrap=5,
            bootstrap_frac=0.8,
            random_state=42,
            verbose=True,
            log_every=2,
        )

        all_gam_results = modeler.run()

        # Merge elasticity data into results
        all_gam_results = all_gam_results.merge(
            elasticity_df[["product", "ratio", "elasticity_score"]],
            on="product",
            how="left",
        )

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

        # Actual daily revenue
        den = pd.to_numeric(df.get("days_sold"), errors="coerce")
        den = den.replace(0, np.nan)
        act_rev = pd.to_numeric(df.get("revenue_actual"), errors="coerce")
        daily_act = (act_rev / den).where(den.notna(), act_rev)

        out = {}

        # P50 metrics
        pred50_rev = pd.to_numeric(df.get("revenue_pred_0.5"), errors="coerce")
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
        if all_gam_results is None or all_gam_results.empty:
            return self.empty_fig("No model data")

        need_cols = [
            "product",
            "asp",
            "revenue_actual",
            "revenue_pred_0.025",
            "revenue_pred_0.5",
            "revenue_pred_0.975",
        ]
        for c in need_cols:
            if c not in all_gam_results.columns:
                return self.empty_fig(f"Missing column: {c}")

        product_lst = all_gam_results["product"].dropna().unique()
        pltly_qual = px.colors.qualitative.Dark24
        pltly_qual.extend(px.colors.qualitative.Vivid)
        colors = random.sample(pltly_qual, len(product_lst))
        color_dct = {p: colors[i] for i, p in enumerate(product_lst)}

        fig = go.Figure()

        for group_name, g in all_gam_results.groupby("product"):
            g = g.dropna(subset=["asp"]).copy()
            if g.empty:
                continue

            # Actual revenue points
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_actual"],
                    mode="markers",
                    opacity=0.5,
                    name=f"{group_name} • Actual Revenue",
                    marker=dict(color=color_dct[group_name], size=8, symbol="circle"),
                    legendgroup=group_name,
                    hovertemplate="ASP=%{x:$,.2f}<br>Actual Rev=%{y:$,.0f}<extra></extra>",
                )
            )

            # Revenue band (P2.5-P97.5)
            fig.add_trace(
                go.Scatter(
                    name=f"{group_name} (Rev band)",
                    x=g["asp"].tolist() + g["asp"].tolist()[::-1],
                    y=g["revenue_pred_0.975"].tolist()
                    + g["revenue_pred_0.025"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    legendgroup=group_name,
                    showlegend=False,
                    opacity=0.4,
                )
            )

            # Add markers for recommended (P50), conservative (P2.5) and optimistic (P97.5) prices
            best_rows = {
                "Recommended (P50)": ("0.5", g.loc[g["revenue_pred_0.5"].idxmax()]),
                "Conservative (P2.5)": (
                    "0.025",
                    g.loc[g["revenue_pred_0.025"].idxmax()],
                ),
                "Optimistic (P97.5)": (
                    "0.975",
                    g.loc[g["revenue_pred_0.975"].idxmax()],
                ),
            }

            marker_colors = {
                "Conservative (P2.5)": "#1F6FEB",
                "Optimistic (P97.5)": "#238636",
                "Recommended (P50)": "#B82132",
            }

            for label, (quantile, row) in best_rows.items():
                pred_col = f"revenue_pred_{quantile}"
                fig.add_trace(
                    go.Scatter(
                        x=[row["asp"]],
                        y=[row[pred_col]],
                        mode="markers",
                        marker=dict(
                            color=marker_colors[label], size=16, symbol="diamond"
                        ),
                        name=f"{group_name} • {label}",
                        legendgroup=group_name,
                        hovertemplate=f"{label}<br>Price=%{{x:$,.2f}}<br>Rev=%{{y:$,.0f}}<extra></extra>",
                    )
                )

            # Revenue P50 line
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_pred_0.5"],
                    mode="lines",
                    name=f"{group_name} • Expected Revenue (P50)",
                    line=dict(color="#B82132", width=2),
                    legendgroup=group_name,
                    hovertemplate="ASP=%{x:$,.2f}<br>Expected Rev=%{y:$,.0f}<extra></extra>",
                )
            )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            margin=dict(r=8, t=40, b=40, l=60),
            template=self.template,
            yaxis=dict(
                title="Expected Daily Revenue",
                tickprefix="$",
                separatethousands=True,
            ),
            xaxis=dict(
                title="Average Selling Price (ASP)",
                tickprefix="$",
                separatethousands=True,
            ),
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


# if __name__ == "__main__":
#     pricing_df, product_df = pd.read_csv('data/pricing.csv'), pd.read_csv('data/products.csv')

#     PricingPipeline(pricing_df,product_df,top_n=10, use_grid_search=True).assemble_dashboard_frames()


# all_gam_results = GAMModeler(
#     DataEngineer(pricing_df, product_df, top_n=10).prepare()).run()
# # Create a viz instance first, then call the method
# viz_instance = viz()
# viz_instance.gam_results(all_gam_results)
