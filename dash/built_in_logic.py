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
            tail_q=0.10
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
        random_state: int = 42
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
        X: np.ndarray,               # shape (n, 1) price in col 0
        y: np.ndarray,               # target (e.g., daily revenue or units)
        w: Optional[np.ndarray] = None,     # sample weights (already clipped/normalized)
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
        y_lo = float(np.median(y[p <= q10])) if np.any(p <= q10) else float(np.median(y))
        y_hi = float(np.median(y[p >= q90])) if np.any(p >= q90) else float(np.median(y))

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
        gam = ExpectileGAM(terms, lam=float(lam), expectile=float(expectile),
                        max_iter=int(max_iter), tol=float(tol))
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
            if qs[i] <= qs[i-1]:
                qs[i] = qs[i-1] + 1e-9
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
                X[tr], y[tr],
                None if w is None else w[tr],
                expectile=cfg.get("expectile", self.expectile),
                n_splines=int(cfg["n_splines"]),
                lam=float(cfg["lam"]),
                max_iter=int(cfg.get("max_iter", 6000)),
                tol=float(cfg.get("tol", 2e-4))
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
        phi = (1 + 5 ** 0.5) / 2
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
            
            # Use best-ever solution's parameters as center of search
            current_loglam = np.log(self.best_config_ever['lam'])
            
            # More conservative range updates
            if self.best_score_ever < 1.0:
                range_width = 0.5  # Narrow search around excellent solution
            elif self.best_score_ever < 2.0:
                range_width = 0.7
            else:
                range_width = 1.0
                
            self.loglam_range = (
                max(np.log(50.0), current_loglam - range_width),
                min(np.log(1000.0), current_loglam + range_width)
            )


    def fit(self, X, y, sample_weight=None, verbose=False):
        '''
        
        '''
        current_best_cfg, current_best_sc = None, np.inf
        
        # Regular grid search
        for ns in self.n_splines_grid:
            try:
                cfg, sc = self._golden_search_loglam(
                    X[:, [0]], y, sample_weight,
                    base_cfg=dict(
                        expectile=self.expectile,
                        n_splines=int(ns),
                        max_iter=6000,
                        tol=2e-4
                    )
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
                X[:, [0]], y, sample_weight,
                expectile=self.expectile,
                n_splines=self.best_config_ever['n_splines'],
                lam=self.best_config_ever['lam'],
                max_iter=6000,
                tol=2e-4
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
        tail_strength: float = None,
        tail_p: float = None,
        use_grid_search: bool = True,
        custom_grid: dict = None,
        elasticity_df: pd.DataFrame = None,
        # split + bootstrap controls
        test_size: float = 0.2,
        split_by_date: bool = True,
        n_bootstrap: int = 10,
        bootstrap_frac: float = 0.8,
        random_state: int | None = 42,
        bootstrap_target_rel_se: float | None = 0.04,  # ~4% target; None to disable
        # log
        verbose: bool = False,
        log_every: int = 5,
    ):

        self.topsellers = topsellers if topsellers is not None else pd.DataFrame()
        self.tail_strength = tail_strength
        self.tail_p = tail_p
        self.use_grid_search = use_grid_search
        self.custom_grid = custom_grid or {
            "tail_strength": [0.1, 0.2, 0.3, 0.4, 0.5],
            "tail_p": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
        self.elasticity_df = elasticity_df

        self.test_size = float(test_size)
        self.split_by_date = bool(split_by_date)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_frac = float(bootstrap_frac)
        self.bootstrap_target_rel_se = bootstrap_target_rel_se

        self.random_state = random_state

        self.verbose = bool(verbose)
        self.log_every = int(log_every)

        # propagate verbosity into helpers that print
        self.engineer = DataEngineer(None, None)
        self.engineer._verbose = self.verbose
        
        self.param_search = ParamSearchCV(
            n_splits=4,
            n_splines_grid=(15, 20, 25, 30),  # Updated grid
            loglam_range=(np.log(50.0), np.log(1000.0)),  # Wider range
            lam_iters=6,  # More iterations
            random_state=random_state
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
        if self.use_grid_search:
            ts, tp, _ = self.engineer._grid_search_weights(
                sub, param_grid=self.custom_grid
            )
            return self.engineer._make_weights(
                sub,
                tail_strength=ts,
                tail_p=tp,
                elasticity_df=self.elasticity_df,
            )
        # fallback simple scheme
        de = DataEngineer(None, None)
        dec = de._time_decay(sub)["time_decay_weight"].astype(float).to_numpy()
        vol = sub["shipped_units"].astype(float)
        volume_weight = np.log1p(vol) / np.log1p(max(vol.max(), 1.0))
        if sub["price"].max() > sub["price"].min():
            dens = sub["price"].map(sub["price"].value_counts()) / len(sub)
            price_weight = 1.0 / (dens + 0.1)
            price_weight = price_weight / price_weight.max()
        else:
            price_weight = pd.Series(1.0, index=sub.index)
        w = 0.4 * dec + 0.4 * volume_weight + 0.2 * price_weight
        return (w / w.mean()).to_numpy()

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

    # ---------- _robust_reweight ----------
    def _robust_reweight(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_base: np.ndarray,
        eps_rows_mask: np.ndarray | None = None,
        expectile: float = 0.5,
        kfold_min: int = 3,
        kfold_max: int = 5,
        huber_c: float = 1.345,
        clip_floor: float = 0.25,
        eps: float = 1e-8,
        fac_col: int = 2,
    ) -> np.ndarray:
        X, y, w_base = (
            np.asarray(X, float),
            np.asarray(y, float),
            np.asarray(w_base, float),
        )
        if len(y) < 10:
            return w_base

        X_eps, y_eps, w_eps = self._rr_build_eps_rows(X, y, eps, fac_col)
        kf = self._rr_make_kfolds(len(y), kfold_min, kfold_max)
        yhat = self._rr_oof_predict(
            X, y, w_base, X_eps, y_eps, w_eps, kf, expectile, fac_col
        )

        w_new = self._rr_huber(y, yhat, huber_c, clip_floor, eps_rows_mask)
        mu = float(w_new.mean())
        return (w_new / mu) if mu > 0 else w_new

    # ---- helpers ----
    def _rr_build_eps_rows(self, X, y, eps, fac_col):
        all_lvls = np.unique(X[:, fac_col])
        price_med = float(np.median(X[:, 0])) if X.shape[1] > 0 else 0.0
        disc_med = float(np.median(X[:, 1])) if X.shape[1] > 1 else 0.0
        X_eps = np.column_stack(
            [
                np.full(all_lvls.size, price_med, float),
                np.full(all_lvls.size, disc_med, float),
                all_lvls.astype(float),
            ]
        )
        y_eps = np.full(all_lvls.size, float(np.median(y)), float)
        w_eps = np.full(all_lvls.size, float(eps), float)
        return X_eps, y_eps, w_eps

    def _rr_make_kfolds(self, n, kmin, kmax):
        k = min(max(kmin, (n // 50) or kmin), kmax)
        return KFold(n_splits=k, shuffle=True, random_state=self.random_state)

    def _rr_oof_predict(
        self, X, y, w_base, X_eps, y_eps, w_eps, kf, expectile, fac_col
    ):
        yhat = np.full(len(y), np.nan, float)
        for tr_idx, va_idx in kf.split(X):
            X_tr = np.vstack([X[tr_idx], X_eps])
            y_tr = np.concatenate([y[tr_idx], y_eps])
            w_tr = np.concatenate([w_base[tr_idx], w_eps])
            w_tr = np.maximum(w_tr, 1e-12)
            tuner = GAMTuner(expectile=expectile)
            tuner._verbose = False
            gam = tuner.fit(X_tr, y_tr, sample_weight=w_tr)
            yhat[va_idx] = self._rr_predict_with_gam(gam, X[va_idx])
        return yhat

    def _rr_predict_with_gam(self, gam, Xv):
        Xv = np.asarray(Xv, float)
        Xv_num = gam._scaler.transform(Xv[:, gam._num_idx])
        Xv_fac = Xv[:, gam._fac_idx]
        Xv_s = np.column_stack([Xv_num, Xv_fac])
        return gam.predict(Xv_s)

    def _rr_huber(self, y, yhat, huber_c, clip_floor, eps_rows_mask):
        real_mask = np.ones(len(y), bool) if eps_rows_mask is None else ~eps_rows_mask
        r = y[real_mask] - yhat[real_mask]
        med_abs = float(np.median(np.abs(r - np.median(r))))
        s = 1.4826 * max(med_abs, 1e-9)
        z = np.abs(r) / s
        w_resid_real = np.where(z <= huber_c, 1.0, huber_c / z)
        w_resid = np.ones(len(y), float)
        w_resid[real_mask] = np.clip(w_resid_real, clip_floor, 1.0)
        return w_resid * 1.0  # scale later

    # ---------- _fit_expectiles ----------
    def _fit_expectiles(
        self,
        X_train,
        y_train,
        w_train,
        X_pred,
        qs=(0.025, 0.5, 0.975),
        y_all=None,
        split_mask=None,
        taper_grid=None,
    ) -> dict:
        """
        Orchestrates fitting of multiple expectiles with pre/post processing;
        handles the overall workflow, data preparation, and post-processing.
        """
        # 1) Data preparation
        X_seed, y_seed, w_seed = self._fe_seed_domain(X_train, y_train, w_train, X_pred)
        is_eps = w_seed <= 2e-8
        w_robust = self._robust_reweight(
            X_seed, y_seed,
            w_seed if w_seed is not None else np.ones(len(y_seed)),
            eps_rows_mask=is_eps,
            expectile=0.5
        )

        # 2) Core fitting - use _fe_fit_one_expectile for each q
        preds = {}
        for q in qs:
            preds.update(
                self._fe_fit_one_expectile(q, X_seed, y_seed, X_pred, w_robust)
            )

        # 3) Post-processing
        if y_all is not None and split_mask is not None:
            best_taper = self._fe_pick_taper(
                X_train, X_pred, preds, y_all, 
                ~np.asarray(split_mask, bool),
                w_train, taper_grid
            )
        else:
            best_taper = None

        preds = self._apply_edge_tapers(
            X_train=X_train,
            X_pred=X_pred,
            pred_dict=preds,
            w_train_effective=w_train,
            **(best_taper or dict(
                lo_q=0.10, hi_q=0.90,
                stretch_lo=0.15, stretch_hi=0.15,
                k_anchor=5, scale_sds=True,
            )),
        )
        
        return self._fe_enforce_non_crossing(preds)


    # ---- helpers ----
    def _fe_seed_domain(self, X_train, y_train, w_train, X_pred, fac_col=2, eps=1e-8):
        return self._ensure_factor_domain(
            X_train, y_train, w_train, X_pred, fac_col=fac_col, eps=eps
        )

    def _fe_fit_one_expectile(self, q, X_seed, y_seed, X_pred, w_robust):
        def fit_once(q, boot_idx=None):
            self.param_search.expectile = q
            w_boot = (
                w_robust
                if boot_idx is None
                else (self._bootstrap_weights(len(y_seed), boot_idx) * w_robust)
            )
            
            # Fit using only price column
            best_gam = self.param_search.fit(
                X_seed[:, [0]], y_seed, w_boot,  # Only price column
                verbose=self.verbose
            )
            
            # Predict using only price column
            X_pred_price = X_pred[:, [0]]  # Extract price column
            return np.maximum(best_gam.predict(X_pred_price), 0.0)

        loops = self._bootstrap_loops_for_size(len(y_seed))
        if loops <= 0:
            return {f"units_pred_{q}": fit_once(q)}

        rng = np.random.default_rng(self.random_state)
        idx_all = np.arange(len(y_seed))
        m = max(1, int(len(y_seed) * self.bootstrap_frac))
        preds = []
        target = self.bootstrap_target_rel_se

        for i in range(loops):
            boot_idx = rng.choice(idx_all, size=m, replace=True)
            preds.append(fit_once(q, boot_idx=boot_idx))
            if target and (i + 1) >= 5:
                P = np.vstack(preds)
                mean_vec = P.mean(axis=0)
                se_vec = P.std(axis=0, ddof=1) / np.sqrt(i + 1)
                rel_se = np.nanmean(se_vec / np.maximum(np.abs(mean_vec), 1e-8))
                if rel_se < target:
                    break

        P = np.vstack(preds)
        out = {
            f"units_pred_{q}": P.mean(axis=0),
            f"units_pred_{q}_sd": P.std(axis=0, ddof=1),
        }
        return out
    
    def _fe_pick_taper(
        self, X_train, X_pred, base_pred_dict, y_all, mask_train, w_train, grid
    ):
        # very small, safe defaults if grid not provided
        if grid is None:
            grid = {
                "lo_q": [0.05, 0.10, 0.15],
                "hi_q": [0.85, 0.90, 0.95],
                "stretch_lo": [0.10, 0.15, 0.25],
                "stretch_hi": [0.10, 0.15, 0.25],
                "k_anchor": [3, 5, 7],
                "scale_sds": [True],
            }
        # minimal successive-halving to avoid explosion
        from itertools import product

        keys = list(grid.keys())
        combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]
        rng = np.random.default_rng(self.random_state or 42)
        if len(combos) > 24:
            combos = list(rng.choice(combos, size=24, replace=False))

        def score(params):
            trial = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in base_pred_dict.items()
            }
            trial = self._apply_edge_tapers(
                X_train,
                X_pred,
                trial,
                w_train,
                lo_q=params["lo_q"],
                hi_q=params["hi_q"],
                stretch_lo=params["stretch_lo"],
                stretch_hi=params["stretch_hi"],
                k_anchor=params["k_anchor"],
                scale_sds=params["scale_sds"],
            )
            return self._fe_taper_score(trial, y_all, mask_train)

        best, bestp = np.inf, None
        for p in combos:
            sc = score(p)
            if sc < best:
                best, bestp = sc, p
        return bestp

    def _fe_taper_score(self, preds_dict_after, y_all, mask_train):
        yhat = np.asarray(preds_dict_after.get("units_pred_0.5"), float)
        y = np.asarray(y_all, float)
        mtrain = np.asarray(mask_train, bool)
        if yhat.size != y.size or (~np.isfinite(yhat)).any():
            return np.inf
        rmse = float(np.sqrt(np.mean((y[mtrain] - yhat[mtrain]) ** 2)))

        # jitter penalty
        def _avg_abs_second_diff(z):
            if z.size < 5:
                return 0.0
            d2 = np.diff(z, 2)
            denom = max(np.median(np.abs(z)), 1e-8)
            return float(np.mean(np.abs(d2)) / denom)

        jitter = _avg_abs_second_diff(yhat)
        # edge-lift penalty
        n = len(yhat)
        if n >= 20:
            k = max(1, n // 20)
            left = np.mean(yhat[:k])
            right = np.mean(yhat[-k:])
            core = yhat[k:-k] if (n - 2 * k) > 5 else yhat
            core_med = float(np.median(core))
            lift = max(0.0, left / core_med - 1.0) + max(0.0, right / core_med - 1.0)
            edge_pen = 0.5 * lift
        else:
            edge_pen = 0.0
        return rmse * (1 + 0.6 * jitter + 0.8 * edge_pen)

    def _fe_enforce_non_crossing(self, pred_dict: dict) -> dict:
        qkeys = sorted(
            [
                k
                for k in pred_dict
                if k.startswith("units_pred_") and not k.endswith("_sd")
            ],
            key=lambda k: float(k.replace("units_pred_", "")),
        )
        if not qkeys:
            return pred_dict
        P = np.vstack([pred_dict[k] for k in qkeys])
        P_nc = np.maximum.accumulate(P, axis=0)
        for i, k in enumerate(qkeys):
            pred_dict[k] = np.maximum(P_nc[i], 0.0)
        return pred_dict

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

    def _apply_edge_tapers(
        self,
        X_train: np.ndarray,
        X_pred: np.ndarray,
        pred_dict: dict,
        w_train_effective: np.ndarray | None = None,
        lo_q: float = 0.10,
        hi_q: float = 0.90,
        stretch_lo: float = 0.15,
        stretch_hi: float = 0.15,
        k_anchor: int = 5,
        scale_sds: bool = True,
    ):
        px_tr = np.asarray(X_train, float)[:, 0]
        px_pr = np.asarray(X_pred, float)[:, 0]
        pr = float(px_tr.max() - px_tr.min())
        if not np.isfinite(pr) or pr <= 0:
            return pred_dict

        lo_anchor, hi_anchor = self._tap_anchors(px_tr, w_train_effective, lo_q, hi_q)
        if not (
            np.isfinite(lo_anchor) and np.isfinite(hi_anchor) and hi_anchor > lo_anchor
        ):
            return pred_dict

        x_left0, x_right0 = self._tap_endpoints(
            px_tr, lo_anchor, hi_anchor, stretch_lo, stretch_hi
        )
        mL, mR, L, R = self._tap_ramps(px_pr, lo_anchor, hi_anchor, x_left0, x_right0)

        L_idx = self._tap_anchor_indices(px_pr, lo_anchor, side="left", k=k_anchor)
        R_idx = self._tap_anchor_indices(px_pr, hi_anchor, side="right", k=k_anchor)

        qkeys = sorted(
            [
                k
                for k in pred_dict
                if k.startswith("units_pred_") and not k.endswith("_sd")
            ],
            key=lambda k: float(k.replace("units_pred_", "")),
        )
        for k in qkeys:
            y = np.asarray(pred_dict[k], float)
            y_new, shrink = self._tap_apply_one(y, mL, mR, L, R, L_idx, R_idx)
            pred_dict[k] = np.maximum(y_new, 0.0)
            if scale_sds:
                sk = f"{k}_sd"
                if sk in pred_dict:
                    sd = np.asarray(pred_dict[sk], float)
                    pred_dict[sk] = np.maximum(sd * shrink, 0.0)

        # enforce non-crossing again
        qkeys = sorted(
            [
                k
                for k in pred_dict
                if k.startswith("units_pred_") and not k.endswith("_sd")
            ],
            key=lambda k: float(k.replace("units_pred_", "")),
        )
        if qkeys:
            P = np.vstack([pred_dict[k] for k in qkeys])
            P_nc = np.maximum.accumulate(P, axis=0)
            for i, k in enumerate(qkeys):
                pred_dict[k] = np.maximum(P_nc[i], 0.0)

        pred_dict["_edge_taper_info"] = {
            "lo_anchor": float(lo_anchor),
            "hi_anchor": float(hi_anchor),
            "x_left0": float(x_left0),
            "x_right0": float(x_right0),
        }
        return pred_dict

    # ---- helpers ----
    def _tap_wquantile(self, v, q, w=None) -> float:
        v = np.asarray(v, float)
        if w is None:
            w = np.ones_like(v)
        else:
            w = np.asarray(w, float)
            if w.shape[0] != v.shape[0]:
                w = np.ones_like(v)
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not ok.any():
            return float("nan")
        v, w = v[ok], w[ok]
        order = np.argsort(v)
        v, w = v[order], w[order]
        cw = np.cumsum(w)
        cutoff = float(q) * float(cw[-1])
        return float(v[np.searchsorted(cw, cutoff, side="left")])

    def _tap_anchors(self, px_tr, w_eff, lo_q, hi_q):
        return (
            self._tap_wquantile(px_tr, lo_q, w_eff),
            self._tap_wquantile(px_tr, hi_q, w_eff),
        )

    def _tap_endpoints(self, px_tr, lo_anchor, hi_anchor, stretch_lo, stretch_hi):
        pr = float(px_tr.max() - px_tr.min())
        x_left0 = lo_anchor - stretch_lo * pr
        x_right0 = hi_anchor + stretch_hi * pr
        if x_right0 <= hi_anchor:
            x_right0 = hi_anchor + 1e-6
        if x_left0 >= lo_anchor:
            x_left0 = lo_anchor - 1e-6
        return x_left0, x_right0

    def _tap_ramps(self, px_pr, lo_anchor, hi_anchor, x_left0, x_right0):
        mL = px_pr <= lo_anchor
        mR = px_pr >= hi_anchor
        L = np.zeros_like(px_pr, float)
        R = np.zeros_like(px_pr, float)
        if np.any(mL):
            L[mL] = np.clip((px_pr[mL] - x_left0) / (lo_anchor - x_left0), 0.0, 1.0)
        if np.any(mR):
            R[mR] = np.clip((x_right0 - px_pr[mR]) / (x_right0 - hi_anchor), 0.0, 1.0)
        return mL, mR, L, R

    def _tap_anchor_indices(self, px_pr, anchor, side, k):
        if side == "left":
            inside = np.where(px_pr >= anchor)[0]
            if inside.size:
                return inside[:k]
            j = np.searchsorted(px_pr, anchor, side="left")
            j = min(max(j, 0), len(px_pr) - 1)
            return np.array([j], int)
        else:
            inside = np.where(px_pr <= anchor)[0]
            if inside.size:
                return inside[max(0, inside.size - k) :]
            j = np.searchsorted(px_pr, anchor, side="right") - 1
            j = min(max(j, 0), len(px_pr) - 1)
            return np.array([j], int)

    def _tap_apply_one(self, y, mL, mR, L, R, L_idx, R_idx):
        yL = float(np.median(y[L_idx])) if L_idx.size else 0.0
        yR = float(np.median(y[R_idx])) if R_idx.size else 0.0
        capL = np.maximum(yL * L, 0.0)
        capR = np.maximum(yR * R, 0.0)
        y_new = y.copy()
        if np.any(mL):
            y_new[mL] = np.minimum(y_new[mL], capL[mL])
        if np.any(mR):
            y_new[mR] = np.minimum(y_new[mR], capR[mR])
        if np.any(mR):
            idxR = np.where(mR)[0]
            y_new[idxR] = np.minimum.accumulate(y_new[idxR])
        if np.any(mL):
            idxL = np.where(mL)[0]
            y_new[idxL] = np.minimum.accumulate(y_new[idxL][::-1])[::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            shrink = np.minimum(y_new / np.maximum(y, 1e-9), 1.0)
        return y_new, shrink

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
    def __init__(
        self, pricing_df, product_df, top_n=10, use_grid_search=True, custom_grid=None
    ):
        self.engineer = DataEngineer(pricing_df, product_df, top_n)
        self.use_grid_search = use_grid_search
        self.custom_grid = custom_grid

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

    # -----------------------------
    # Small helpers (pure functions)
    # -----------------------------
    def _build_curr_price_df(self) -> pd.DataFrame:
        """current price df with product tag"""
        if "current_price" not in self.engineer.product_df.columns:
            # Create empty DataFrame with required columns
            return pd.DataFrame(columns=["asin", "product", "current_price"])

        product = self.engineer.product_df.copy()
        product["product"] = DataEng.compute_product_series(product)

        # Keep only the required columns and ensure current_price exists
        out = product[["asin", "product", "current_price"]].copy()
        out["current_price"] = pd.to_numeric(out["current_price"], errors="coerce")

        return out.reset_index(drop=True)

    def _build_core_frames(self):
        """Prepare core data incorporating elasticity"""
        topsellers = self.engineer.prepare()
        topsellers_decayed = self.engineer._time_decay(topsellers)

        # Get elasticity metrics
        elasticity_df = ElasticityAnalyzer.compute(topsellers_decayed)

        # Initialize GAMModeler with elasticity data
        modeler = GAMModeler(
            topsellers,
            elasticity_df=elasticity_df,
            use_grid_search=self.use_grid_search,
            custom_grid=self.custom_grid,
            test_size=0.2,
            split_by_date=True,
            n_bootstrap=5,
            bootstrap_frac=0.8,
            random_state=42,
            verbose=True,
            log_every=2,  # print every 5 bootstrap iters
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

    def _compute_date_meta(self):
        """Compute data range and annualization factor using the pipeline's pricing_df."""
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
        annual_factor = (365.0 / max(1, days_covered)) if days_covered else 1.0
        meta = {
            "data_start": data_start,
            "data_end": data_end,
            "days_covered": days_covered,
        }
        return meta

    def _build_best50(self, all_gam_results):
        """Pick best P50 revenue row per product for KPIs and scenarios."""
        if {"revenue_pred_0.5", "units_pred_0.5"}.issubset(all_gam_results.columns):
            idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
            best50 = (
                all_gam_results.loc[
                    idx,
                    [
                        "product",
                        "asin",
                        "price",
                        "asp",  # used by viz/helpers
                        "units_pred_0.5",  # P50 units
                        "revenue_pred_0.5",  # P50 revenue
                        "pred_0.5",  # alias (units)
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

    def _build_opportunity_summary(self, best50, all_gam_results, curr_price_df):
        """Compute per-product upside at recommended vs current, plus annualization."""
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

    def _normalize_key_types(self, *dfs):
        """Ensure asin is str across frames."""
        for df in dfs:
            if isinstance(df, pd.DataFrame) and "asin" in df.columns:
                df["asin"] = df["asin"].astype(str)

    def _pack_frames(
        self,
        topsellers,
        best_avg,
        all_gam_results,
        elasticity_df,
        curr_price_df,
        opps_summary,
        best50,
        meta,
    ):

        optimizer_results = Optimizer.run(all_gam_results)
        # all_gam_results now has 'weighted_pred'
        kpi = self._compute_model_fit_kpi(optimizer_results["all_gam_results"])

        return {
            "price_quant_df": (
                topsellers.groupby(["price", "product"])["shipped_units"]
                .sum()
                .reset_index()
            ),
            "best_avg": best_avg,
            "best_weighted": optimizer_results["best_weighted"],
            "all_gam_results": optimizer_results[
                "all_gam_results"
            ],  # keep the augmented frame
            "best_optimal_pricing_df": best50.copy(),
            "elasticity_df": elasticity_df[["product", "ratio", "elasticity_score"]],
            "curr_opt_df": best_avg,
            "curr_price_df": curr_price_df,
            "opps_summary": opps_summary,
            "meta": meta,
            "model_fit_kpi": kpi,  # <-- NEW
        }

    def _compute_model_fit_kpi(
        self, all_gam_results: pd.DataFrame
    ) -> tuple[dict, pd.DataFrame]:
        """
        Returns:
            model_fit_kpi: dict with overall % diffs for P50 and Weighted
            model_fit_by_product: DataFrame with per-product % diffs
        Notes:
            • Signed % diff = (Σ(pred_daily) - Σ(actual_daily)) / Σ(actual_daily) * 100
            • Uses only rows with positive actual daily revenue
            • Uses test split if present; else all rows
        """
        df = all_gam_results.copy()

        # Actual daily revenue
        den = pd.to_numeric(df.get("days_sold"), errors="coerce")
        den = den.replace(0, np.nan)
        act_rev = pd.to_numeric(df.get("revenue_actual"), errors="coerce")
        daily_act = (act_rev / den).where(den.notna(), act_rev)

        out = {}

        # ----- P50 -----
        pred50_rev = pd.to_numeric(df.get("revenue_pred_0.5"), errors="coerce")
        daily_pred50 = (pred50_rev / den).where(den.notna(), pred50_rev)

        mask50 = (daily_act > 0) & daily_pred50.notna()
        w50 = act_rev.where(mask50, np.nan)  # revenue-weighted
        pct50 = (daily_pred50 - daily_act) / daily_act

        # If absolute, do: pct50 = np.abs(pct50)
        out["pct_diff_p50"] = (
            float(np.nansum(pct50 * w50) / np.nansum(w50))
            if np.nansum(w50) > 0
            else np.nan
        )

        # ----- Weighted (units blend -> revenue) -----
        if "weighted_pred" in df.columns:
            rev_w = pd.to_numeric(df["weighted_pred"] * df["price"], errors="coerce")
            daily_pred_w = (rev_w / den).where(den.notna(), rev_w)

            maskw = (daily_act > 0) & daily_pred_w.notna()
            ww = act_rev.where(maskw, np.nan)
            pctw = (daily_pred_w - daily_act) / daily_act
            # If you want absolute, do: pctw = np.abs(pctw)
            out["pct_diff_weighted"] = (
                float(np.nansum(pctw * ww) / np.nansum(ww))
                if np.nansum(ww) > 0
                else np.nan
            )
        else:
            out["pct_diff_weighted"] = np.nan

        return out

    # -----------------------------
    # Public API
    # -----------------------------
    def assemble_dashboard_frames(self) -> dict:
        # 1) core
        topsellers, elasticity_df, all_gam_results = self._build_core_frames()

        # 2) optimizer tables
        best_avg = self._compute_best_tables(all_gam_results, topsellers)

        # 3) current prices
        curr_price_df = self._build_curr_price_df()

        # 4) meta (dates only, no annual_factor)
        meta = self._compute_date_meta()

        # 5) best-50 (P50 revenue)
        best50 = self._build_best50(all_gam_results)

        # 6) opportunities - remove annual_factor parameter
        opps_summary = self._build_opportunity_summary(
            best50, all_gam_results, curr_price_df  # Removed annual_factor
        )

        # 7) normalize key dtype across frames
        self._normalize_key_types(
            best_avg, all_gam_results, curr_price_df, topsellers, opps_summary, best50
        )

        # 8) pack and return
        return self._pack_frames(
            topsellers=topsellers,
            best_avg=best_avg,
            all_gam_results=all_gam_results,
            elasticity_df=elasticity_df,
            curr_price_df=curr_price_df,
            opps_summary=opps_summary,
            best50=best50,
            meta=meta,
        )


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

