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
from typing import Optional, Dict, Any
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
    "asp",
    "__intercept__",
]
CAT_COLS = ["event_encoded", "product_encoded"]
NUM_COLS = ["price", "deal_discount_percent"]


DEFAULT_PARAM_SEARCH = dict(
    n_splines_grid=(20, 30, 40),
    loglam_range=(np.log(0.001), np.log(1.0)),
    expectile=(0.025, 0.5, 0.975),
    alpha=0.01,
    random_state=42,
    verbose=False,
)


def _v_best(obj, msg: str):
    """
    Logging func -
    Print only when obj._verbose is True. Use for 'best only' logs.
    """
    if getattr(obj, "_verbose", False):

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
    def __init__(self, decay_rate=-0.05, normalize=True):
        self.decay_rate = decay_rate
        self.normalize = normalize

    def _make_weights(self, df, col="order_date"):
        """Compute exponential time-decay weights based on recency."""
        if col not in df.columns:
            return np.ones(len(df))

        dates = pd.to_datetime(df[col], errors="coerce")
        max_date = dates.max()
        delta_days = (max_date - dates).dt.days.fillna(0)

        # Exponential decay
        w = np.exp(self.decay_rate * delta_days)

        if self.normalize:
            w = w / np.nanmean(w[w > 0])

        return w


class _TuneResult:
    def __init__(
        self,
        n_splines: int,
        lam: float,
        models: Dict[str, Any],
        metrics: Dict[str, float],
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.models = models
        self.metrics = metrics


class ParamSearchCV:
    """
    Handles training and returning trained models + metrics.

    Joint expectile tuning for (0.025, 0.5, 0.975) using interval score + optional width penalty.
    Backward-compatible signature: keeps numeric/categorical preprocessing args, n_splits, lam_iters, weighting.
    """

    def __init__(
        self,
        numeric_cols=None,
        categorical_cols=None,
        n_splits: int = 3,
        n_splines_grid: tuple = (10, 14, 18),
        loglam_range: tuple = (np.log(0.001), np.log(1.0)),
        lam_iters: int = 6,
        expectile=(0.025, 0.5, 0.975),
        alpha: float = 0.10,
        random_state: int = 42,
        verbose: bool = False,
        weighting=None,
        width_penalty: float = 15.5,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.n_splits = int(n_splits)
        self.n_splines_grid = tuple(n_splines_grid)
        self.loglam_range = (float(loglam_range[0]), float(loglam_range[1]))
        self.lam_iters = int(lam_iters)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.weighting = weighting
        self.width_penalty = float(width_penalty)

        # normalize expectiles
        if isinstance(expectile, (list, tuple)):
            es = [float(e) for e in expectile]
        else:
            es = [float(expectile)]
        self.expectiles = tuple(sorted(set(es)))

        # lam grid using lam_iters
        self.lam_grid = np.exp(
            np.linspace(
                self.loglam_range[0], self.loglam_range[1], max(1, self.lam_iters)
            )
        )

        self.best_ = None

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
        self, X_train, y_train, X_val, y_val,
        base_ns, base_lam, expectiles
    ):
        """
        Adaptively tune time-decay rate after best (n_splines, lam) found.
        Re-fits GAMs with decay-weighted samples and evaluates on weighted validation set.
        """
        print("\n🌊 Fine-tuning time-decay rate adaptively...")
        import time
        start = time.time()

        search_min, search_max = -0.20, -0.01
        best_score, best_decay = float("inf"), None

        for step in range(6):  # about 6 adaptive iterations
            candidates = np.linspace(search_min, search_max, num=3)
            scores = []

            for d in candidates:
                # --- Recompute weights for this decay candidate ---
                W = Weighting(decay_rate=d)
                w_train_d = np.asarray(W._make_weights(X_train), dtype=float)
                w_val_d   = np.asarray(W._make_weights(X_val), dtype=float)

                # --- Refit GAM models (same spline and lambda) ---
                preds_val = {}
                for e in expectiles:
                    model = self._fit_one(e, base_ns, base_lam, X_train, y_train, w_train_d)
                    preds_val[e] = model.predict(X_val)

                # --- Weighted validation metrics ---
                p025, p50, p975 = preds_val.get(0.025), preds_val.get(0.5), preds_val.get(0.975)

                # weighted RMSE
                rmse = np.sqrt(np.average((y_val - p50) ** 2, weights=w_val_d))

                # weighted interval score
                interval_width = p975 - p025
                # penalize uncovered points more if they're recent (high weight)
                penalty = np.where(y_val < p025, p025 - y_val, np.where(y_val > p975, y_val - p975, 0))
                iscore = np.average(interval_width + (2 / self.alpha) * penalty, weights=w_val_d)

                score = iscore + rmse
                scores.append((score, d))

                print(f"   → decay={d:6.3f} | score={score:8.2f} | RMSE={rmse:6.2f}")

            # --- Find local best & narrow range ---
            best_local_score, best_local_decay = min(scores, key=lambda x: x[0])
            if best_local_score < best_score:
                best_score, best_decay = best_local_score, best_local_decay

            # adaptive zoom-in around best decay
            span = (search_max - search_min) * 0.4
            search_min = best_local_decay - span / 2
            search_max = best_local_decay + span / 2

        duration = time.time() - start
        print(f"🌟 Optimal decay rate ≈ {best_decay:.4f} (score={best_score:.2f}) "
            f"after {duration:.1f}s\n")

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

    def fit(self, X_train, y_train, X_val=None, y_val=None, w_train=None, w_val=None):
        """
        Try multiple combos of hyperparameters across all expectiles
        & find the best performing model

        Parameters:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            w_train: Sample weights for training
            w_val: Sample weights for validation scoring
        """
        print("🔍 Searching hyperparameter grid...")
        best_pair = None

        # Default to training data if val not provided
        if X_val is None or y_val is None:
            X_val, y_val, w_val = X_train, y_train, w_train

        total = len(self.n_splines_grid) * len(self.lam_grid)
        counter = 0
        for ns in self.n_splines_grid:
            for lam in self.lam_grid:
                counter += 1
                print(
                    f"    → [{counter}/{total}] Trying n_splines={ns}, λ={lam:.4f}",
                    flush=True,
                )
                models, preds_val = {}, {}

                for e in self.expectiles:
                    model = self._fit_one(e, ns, lam, X_train, y_train, w_train)
                    models[e] = model
                    preds_val[e] = model.predict(X_val)

                # Evaluate
                p025, p50, p975 = (
                    preds_val.get(0.025),
                    preds_val.get(0.5),
                    preds_val.get(0.975),
                )
                iscore = self.interval_score(
                    y_val,
                    p025,
                    p975,
                    p50,
                    alpha=self.alpha,
                    width_penalty=self.width_penalty,
                )
                r50 = np.sqrt(np.mean((y_val - p50) ** 2))
                score = iscore + r50
                metrics = {
                    "interval_score": iscore,
                    "rmse_val": r50,
                    "n_splines": ns,
                    "lam": lam,
                }

                if (best_pair is None) or (score < best_pair[0]):
                    best_pair = (score, _TuneResult(ns, lam, models, metrics))
                    print(
                        f"        🌟 New best: n_splines={ns}, λ={lam:.4f} (score={score:.2f})"
                    )

        self.best_ = best_pair[1]
 
        # Adaptive fine-tune of decay
        best_decay, best_score = self._tune_decay(
            X_train, y_train, X_val, y_val,
            base_ns=self.best_.n_splines,
            base_lam=self.best_.lam,
            expectiles=self.expectiles
        )

        self.best_.metrics["decay_rate"] = best_decay
        self.best_.metrics["final_score"] = best_score
        print(f"✅ Best config: n_splines={self.best_.n_splines}, λ={self.best_.lam:.4f}, decay={best_decay:.4f}")
        print(
            f"📈 Validation RMSE={self.best_.metrics['rmse_val']:.2f} | Interval Score={self.best_.metrics['interval_score']:.2f}\n"
        )
        return self.best_

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
        self, X_train, y_train, X_pred, X_val=None, y_val=None, w_train=None, w_val=None
    ):
        """
        Fit the best GAMs for each expectile and generate predictions for X_pred.

        Parameters
            X_train : pd.DataFrame or np.ndarray
                Training feature matrix.
            y_train : array-like
                Training target values.
            X_pred : pd.DataFrame or np.ndarray
                Feature matrix to predict on (usually same as X_train for in-sample plots).
            w_train : array-like, optional
                Sample weights (time-decay, rarity, etc).

        Returns
            dict with keys:
            - "params":  tuned parameters (n_splines, lam)
            - "metrics": model performance summary (interval score, RMSE, etc.)
            - "predictions": dict of arrays keyed by "units_pred_<expectile>"
        """

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

        self._log(
            f"[DBG GAMModeler] Using ParamSearchCV.best_ = {best}\n"
            f"    Validation metrics: {getattr(best, 'metrics', {})}\n"
            f"    Prediction keys: {list(preds.keys())}"
        )

        return {
            "params": getattr(best, "params", {}),
            "metrics": getattr(best, "metrics", {}),
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
        self.param_search_kwargs = param_search_kwargs or DEFAULT_PARAM_SEARCH

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

        # --- Weights (decay + rarity) ---
        W = Weighting(decay_rate=-0.05)
        w_raw = np.asarray(W._make_weights(topsellers), dtype=float)
        rarity = 1.0 / (1.0 + topsellers.groupby("price")["price"].transform("count"))
        w_raw *= rarity.values
        w_stable = np.clip(
            np.where(np.isfinite(w_raw), w_raw, 1.0),
            (
                np.nanquantile(w_raw[np.isfinite(w_raw)], 0.20)
                if np.isfinite(w_raw).any()
                else 0.1
            ),
            None,
        )
        print("⚖️  Weights computed (time-decay × rarity).")

        # Elasticity diagnostic
        try:
            elasticity_df = ElasticityAnalyzer.compute(topsellers)
        except Exception:
            elasticity_df = pd.DataFrame(
                columns=["product", "ratio", "elasticity_score"]
            )

        # --- Feature Matrix ---
        X = topsellers[EXPECTED_COLS].copy()
        y = np.asarray(topsellers["shipped_units"], dtype=int)

        # Scale numeric features
        num_cols = ["price", "deal_discount_percent", "asp"]
        if all(col in X.columns for col in num_cols):
            X[num_cols] = StandardScaler().fit_transform(X[num_cols])

        # Split into training and validation
        print(
            "\n─────────────────────────────── ✂️  Splitting Train / Validation ───────────────────────────────"
        )
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, w_stable, test_size=0.2, random_state=42
        )
        print(
            f"📊 Training set: {len(X_train):,} rows | Validation set: {len(X_val):,} rows"
        )

        # --- Fit + Predict ---
        print(
            "\n─────────────────────────────── ⚙️  Tuning Expectile GAMs (Grid Search) ───────────────────────────────"
        )
        param_search = ParamSearchCV(
            numeric_cols=num_cols,
            categorical_cols=["event_encoded", "product_encoded"],
            **self.param_search_kwargs,
        )
        modeler = GAMModeler(param_search)
        res = modeler.fit_predict_expectiles(
            X_train=X_train,
            y_train=y_train,
            X_pred=X,  # still predict on full set for dashboard
            X_val=X_val,
            y_val=y_val,
            w_train=w_train,
            w_val=w_val,
        )
        print(
            "\n─────────────────────────────── ✅ Generating Prediction Frames ───────────────────────────────"
        )

        # --- Results Assembly ---
        all_gam_results = (
            topsellers[["product", "price", "asin", "asp"]]
            .copy()
            .reset_index(drop=True)
        )

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

        # deal discount
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


# if __name__ == "__main__":
#     pricing_df, product_df = pd.read_csv('data/pricing.csv'), pd.read_csv('data/products.csv')

#     PricingPipeline(pricing_df,product_df,top_n=10, use_grid_search=True).assemble_dashboard_frames()


# all_gam_results = GAMModeler(
#     DataEngineer(pricing_df, product_df, top_n=10).prepare()).run()
# # Create a viz instance first, then call the method
# viz_instance = viz()
# viz_instance.gam_results(all_gam_results)
