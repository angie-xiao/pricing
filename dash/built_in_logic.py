# --------- built_in_logic.py  ---------
# (RMSE-focused; Top-N only; adds annualized opps & data range)
from __future__ import annotations
import random
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Iterable, Union

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
from pygam import ExpectileGAM, s, l
from sklearn.ensemble import GradientBoostingRegressor

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
    def __init__(
        self,
        decay_rate: float = -0.015,   # negative → older rows get smaller weight
        rarity_cap: float = 1.15,
        rarity_beta: float = 0.25,
        clip_min: float = 0.35,
        clip_max: float = 2.0,
    ):
        self.decay_rate = float(decay_rate)
        self.rarity_cap = float(rarity_cap)
        self.rarity_beta = float(rarity_beta)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def _time_decay(self, df: pd.DataFrame) -> np.ndarray:
        """
        Exponential time-decay using 'order_date' if present.
        weight = exp(decay_rate * days_since_order)
        Missing/invalid dates → 1.0
        """
        if df is None or df.empty:
            return np.array([], dtype=float)

        if "order_date" not in df.columns:
            return np.ones(len(df), dtype=float)

        odt = pd.to_datetime(df["order_date"], errors="coerce")
        if odt.notna().any():
            ref = odt.max().normalize()
            days = (ref - odt).dt.days.astype("float64")
            days = np.where(np.isfinite(days), days, 0.0)
        else:
            return np.ones(len(df), dtype=float)

        return np.exp(self.decay_rate * days).astype(float)

    def _rarity_multiplier(self, prices: np.ndarray) -> np.ndarray:
        """
        Rarity = inverse frequency at (rounded) price.
        Round to cents to avoid fragmentation.
        """
        p = pd.to_numeric(prices, errors="coerce").astype(float)
        if p.size == 0:
            return np.array([], dtype=float)

        bucketed = np.round(p, 2)
        vc = pd.Series(bucketed).value_counts(dropna=False)
        counts = pd.Series(bucketed).map(vc).to_numpy(dtype=float)

        valid = np.isfinite(counts) & (counts > 0)
        if not valid.any():
            return np.ones_like(counts, dtype=float)

        mean_c = counts[valid].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = mean_c / counts  # fewer transactions → bigger weight
        raw = np.where(np.isfinite(raw), raw, 1.0)

        w = np.power(raw, self.rarity_beta)
        w = np.clip(w, 1.0 / self.rarity_cap, self.rarity_cap)
        return np.where(np.isfinite(w), w, 1.0).astype(float)

    def _make_weights(self, df: pd.DataFrame, base_weights: Optional[np.ndarray] = None) -> np.ndarray:
        decay = self._time_decay(df)
        prices = pd.to_numeric(df.get("price", np.nan), errors="coerce").to_numpy()
        rarity = self._rarity_multiplier(prices)

        w = decay * rarity
        if base_weights is not None:
            w = w * np.asarray(base_weights, dtype=float)

        w = np.nan_to_num(w, nan=1.0, posinf=self.clip_max, neginf=self.clip_min)
        w = np.clip(w, self.clip_min, self.clip_max)
        mean = w.mean() if w.size else 1.0
        w = w / (mean if mean else 1.0)
        return w.astype(float)


# -------------------- ParamSearchCV with interval scoring (updated) --------------------
def interval_score(y_true, y_low, y_up, alpha=0.05, width_penalty=0.5) -> float:
    y_true = np.asarray(y_true, float)
    L = np.asarray(y_low, float)
    U = np.asarray(y_up, float) 
    width = U - L
    below = (L - y_true) * (y_true < L)
    above = (y_true - U) * (y_true > U)
    score = width * (1 + width_penalty) + (2.0/alpha) * (below + above)
    return float(np.mean(score))

    

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class _TuneResult:
    def __init__(self, n_splines: int, lam: float, models: Dict[str, Any], metrics: Dict[str, float]):
        self.n_splines = n_splines
        self.lam = lam
        self.models = models
        self.metrics = metrics


class ParamSearchCV:
    """
    Joint expectile tuning for (0.025, 0.5, 0.975) using interval score + optional width penalty.
    Backward-compatible signature: keeps numeric/categorical preprocessing args, n_splits, lam_iters, weighting.
    """
    def __init__(
        self,
        numeric_cols=None,
        categorical_cols=None,
        n_splits: int = 3,
        n_splines_grid: tuple = (10,14,18), 
        loglam_range: tuple = (np.log(0.05), np.log(5.0)), # Increased upper bound
        lam_iters: int = 6,
        expectile=(0.025, 0.5, 0.975),
        alpha: float = 0.10,
        random_state: int = 42,
        verbose: bool = False,
        weighting=None,
        width_penalty: float = 0.5,
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
        self.lam_grid = np.exp(np.linspace(self.loglam_range[0], self.loglam_range[1], max(1, self.lam_iters)))

        self.best_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _fit_one(self, e, ns, lam, X, y, w=None):
        # gam = ExpectileGAM(expectile=e, lam=lam, terms=s(0, n_splines=ns))
        gam = ExpectileGAM(
            expectile=e,
            lam=lam,
            terms=s(0, n_splines=ns, basis='ps'), 
            max_iter=500,  # Increased iterations
            tol=1e-4      # Tighter convergence
        )
        gam.fit(X, y, weights=w)
        return gam

    def fit(self, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
        # honor user-provided weighting object if given; otherwise use passed arrays
        weights = w_train

        use_joint = (len(self.expectiles) >= 3 and 0.025 in self.expectiles and 0.5 in self.expectiles and 0.975 in self.expectiles)
        best_pair = None

        for ns in self.n_splines_grid:
            for lam in self.lam_grid:
                if use_joint:
                    m025 = self._fit_one(0.025, ns, lam, X_train, y_train, weights)
                    m050 = self._fit_one(0.50,  ns, lam, X_train, y_train, weights)
                    m975 = self._fit_one(0.975, ns, lam, X_train, y_train, weights)

                    p025_val = m025.predict(X_val)
                    p50_val  = m050.predict(X_val)
                    p975_val = m975.predict(X_val)

                    iscore = interval_score(y_val, p025_val, p975_val, alpha=self.alpha)
                    width = float(np.mean(p975_val - p025_val))
                    cov = float(np.mean((y_val >= p025_val) & (y_val <= p975_val)))
                    r50 = rmse(y_val, p50_val)
                    score = iscore + self.width_penalty * width

                    models = {"p025": m025, "p50": m050, "p975": m975}
                    metrics = {"interval_score": iscore, "rmse_p50": r50, "coverage": cov, "width": width}
                    self._log(f"[TUNE] ns={ns} lam={lam:.4g} interval={iscore:.4g} width={width:.0f} cov={cov:.3f} rmse50={r50:.4g}")
                else:
                    e = self.expectiles[0]
                    m = self._fit_one(e, ns, lam, X_train, y_train, weights)
                    p_val = m.predict(X_val)
                    r = rmse(y_val, p_val)
                    score = r
                    key = "p50" if abs(e - 0.5) < 1e-6 else f"e{e}"
                    models = {key: m}
                    metrics = {"rmse": r}
                    self._log(f"[TUNE] ns={ns} lam={lam:.4g} rmse={r:.4g}")

                if (best_pair is None) or (score < best_pair[0]):
                    best_pair = (score, type("_TuneResult", (), {"n_splines": ns, "lam": float(lam), "models": models, "metrics": metrics})())

        self.best_ = best_pair[1]
        return self.best_

    def predict_expectiles(self, X):
        if self.best_ is None:
            raise RuntimeError("ParamSearchCV has not been fit yet.")
        return {k: m.predict(X) for k, m in self.best_.models.items()}

    def fit_predict_expectiles(self, X_train, y_train, X_val, y_val, X_pred, w_train=None, w_val=None):
        best = self.fit(X_train, y_train, X_val, y_val, w_train=w_train, w_val=w_val)
        preds = self.predict_expectiles(X_pred)
        return {"params": {"n_splines": best.n_splines, "lam": best.lam}, "metrics": best.metrics, "pred": preds}


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
        Fit expectile GAMs via ParamSearchCV using a simple temporal holdout (or index split if no date),
        then predict expectiles on X_pred.
        """
        # ---- 1) Prepare arrays and optional weights ----
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train, dtype=float).ravel()
        X_pr = np.asarray(X_pred, dtype=float)
        sw = None
        if w_train is not None:
            sw = np.asarray(w_train, dtype=float)
            if sw.shape[0] != y_tr.shape[0]:
                sw = None  # guard against mismatched lengths

        # ---- 2) Build a validation split ----
        n = X_tr.shape[0]
        if n >= 10:
            # Prefer temporal split if 'order_date' available via DataEngineer
            try:
                # Attempt to sort by order_date from self.param_search.weighting DF if present
                # Fall back to index-based 80/20 split
                idx = np.arange(n)
                cut = max(1, int(0.8 * n))
                tr_idx, va_idx = idx[:cut], idx[cut:]
            except Exception:
                idx = np.arange(n); cut = max(1, int(0.8 * n)); tr_idx, va_idx = idx[:cut], idx[cut:]
        else:
            # Tiny sample: use leave-one-out style single val
            idx = np.arange(n)
            tr_idx, va_idx = idx[:-1], idx[-1:]

        X_trn, y_trn = X_tr[tr_idx], y_tr[tr_idx]
        X_val, y_val = X_tr[va_idx], y_tr[va_idx]
        w_trn = sw[tr_idx] if sw is not None else None
        w_val = sw[va_idx] if sw is not None else None

        # ---- 3) Tune and predict via ParamSearchCV ----
        ps = self.param_search
        ps.fit(X_trn, y_trn, X_val, y_val, w_train=w_trn, w_val=w_val)
        pred_map = ps.predict_expectiles(X_pr)  # keys like 'p025','p50','p975'

        # Normalize keys to your expected names
        preds = {}
        for k, arr in pred_map.items():
            if k == 'p50' or k == 'e0.5':
                preds['units_pred_0.5'] = np.clip(np.asarray(arr, dtype=float), 0.0, None)
            elif k == 'p025' or k == 'e0.025':
                preds['units_pred_0.025'] = np.clip(np.asarray(arr, dtype=float), 0.0, None)
            elif k == 'p975' or k == 'e0.975':
                preds['units_pred_0.975'] = np.clip(np.asarray(arr, dtype=float), 0.0, None)

        return {"predictions": preds}

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
            model = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, **base_kwargs
            )
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
            n_splines_grid=(10,14,18),
            loglam_range=(np.log(0.05), np.log(5.0)), # or even up to np.log(10.0)
            expectile=(0.025, 0.5, 0.975),
            alpha=0.10,               
            width_penalty=0.5,       # try 0.2–0.5; raise if bands still too wide
            verbose=True,
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
        if "asp" not in topsellers.columns and "price" in topsellers.columns:
            topsellers["asp"] = pd.to_numeric(topsellers["price"], errors="coerce")
        if "__intercept__" not in topsellers.columns:
            topsellers["__intercept__"] = 1.0

        # 2) Weights for training / elasticity hts(topsellers), dtype=float)
        W = Weighting(decay_rate=-0.01)  # pick your decay_rate
        topsellers["time_weight"] = W._time_decay(topsellers)

        w_raw = np.asarray(W._make_weights(topsellers), dtype=float)

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
            elasticity_df = pd.DataFrame(
                columns=["product", "ratio", "elasticity_score"]
            )

        # 4) Design matrix and target strictly from EXPECTED_COLS
        X = topsellers[EXPECTED_COLS].copy()
        y = np.asarray(topsellers["shipped_units"], dtype=float)

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
        all_gam_results = topsellers[["product"]].copy().reset_index(drop=True)
        for k, arr in res.get("predictions", {}).items():
            all_gam_results[k] = np.asarray(arr, dtype=float)

        if all(
            c in all_gam_results.columns
            for c in ["units_pred_0.025", "units_pred_0.5", "units_pred_0.975"]
        ):
            all_gam_results["units_pred_avg"] = (
                all_gam_results[
                    ["units_pred_0.025", "units_pred_0.5", "units_pred_0.975"]
                ]
                .to_numpy()
                .mean(axis=1)
            )
        elif "units_pred_0.5" in all_gam_results.columns:
            all_gam_results["units_pred_avg"] = all_gam_results["units_pred_0.5"]
        else:
            all_gam_results["units_pred_avg"] = np.nan

        # Carry identifiers & prices
        for col in ["asin", "price", "asp"]:
            if col in topsellers.columns:
                all_gam_results[col] = topsellers[col].reset_index(drop=True)

        # P50 price used by tables
        if "pred_0.5" not in all_gam_results.columns:
            all_gam_results["pred_0.5"] = (
                all_gam_results["asp"]
                if "asp" in all_gam_results.columns
                else all_gam_results.get("price")
            )

        # Actual daily revenue (price × daily units)
        if "revenue_actual" not in all_gam_results.columns:
            price_col = (
                "asp"
                if "asp" in topsellers.columns
                else ("price" if "price" in topsellers.columns else None)
            )
            if price_col is not None and "shipped_units" in topsellers.columns:
                pr = pd.to_numeric(topsellers[price_col], errors="coerce").reset_index(
                    drop=True
                )
                pu = pd.to_numeric(
                    topsellers["shipped_units"], errors="coerce"
                ).reset_index(drop=True)
                all_gam_results["revenue_actual"] = pr * pu
            else:
                all_gam_results["revenue_actual"] = pd.Series(
                    np.nan, index=all_gam_results.index
                )

        if (
            "daily_rev" not in all_gam_results.columns
            and "revenue_actual" in all_gam_results.columns
        ):
            all_gam_results["daily_rev"] = pd.to_numeric(
                all_gam_results["revenue_actual"], errors="coerce"
            )

        # Revenue predictions from units × price
        price_col = "asp" if "asp" in all_gam_results.columns else "price"
        if price_col in all_gam_results.columns:
            for q in ("0.025", "0.5", "0.975"):
                up, rp = f"units_pred_{q}", f"revenue_pred_{q}"
                if up in all_gam_results.columns and rp not in all_gam_results.columns:
                    all_gam_results[rp] = pd.to_numeric(
                        all_gam_results[up], errors="coerce"
                    ) * pd.to_numeric(all_gam_results[price_col], errors="coerce")

        # Merge elasticity if available
        if not elasticity_df.empty and "product" in elasticity_df.columns:
            keep_cols = ["product"] + [
                c for c in ["ratio", "elasticity_score"] if c in elasticity_df.columns
            ]
            all_gam_results = all_gam_results.merge(
                elasticity_df[keep_cols], on="product", how="left"
            )

        # Debug: sanity medians
        try:
            med_act = float(
                np.nanmedian(all_gam_results.get("daily_rev", pd.Series([np.nan])))
            )
            med_p50 = float(
                np.nanmedian(
                    all_gam_results.get("revenue_pred_0.5", pd.Series([np.nan]))
                )
            )
            print(
                f"[DBG] med(actual_daily_rev)={med_act:.2f}  med(pred50_rev)={med_p50:.2f}",
                flush=True,
            )
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

    def gam_results(self, all_gam_results: pd.DataFrame):
        """
        Single y-axis revenue chart with:
        - Revenue band (P2.5 - P97.5)
        - P50 revenue line
        - Actual revenue points
        - Markers for recommended, conservative and optimistic prices
        """

        # 0) Guards
        if all_gam_results is None or getattr(all_gam_results, "empty", True):
            return self.empty_fig("No model data")

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

        # 1) Figure
        fig = go.Figure()

        # 2) Calculate time-based opacity for actual points
        if 'order_date' in all_gam_results.columns:
            dates = pd.to_datetime(all_gam_results['order_date'])
            # Scale dates to 0-1 range for opacity
            date_nums = (dates - dates.min()) / (dates.max() - dates.min())
            # Map to opacity range 0.15-0.55
            opacities = 0.15 + (0.4 * date_nums)
        else:
            opacities = pd.Series(0.55, index=all_gam_results.index)

        # 2) Per-product plotting, **sorted by ASP**
        for group_name, g in all_gam_results.groupby("product"):
            g = g.dropna(subset=["asp"]).copy()
            if g.empty:
                continue
            g = g.sort_values("asp")

            # Get opacities for this group
            group_opacities = opacities[g.index]

            # Actual revenue points with varying opacity
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_actual"],
                    mode="markers",
                    marker=dict(
                        size=8, 
                        symbol="circle",
                        opacity=group_opacities  # Individual point opacities
                    ),
                    name=f"{group_name} • Actual Revenue",
                    legendgroup=group_name,
                    hovertemplate=(
                        "ASP=%{x:$,.2f}<br>"
                        "Actual Rev=%{y:$,.0f}<br>"
                        "Date=%{customdata}<extra></extra>"
                    ),
                    customdata=g['order_date'] if 'order_date' in g.columns else None
                )
            )

            # Revenue band (P2.5–P97.5), drawn as two traces to avoid self-crossing
            fig.add_trace(
                go.Scatter(
                    name=f"{group_name} (Rev band upper)",
                    x=g["asp"],
                    y=g["revenue_pred_0.975"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=group_name,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name=f"{group_name} (Rev band)",
                    x=g["asp"],
                    y=g["revenue_pred_0.025"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=group_name,
                )
            )

            # Diamonds for recommended / conservative / optimistic prices
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
                        legendgroup=group_name,
                        hovertemplate=f"{label}<br>Price=%{{x:$,.2f}}<br>Rev=%{{y:$,.0f}}<extra></extra>",
                    )
                )

            # P50 line (sorted!)
            fig.add_trace(
                go.Scatter(
                    x=g["asp"],
                    y=g["revenue_pred_0.5"],
                    mode="lines",
                    name=f"{group_name} • Expected Revenue (P50)",
                    line=dict(width=2),
                    legendgroup=group_name,
                    hovertemplate="ASP=%{x:$,.2f}<br>Expected Rev=%{y:$,.0f}}<extra></extra>",
                )
            )

        # 3) Layout (avoid crashing if self.template is not set)
        template = getattr(self, "template", None)
        fig.update_layout(
            template=template if template else None,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            margin=dict(r=8, t=40, b=40, l=60),
        )
        fig.update_yaxes(
            title="Expected Daily Revenue", tickprefix="$", separatethousands=True
        )
        fig.update_xaxes(
            title="Average Selling Price (ASP)", tickprefix="$", separatethousands=True
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
