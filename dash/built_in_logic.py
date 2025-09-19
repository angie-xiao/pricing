# + event info

# --------- built_in_logic.py  ---------
# (RMSE-focused; Top-N only; adds annualized opps & data range)
import os, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# viz
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

# ML
from pygam import ExpectileGAM, s
from sklearn.preprocessing import StandardScaler, LabelEncoder

# local import
from helpers import DataEng


def _p(msg: str):
    """Timestamped console print with flush."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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

    def _grid_search_weights(self, sub: pd.DataFrame, param_grid: dict = None) -> tuple:
        if param_grid is None:
            param_grid = {
                "tail_strength": np.linspace(0.0, 1.0, 5),
                "tail_p": np.linspace(0.1, 2.0, 5),
            }

        best_rmse = float("inf")
        best_params = None

        if getattr(self, "_verbose", False):
            _p(
                f"[tuning] weight grid-search... combos={len(param_grid['tail_strength'])*len(param_grid['tail_p'])}"
            )

        y_true = sub["shipped_units"].values
        for tail_strength in param_grid["tail_strength"]:
            for tail_p in param_grid["tail_p"]:
                weights = self._make_weights(sub, tail_strength, tail_p)

                # Skip if weights are too dispersed (no log spam)
                if weights.std() > 2.0:
                    continue

                if len(weights) > 0:
                    y_pred = np.average(y_true, weights=weights)
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = (tail_strength, tail_p)

        if best_params is None:
            if getattr(self, "_verbose", False):
                _p("[tuning] grid-search failed; using defaults (0.3, 0.5)")
            return (0.3, 0.5, None)

        if getattr(self, "_verbose", False):
            ts, tp = best_params
            _p(
                f"[tuning] done | best tail_strength={ts:.3f}, tail_p={tp:.3f}, rmse={best_rmse:.2f}"
            )

        return (*best_params, best_rmse)

    def _make_weights(
        self,
        sub: pd.DataFrame,
        tail_strength: float = None,
        tail_p: float = None,
        elasticity_df: pd.DataFrame = None,
    ) -> np.ndarray:
        """
        Create weights based on time decay, price outliers, and elasticity
        """
        # If parameters not provided, do grid search
        if tail_strength is None or tail_p is None:
            tail_strength, tail_p, _ = self._grid_search_weights(sub)

        # Time decay weights
        decayed_df = self._time_decay(sub)
        w = decayed_df["time_decay_weight"].values

        # Modify the outlier adjustment formula
        if tail_strength > 0:
            asp = sub["price"].values.astype(float)
            q25, q75 = np.percentile(asp, [25, 75])
            iqr = q75 - q25 if q75 > q25 else 1.0

            # Add a cap to relative distance calculation
            rel_dist = np.abs(asp - np.median(asp)) / iqr
            max_dist = 1.5  # Reduce from 2.0 to 1.5

            # Add dampening factor and reduce power
            dampening = 0.7
            w *= 1.0 + (tail_strength * dampening) * np.minimum(rel_dist, max_dist) ** (
                tail_p * 0.8
            )

        # Incorporate elasticity if available
        if elasticity_df is not None:
            product = sub["product"].iloc[0]
            elasticity_score = (
                elasticity_df.loc[
                    elasticity_df["product"] == product, "elasticity_score"
                ].iloc[0]
                if product in elasticity_df["product"].values
                else 50
            )

            # Scale factor based on elasticity (0.8 to 1.2)
            elasticity_factor = 0.8 + (0.4 * elasticity_score / 100)
            w *= elasticity_factor

        return w / np.mean(w) if w.size > 0 else w

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

    def prepare(self) -> pd.DataFrame:
        """ """
        # normalize columns
        self.pricing_df = DataEng.clean_cols(self.pricing_df)
        self.product_df = DataEng.clean_cols(self.product_df)

        # merge
        df = self.pricing_df.merge(self.product_df, how="left", on="asin")

        # product label
        df["product"] = DataEng.compute_product_series(df)
        # keep if they exist; ignore otherwise
        for c in ("tag", "variation"):
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        # data type
        df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")

        # make sure nothing is neg
        if "shipped_units" in df.columns:
            df["shipped_units"] = self._nonneg(df["shipped_units"])
        if "price" in df.columns:
            df["price"] = self._nonneg(df["price"])

        # ---- GLOBAL OVERRIDE: synthesize order_date over a chosen window ----
        # If you truly want to override even partial data, just always run this line:
        # df = self._synthesize_order_dates(
        #     df,
        #     start="2023-09-17",
        #     end="2025-09-17",
        #     seed=42,
        #     col="order_date",
        # )
        # --------------------------------------------------------------------

        # persist back so meta/date-range uses the same synthetic dates
        self.pricing_df = df.copy()

        # count how many days a price was sold at
        df_days_asp = self._days_at_price(df)

        # manually calculate rev
        df_days_asp["revenue"] = (
            df_days_asp["shipped_units"] * df_days_asp["price"]
        ).clip(lower=0)

        # ----------- reog this step later -----------
        # group by asin - event - date
        df_agg = (
            df_days_asp[
                ["asin", "event_name", "order_date", "shipped_units", "revenue"]
            ]
            .groupby(["asin", "event_name", "order_date"])[["revenue", "shipped_units"]]
            .sum()
            .reset_index()
        )
        df_agg["price"] = df_agg["revenue"] / df_agg["shipped_units"]
        df_agg["price"] = round(df_agg["price"], 2)

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

        # calculate daily shipped units &
        df_agg_event["daily_rev"] = df_agg_event["revenue"] / df_agg_event["days_sold"]
        df_agg_event["daily_units"] = (
            df_agg_event["shipped_units"] / df_agg_event["days_sold"]
        )
        df_agg_event.drop(columns=["revenue", "shipped_units"], inplace=True)
        df_agg_event.rename(
            columns={"daily_rev": "revenue", "daily_units": "shipped_units"},
            inplace=True,
        )
        # -------------------------------------------------------

        # Top-N products by total revenue
        top_n_products = (
            df_agg_event.groupby("product")["revenue"]
            .sum()
            .reset_index()
            .sort_values("revenue", ascending=False)["product"]
            .head(self.top_n)
            .tolist()
        )

        # filtering
        filtered = df_days_asp[(df_days_asp["product"].isin(top_n_products))].copy()

        # Normalize dtype for downstream joins
        filtered["asin"] = filtered["asin"].astype(str)
        filtered.rename(columns={"revenue": "revenue_share_amt"}, inplace=True)
        filtered["revenue_share_amt"] = self._nonneg(filtered["revenue_share_amt"])

        # encode categorical vars after filtering
        res = self._label_encoder(filtered)

        return res


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


class GAMTuner:

    def __init__(self, expectile=0.5, lam_grid=None, n_splines_grid=None):
        """
        params
            expectile
            lam_grid
            n_splines_grid
        """
        self.expectile = expectile
        # Expanded lambda grid for better smoothing control
        self.lam_grid = (
            np.logspace(-5, 5, 20) if lam_grid is None else np.array(lam_grid)
        )
        # More flexible spline options
        self.n_splines_grid = n_splines_grid or [20, 30, 40]

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        scaler_X = StandardScaler()
        Xs = scaler_X.fit_transform(X.reshape(-1, 1).astype(float))
        best_gam = None
        best_score = float("inf")
        best_ns = None

        if getattr(self, "_verbose", False):
            lam_len = (
                len(self.lam_grid)
                if isinstance(self.lam_grid, (list, np.ndarray))
                else "?"
            )
            _p(
                f"[fit] expectile={self.expectile} | n={len(Xs)} | splines={self.n_splines_grid} | lam={lam_len}"
            )

        yf = y.astype(float)

        # normalized weights for scoring (gridsearch still uses original weights)
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            sw[~np.isfinite(sw)] = 0.0
            sw_sum = sw.sum()
            sw = (sw / sw_sum) if sw_sum > 0 else None

        for n_splines in self.n_splines_grid:
            terms = s(0, basis="ps", n_splines=n_splines)
            gam = ExpectileGAM(terms, expectile=self.expectile, fit_intercept=True)
            try:
                gam = gam.gridsearch(
                    Xs, yf, lam=self.lam_grid, weights=sample_weight, progress=False
                )
                y_pred = gam.predict(Xs)

                # Weighted RMSE (no deviance call)
                resid2 = (yf - y_pred) ** 2
                mse = (
                    float((resid2 * sw).sum())
                    if sw is not None
                    else float(resid2.mean())
                )
                rmse = np.sqrt(mse)

                # penalize overly wiggly funcs to prevent extrapolating
                denom = max(np.mean(np.abs(y_pred)), 1e-9)
                smoothness_penalty = float(np.mean(np.abs(np.diff(y_pred, 2))) / denom)
                score = rmse * (1 + 0.2 * smoothness_penalty)

                if score < best_score:
                    best_score = score
                    best_gam = gam
                    best_ns = n_splines
            except Exception:
                # swallow per-spline errors silently to avoid noise
                continue

        if best_gam is None:
            terms = s(0, basis="ps", n_splines=25)
            best_gam = ExpectileGAM(terms, expectile=self.expectile).fit(
                Xs, yf, weights=sample_weight
            )
            best_ns = 25

        if getattr(self, "_verbose", False):
            _p(f"[fit] done | best_n_splines={best_ns} | score={best_score:.4f}")

        best_gam._scaler_X = scaler_X
        return best_gam


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
        # NEW
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

        if self.verbose:
            _p(
                "[init] GAMModeler ready | "
                f"rows={len(self.topsellers)} | gridsearch={self.use_grid_search} | "
                f"test_size={self.test_size} | bootstraps={self.n_bootstrap} | "
                f"bootstrap_frac={self.bootstrap_frac}"
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

    def _make_design_matrix(self, sub: pd.DataFrame):
        if sub is None or sub.empty:
            return None, None, None, None, None

        # Essentials only
        if self.verbose:
            p = (
                str(sub["product"].iloc[0])
                if "product" in sub.columns and len(sub)
                else "?"
            )
            _p(f"[model] {p}: rows={len(sub)}, unique_prices={sub['price'].nunique()}")

        sub = sub.sort_values("price").reset_index(drop=True)

        # Build weights (grid search message is already compact inside DataEngineer)
        if self.use_grid_search:
            best_tail_strength, best_tail_p, _ = self.engineer._grid_search_weights(
                sub, param_grid=self.custom_grid
            )
            w = self.engineer._make_weights(
                sub, tail_strength=best_tail_strength, tail_p=best_tail_p
            )
        else:
            de = DataEngineer(None, None)
            sub_decayed = de._time_decay(sub)
            time_decay_weight = sub_decayed["time_decay_weight"]
            volume = sub["shipped_units"]
            volume_weight = np.log1p(volume) / np.log1p(volume.max())
            price_range = sub["price"].max() - sub["price"].min()
            if price_range > 0:
                price_counts = sub["price"].value_counts()
                price_density = sub["price"].map(price_counts) / len(sub)
                price_weight = 1 / (price_density + 0.1)
                price_weight = price_weight / price_weight.max()
            else:
                price_weight = pd.Series(1.0, index=sub.index)
            w = time_decay_weight * 0.4 + volume_weight * 0.4 + price_weight * 0.2
            w = w / w.mean()

        sub = pd.concat([sub, pd.Series(w, name="wt", index=sub.index)], axis=1)
        sub_clean = sub.dropna(subset=["price", "shipped_units"]).copy()
        if sub_clean.shape[0] < 5 or sub_clean["price"].nunique() < 3:
            if self.verbose:
                _p("[split] insufficient data; skip")
            return None, None, None, None, None

        # Time-aware split (or random fallback)
        is_train = pd.Series(True, index=sub_clean.index)
        used_time_split = False
        cutoff = None
        if self.split_by_date and "order_date" in sub_clean.columns:
            dates = pd.to_datetime(sub_clean["order_date"], errors="coerce")
            if dates.notna().sum() >= 5:
                cutoff = dates.quantile(1 - self.test_size)
                is_train = dates <= cutoff
                used_time_split = True

        if is_train.sum() < 3 or (~is_train).sum() < 1:
            seed = abs(hash(str(sub_clean["product"].iloc[0]))) % (2**32 - 1)
            rng = np.random.default_rng(self.random_state or seed)
            is_train = pd.Series(
                rng.random(len(sub_clean)) >= self.test_size, index=sub_clean.index
            )
            if is_train.sum() < 3:
                is_train[:] = True

        sub_clean["split"] = np.where(is_train, "train", "test")

        if self.verbose:
            method = "time" if used_time_split else "random"
            _p(
                f"[split] {method} | train={int(is_train.sum())}, test={int((~is_train).sum())}"
            )

        X_all = sub_clean[["price"]].to_numpy(dtype=float)
        y_all = sub_clean["shipped_units"].to_numpy(dtype=float)
        w_all = sub_clean["wt"].to_numpy(dtype=float)
        return (
            sub_clean,
            X_all[is_train.values],
            y_all[is_train.values],
            w_all[is_train.values],
            X_all,
        )

    def _fit_expectiles(
        self, X_train, y_train, w_train, X_pred, qs=(0.025, 0.5, 0.975)
    ) -> dict:
        out = {}
        rng = np.random.default_rng(self.random_state)

        def fit_once(q, idx):
            tuner = GAMTuner(expectile=q)
            tuner._verbose = self.verbose
            gam = tuner.fit(
                X_train[idx],
                y_train[idx],
                sample_weight=(None if w_train is None else w_train[idx]),
            )
            yhat = gam.predict(gam._scaler_X.transform(X_pred))
            return np.maximum(yhat, 0.0)

        n_train = len(X_train)
        for q in qs:
            loops = self._bootstrap_loops_for_size(n_train)
            if self.verbose:
                mode = "bootstrap" if loops > 0 else "single-fit"
                _p(f"[fit] q={q} | {mode}" + (f" x{loops}" if loops else ""))

            if loops <= 0:
                out[f"units_pred_{q}"] = fit_once(q, np.arange(n_train))
                continue

            idx_all = np.arange(n_train)
            m = max(1, int(n_train * self.bootstrap_frac))
            preds = []
            target = self.bootstrap_target_rel_se

            for i in range(loops):
                sample_idx = rng.choice(idx_all, size=m, replace=True)
                preds.append(fit_once(q, sample_idx))

                # early stop after a few iterations if relative SE is small
                if target and (i + 1) >= 5:
                    P = np.vstack(preds)  # [i+1, n_pred]
                    mean_vec = P.mean(axis=0)
                    se_vec = P.std(axis=0, ddof=1) / np.sqrt(i + 1)
                    rel_se = np.nanmean(se_vec / np.maximum(np.abs(mean_vec), 1e-8))
                    if rel_se < target:
                        if self.verbose:
                            _p(
                                f"[bootstrap] q={q} early stop {i+1}/{loops} (rel_se≈{rel_se:.3f})"
                            )
                        break

            P = np.vstack(preds)
            out[f"units_pred_{q}"] = P.mean(axis=0)
            out[f"units_pred_{q}_sd"] = P.std(axis=0, ddof=1)
        return out

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

    # --------- main ---------
    def run(self) -> pd.DataFrame:
        all_results = []
        for product_name, sub in self.topsellers.groupby("product"):
            if self.verbose:
                _p(f"[start] {product_name}")
            sub_clean, X_tr, y_tr, w_tr, X_pred = self._make_design_matrix(sub)
            if sub_clean is None:
                if self.verbose:
                    _p(f"[skip] {product_name}")
                continue
            preds = self._fit_expectiles(X_tr, y_tr, w_tr, X_pred)
            group_df = self._assemble_group_results(sub_clean, preds)
            all_results.append(group_df)
            if self.verbose:
                _p(f"[done] {product_name} | rows_out={len(group_df)}")

        if not all_results:
            if self.verbose:
                _p("[run] no results")
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


# --------------------------- viz ---------------------------
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


# if __name__ == "__main__":
#     pricing_df, product_df = pd.read_csv('data/pricing.csv'), pd.read_csv('data/products.csv')

#     PricingPipeline(pricing_df,product_df,top_n=10, use_grid_search=True).assemble_dashboard_frames()


# all_gam_results = GAMModeler(
#     DataEngineer(pricing_df, product_df, top_n=10).prepare()).run()
# # Create a viz instance first, then call the method
# viz_instance = viz()
# viz_instance.gam_results(all_gam_results)
