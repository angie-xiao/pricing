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


class DataEngineer:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n

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
        """
        Grid search to find optimal weight parameters by minimizing RMSE
        between actual and predicted values.

        params:
            sub: DataFrame containing the data
            param_grid: Dictionary of parameters to search. If None, uses default grid

        returns:
            tuple: (best_tail_strength, best_tail_p, best_rmse)
        """
        if param_grid is None:
            param_grid = {
                "tail_strength": np.linspace(
                    0.0, 1.0, 5
                ),  # [0.0, 0.25, 0.5, 0.75, 1.0]
                "tail_p": np.linspace(0.1, 2.0, 5),  # [0.1, 0.575, 1.05, 1.525, 2.0]
            }

        best_rmse = float("inf")
        best_params = None

        # Get actual values
        y_true = sub["shipped_units"].values

        # robustness check
        for tail_strength in param_grid["tail_strength"]:
            for tail_p in param_grid["tail_p"]:
                weights = self._make_weights(sub, tail_strength, tail_p)
                
                # Add robustness check
                if weights.std() > 2.0:  # Skip if weights are too dispersed
                    continue
                    
                if len(weights) > 0:
                    y_pred = np.average(y_true, weights=weights)
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = (tail_strength, tail_p)

        if best_params is None:
            return (0.3, 0.5, None)  # default values if search fails

        return (*best_params, best_rmse)

    def _make_weights(
        self, sub: pd.DataFrame, tail_strength: float = None, tail_p: float = None
    ) -> np.ndarray:
        """
        Create weights based on time decay and price outliers

        params:
            sub: DataFrame containing the data
            tail_strength: controls outlier weighting intensity (0 = no extra weight, 1 = aggressive)
            tail_p: controls outlier weight distribution (1: linear, >1: concave, <1: convex)

        returns:
            numpy array of weights
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
            w *= 1.0 + (tail_strength * dampening) * np.minimum(rel_dist, max_dist) ** (tail_p * 0.8)

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
        df_days_asp["revenue"] = df_days_asp["shipped_units"] * df_days_asp["price"]

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
        elasticity["pct_change_price"] = 100.0 * (
            np.log(np.maximum(elasticity["asp_max"], eps))
            - np.log(np.maximum(elasticity["asp_min"], eps))
        )
        elasticity["pct_change_qty"] = 100.0 * (
            np.log(np.maximum(elasticity["shipped_units_max"], eps))
            - np.log(np.maximum(elasticity["shipped_units_min"], eps))
        )
        elasticity["ratio"] = elasticity["pct_change_qty"] / np.where(
            elasticity["pct_change_price"] == 0, np.nan, elasticity["pct_change_price"]
        )
        elasticity["pct"] = elasticity["ratio"].rank(pct=True) * 100
        return elasticity.sort_values("ratio", ascending=False).reset_index(drop=True)


class GAMTuner:
    def __init__(self, expectile=0.5, lam_grid=None, n_splines_grid=None):
        self.expectile = expectile
        self.lam_grid = np.logspace(-3, 3, 10) if lam_grid is None else np.array(lam_grid)
        self.n_splines_grid = n_splines_grid or [10, 20, 30]

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        scaler_X = StandardScaler()
        Xs = scaler_X.fit_transform(X.reshape(-1, 1).astype(float))
        
        # Single term for price only
        terms = s(0, basis='ps', n_splines=20)
        
        gam = ExpectileGAM(terms, expectile=self.expectile)
        try:
            gam = gam.gridsearch(
                Xs,
                y.astype(float),
                lam=self.lam_grid,
                weights=sample_weight,
            )
        except Exception as e:
            print(f"Gridsearch failed; fallback: {e}")
            gam = ExpectileGAM(terms, expectile=self.expectile).fit(
                Xs, y.astype(float), weights=sample_weight
            )
        
        gam._scaler_X = scaler_X
        return gam

class GAMModeler:
    def __init__(
        self,
        topsellers: pd.DataFrame,
        tail_strength: float = None,
        tail_p: float = None,
        # use_grid_search: bool = False,
        # custom_grid: dict =(
        #     {
        #         "tail_strength": [0.1, 0.2, 0.3, 0.4, 0.5],  # Reduced from [0.1, 0.3, 0.5, 0.7, 0.9]
        #         "tail_p": [0.2, 0.4, 0.6, 0.8, 1.0],  # Reduced from [0.2, 0.6, 1.0, 1.4, 1.8]
        #     }
        # ),
    ):
        """
        Parameters
        ----------
        topsellers : DataFrame
            Source data (Top-N products) with required columns:
            ['product','asin','price','deal_discount_percent',
             'event_encoded','product_encoded','shipped_units','days_sold', ...]
        tail_strength : float, optional
            Outlier weighting intensity (0 = none, 1 = aggressive)
        tail_p : float, optional
            Outlier weight distribution (1 = linear, >1 concave, <1 convex)
        use_grid_search : bool, optional
            Whether to use grid search to find optimal parameters
        custom_grid : dict, optional
            Custom parameter grid for grid search
        """
        self.topsellers = topsellers
        self.tail_strength = tail_strength
        self.tail_p = tail_p
        # self.use_grid_search = use_grid_search
        # self.custom_grid = custom_grid
        self.engineer = DataEngineer(None, None)

    # --------- helpers ---------
    def _required_cols(self) -> list:
        return [
            "price",
            "deal_discount_percent",
            "event_encoded",
            "product_encoded",
            "shipped_units",
        ]

    def _make_design_matrix(self, sub: pd.DataFrame) -> tuple:
        """Prepare subgroup rows: sort, add weights, drop NaNs, build (X, y, w)."""
        if sub is None or sub.empty:
            return None, None, None, None

        # sort by price for consistent curves
        sub = sub.sort_values("price").reset_index(drop=True)

        # add weights (simpler version)
        w = np.ones(len(sub))  # equal weights or modify as needed
        sub = pd.concat([sub, pd.Series(w, name="wt")], axis=1)

        # keep only rows with complete price & target
        sub_clean = sub.dropna(subset=["price", "shipped_units"]).copy()
        if sub_clean.shape[0] < 5:  # minimum points needed
            return None, None, None, None

        # design matrix is just price
        X = sub_clean[["price"]].to_numpy(dtype=float)
        y = sub_clean["shipped_units"].to_numpy(dtype=float)
        w = sub_clean["wt"].to_numpy(dtype=float)

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            return None, None, None, None

        return sub_clean, X, y, w


    def _fit_expectiles(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, qs=(0.025, 0.5, 0.975)
    ) -> dict:
        """
        Fit Expectile GAM for each quantile in qs and return predictions dict
        keyed as {'units_pred_{q}': yhat, 'revenue_pred_{q}': yhat * price}.
        Note: price vector is not available here; caller provides it.
        """
        out = {}
        for q in qs:
            gam = GAMTuner(expectile=q).fit(X, y, sample_weight=w)
            yhat = gam.predict(gam._scaler_X.transform(X))
            out[f"units_pred_{q}"] = yhat
        return out

    def _assemble_group_results(
        self, sub_clean: pd.DataFrame, preds: dict
    ) -> pd.DataFrame:
        """
        Merge subgroup fields + predictions, and derive revenue preds from units * price.
        """
        res_pred = pd.DataFrame(preds, index=sub_clean.index)

        # add revenue predictions from units * price
        for k in list(preds.keys()):
            if k.startswith("units_pred_"):
                q = k.replace("units_pred_", "")
                res_pred[f"revenue_pred_{q}"] = preds[k] * sub_clean["price"].values

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
        """
        Add shared derived columns for downstream UI/KPIs.
        """
        if df is None or df.empty:
            return df

        # normalize keys / aliases
        df["asp"] = df["price"]
        df["asin"] = df["asin"].astype(str)

        # actual revenue
        df["revenue_actual"] = df["shipped_units"] * df["price"]

        # average predictions
        units_cols = [c for c in df.columns if c.startswith("units_pred_")]
        if units_cols:
            df["units_pred_avg"] = df[units_cols].mean(axis=1)
            df["revenue_pred_avg"] = df["units_pred_avg"] * df["price"]
        else:
            df["units_pred_avg"] = np.nan
            df["revenue_pred_avg"] = np.nan

        # daily metrics for fit & coverage
        if "days_sold" in df.columns:
            den = df["days_sold"].replace(0, np.nan)
            df["daily_rev"] = df["revenue_actual"] / den
            df["daily_units"] = df["shipped_units"] / den
        else:
            df["daily_rev"] = df["revenue_actual"]
            df["daily_units"] = df["shipped_units"]

        # alias for helpers that expect 'pred_0.5' (units)
        if "units_pred_0.5" in df.columns and "pred_0.5" not in df.columns:
            df["pred_0.5"] = df["units_pred_0.5"]

        return df

    # --------- main ---------
    def run(self) -> pd.DataFrame:
        """
        Orchestrates the per-product pipeline:
          1) prepare subgroup & design
          2) fit expectile GAMs
          3) assemble subgroup results
          4) concatenate & postprocess for UI/KPIs
        """
        all_results = []

        for product_name, sub in self.topsellers.groupby("product"):
            sub_clean, X, y, w = self._make_design_matrix(sub)
            if sub_clean is None:
                continue

            preds = self._fit_expectiles(X, y, w)
            group_df = self._assemble_group_results(sub_clean, preds)
            all_results.append(group_df)

        if not all_results:
            return pd.DataFrame()

        all_gam_results = pd.concat(all_results, axis=0, ignore_index=True)
        return self._postprocess_all(all_gam_results)


class Optimizer:
    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        return {
            # Finds price that maximizes average predicted units
            "best_avg": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_avg"
            ),
            # Finds price that maximizes median predicted units
            "best50": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_0.5"
            ),
            # Finds price that maximizes optimistic predicted units
            "best975": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_0.975"
            ),
            # Finds price that maximizes conservative predicted units
            "best25": DataEng.pick_best_by_group(
                all_gam_results, "product", "units_pred_0.025"
            ),
            "all_gam_results": all_gam_results,
        }


class PricingPipeline:
    def __init__(
        self,
        pricing_df,
        product_df,
        top_n=10,
        # use_grid_search=True,
        # custom_grid=(
        #     {
        #         "tail_strength": [0.1, 0.2, 0.3, 0.4, 0.5],  # Reduced from [0.1, 0.3, 0.5, 0.7, 0.9]
        #         "tail_p": [0.2, 0.4, 0.6, 0.8, 1.0],  # Reduced from [0.2, 0.6, 1.0, 1.4, 1.8]
        #     }
        # ),
    ):
        self.engineer = DataEngineer(pricing_df, product_df, top_n)
        # self.use_grid_search = use_grid_search
        # self.custom_grid = custom_grid

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
        product = self.engineer.product_df.copy()
        product["product"] = DataEng.compute_product_series(product)
        out = product[["asin", "product", "current_price"]]
        return out.reset_index(drop=True)

    def _build_core_frames(self):
        """Prepare core data, decay, elasticity, and GAM model results."""
        topsellers = self.engineer.prepare()  # Top-N only
        topsellers_decayed = self.engineer._time_decay(topsellers)  # time decay
        elasticity_df = ElasticityAnalyzer.compute(topsellers_decayed)  # UI only

        # Initialize GAMModeler with grid search parameters
        modeler = GAMModeler(
            topsellers,
            # use_grid_search=self.use_grid_search,
            # custom_grid=self.custom_grid,
        )

        # Run the model
        all_gam_results = modeler.run()

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
        """Compute data range and annualization factor using the pipeline’s pricing_df."""
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

    def _build_opportunity_summary(
        self, best50, all_gam_results, curr_price_df
    ):
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
                    "delta_units_annual": (
                        du * 365 if pd.notna(du) else np.nan
                    ),
                    "delta_revenue_annual": (
                        dr * 365 if pd.notna(dr) else np.nan
                    ),
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
        """Final dictionary payload used by the app."""
        return {
            "price_quant_df": (
                topsellers.groupby(["price", "product"])["shipped_units"]
                .sum()
                .reset_index()
            ),
            "best_avg": best_avg,
            "all_gam_results": all_gam_results,
            "best_optimal_pricing_df": best50.copy(),  # consumed by Overview KPIs/callbacks
            "elasticity_df": elasticity_df[["product", "ratio", "pct"]],
            "curr_opt_df": best_avg,
            "curr_price_df": curr_price_df,
            "opps_summary": opps_summary,
            "meta": meta,
        }

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

#     all_gam_results = GAMModeler(
#         DataEngineer(pricing_df, product_df, top_n=10).prepare()).run()

#     # Create a viz instance first, then call the method
#     viz_instance = viz()
#     viz_instance.gam_results(all_gam_results)
