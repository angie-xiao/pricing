# --------- built_in_logic.py (RMSE-focused; no weekly agg; no y-normalization) ---------
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygam import ExpectileGAM, s
from sklearn.metrics import mean_squared_error
from functools import reduce
import operator as op
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

# --------------------------- Data engineering ---------------------------
class DataEngineer:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n

    def prepare(self) -> pd.DataFrame:
        def _clean_cols(df):
            df = df.copy()
            df.columns = df.columns.str.strip().str.lower()
            return df

        self.pricing_df = _clean_cols(self.pricing_df)
        self.product_df = _clean_cols(self.product_df)

        candidate_keys = ["asin", "sku", "item_sku", "product_id", "parent_asin"]
        common_keys = [k for k in candidate_keys if k in self.pricing_df and k in self.product_df]
        if not common_keys:
            raise KeyError(
                f"No common join key. Tried {candidate_keys}. "
                f"pricing_df cols={list(self.pricing_df.columns)}, "
                f"product_df cols={list(self.product_df.columns)}"
            )
        join_key = common_keys[0]
        self.pricing_df[join_key] = self.pricing_df[join_key].astype(str).str.strip()
        self.product_df[join_key] = self.product_df[join_key].astype(str).str.strip()

        df = self.pricing_df.merge(self.product_df, how="left", on=join_key, validate="m:1")

        # stable identifiers / labels
        if "product_key" not in df.columns:
            df["product_key"] = df[join_key].astype(str)
        for req in ["tag", "weight", "order_date", "shipped_units", "revenue_share_amt", "event_name"]:
            if req not in df:
                raise KeyError(f"Missing required column: {req}")
        df["product"] = (df["tag"] + " " + df["weight"].astype(str)).str.upper()
        df["order_date"] = pd.to_datetime(df["order_date"])

        # ---- daily aggregation per product_key, then derive ASP ----
        daily = (
            df.groupby(["product_key", "product", "event_name", "order_date"], as_index=False)[
                ["shipped_units", "revenue_share_amt"]
            ]
            .sum()
        )
        daily = daily[daily["shipped_units"] > 0]
        daily["asp"] = daily["revenue_share_amt"] / daily["shipped_units"]
        daily = daily[daily["asp"] > 0]
        daily["asp"] = daily["asp"].round(2)

        # ---- product × ASP table with total units, revenue, and days_sold ----
        asp_product_df = (
            daily.groupby(["product_key", "product", "event_name", "asp"])
            .agg(
                shipped_units=("shipped_units", "sum"),
                revenue_share_amt=("revenue_share_amt", "sum"),
                days_sold=("order_date", "nunique"),
            )
            .reset_index()
        )

        # Top N products by total revenue (keep all rows for chosen products, no low-volume filter)
        top_n_products = (
            asp_product_df.groupby("product")["revenue_share_amt"]
            .sum()
            .reset_index()
            .sort_values("revenue_share_amt", ascending=False)["product"]
            .head(self.top_n)
            .tolist()
        )

        return asp_product_df[
            (asp_product_df["product"].isin(top_n_products)) &
            (asp_product_df["event_name"] == "NO DEAL")
        ]


# --------------------------- Elasticity (summary only for UI) ---------------------------
class ElasticityAnalyzer:
    """
    Product-level summary using log-differences (≈ % changes) with tiny epsilon for safety.
    (Used for KPI/UX only; NOT used in modeling or selection.)
    """
    @staticmethod
    def compute(topsellers: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-9
        elasticity = (
            topsellers.groupby("product")
            .agg(
                asp_max=("asp", "max"),
                asp_min=("asp", "min"),
                shipped_units_max=("shipped_units", "max"),
                shipped_units_min=("shipped_units", "min"),
                product_count=("product", "count"),
            )
            .reset_index()
        )
        elasticity["pct_change_price"] = 100.0 * (
            np.log(np.maximum(elasticity["asp_max"], eps)) - np.log(np.maximum(elasticity["asp_min"], eps))
        )
        elasticity["pct_change_qty"] = 100.0 * (
            np.log(np.maximum(elasticity["shipped_units_max"], eps)) - np.log(np.maximum(elasticity["shipped_units_min"], eps))
        )
        elasticity["ratio"] = elasticity["pct_change_qty"] / np.where(
            elasticity["pct_change_price"] == 0, np.nan, elasticity["pct_change_price"]
        )
        elasticity["pct"] = elasticity["ratio"].rank(pct=True) * 100
        return elasticity.sort_values("ratio", ascending=False).reset_index(drop=True)


# --------------------------- GAM (auto-gridsearch; raw y) ---------------------------

class GAMTuner:
    """
    Two-term ExpectileGAM:
      y ~ s(price, order=3) + s(days_sold, order=2)
    - Standardize both features; keep y in unit space.
    - Auto gridsearch on lam & n_splines (orders fixed per term).
    """
    def __init__(self, expectile=0.5, lam_grid=None, n_splines_grid=None):
        self.expectile = expectile
        self.lam_grid = np.array(lam_grid) if lam_grid is not None else np.logspace(-4, 3, 8)
        self.n_splines_grid = n_splines_grid or [5, 10, 20, 30]

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        # standardize both features (price, days_sold)
        scaler_X = StandardScaler()
        Xs = scaler_X.fit_transform(X.astype(float))

        terms = s(0, spline_order=3) + s(1, spline_order=2)
        gam = ExpectileGAM(terms, expectile=self.expectile)
        try:
            gam = gam.gridsearch(
                Xs, y.astype(float),
                lam=self.lam_grid,
                n_splines=self.n_splines_grid,
                weights=sample_weight
            )
        except Exception as e:
            print(f"Gridsearch failed, fallback: {e}")
            gam = ExpectileGAM(s(0, spline_order=3) + s(1, spline_order=2),
                               expectile=self.expectile).fit(Xs, y.astype(float), weights=sample_weight)

        # cache scaler (no y transform)
        gam._scaler_X = scaler_X
        return gam


# --------------------------- Modeling (weights = reaction-per-day × tail boost) ---------------------------
class GAMModeler:
    def __init__(self, topsellers: pd.DataFrame,
                 gamma_time: float = 0.7,      # inverse-exponential strength for days_sold
                 tail_strength: float = 0.6,   # price tail boost factor
                 tail_p: float = 1.0):         # |z_price|^p
        self.topsellers = topsellers
        self.gamma_time = float(gamma_time)
        self.tail_strength = float(tail_strength)
        self.tail_p = float(tail_p)

    def _make_weights(self, sub: pd.DataFrame) -> np.ndarray:
        units = np.clip(sub["shipped_units"].values.astype(float), 1.0, None)
        # base signal weight: sqrt(units)
        w = np.sqrt(units)

        # inverse-exponential time weight: fewer days_sold -> larger weight
        days = np.clip(sub["days_sold"].values.astype(float), 1.0, None)
        d_med = np.median(days) if np.median(days) > 0 else 1.0
        w_time = np.exp(-self.gamma_time * (days / d_med))
        w *= w_time

        # price tail boost
        asp = sub["asp"].values.astype(float)
        mu, sd = float(np.mean(asp)), float(np.std(asp)) if np.std(asp) != 0 else 1.0
        z = (asp - mu) / sd
        if self.tail_strength > 0:
            w *= (1.0 + self.tail_strength * np.power(np.abs(z), self.tail_p))

        return w

    def run(self) -> pd.DataFrame:
        all_results = []
        for product, sub in self.topsellers.groupby("product"):
            sub = sub.sort_values("asp").reset_index(drop=True)
            # 2D features: [price, days_sold]
            X = sub[["asp", "days_sold"]].to_numpy(dtype=float)
            y = sub["shipped_units"].to_numpy(dtype=float)
            weights = self._make_weights(sub)

            out = {}
            # fit expectiles with same X & weights
            for q in [0.025, 0.5, 0.975]:
                gam = GAMTuner(expectile=q).fit(X, y, sample_weight=weights)
                y_pred = gam.predict(gam._scaler_X.transform(X))  # y in unit space
                out[f"pred_{q}"] = y_pred

            preds = pd.DataFrame(out, index=sub.index)

            # optional single-parameter level calibration to align P50
            try:
                p50 = preds["pred_0.5"].values
                denom = float(np.sum(p50 * p50))
                alpha = (float(np.sum(y * p50)) / denom) if denom > 0 else 1.0
                for k in list(out.keys()):
                    preds[k] = preds[k] * alpha
            except Exception:
                pass

            # Diagnostics-only elasticity (kept if you show it in UI)
            # preds["elasticity"] = _empirical_elasticity(sub["asp"].values, preds["pred_0.5"].values)

            results = pd.concat(
                [sub[["asp", "days_sold", "product", "product_key", "shipped_units"]].reset_index(drop=True),
                 preds.reset_index(drop=True)],
                axis=1
            )
            all_results.append(results)

        all_gam_results = pd.concat(all_results, axis=0, ignore_index=True)

        # revenue transforms
        for col in [c for c in all_gam_results.columns if c.startswith("pred_")]:
            all_gam_results[f"revenue_{col}"] = all_gam_results["asp"] * all_gam_results[col]
        rev_cols = [c for c in all_gam_results.columns if c.startswith("revenue_pred_")]
        all_gam_results["revenue_pred_avg"] = all_gam_results[rev_cols].mean(axis=1)
        all_gam_results["revenue_actual"] = all_gam_results["asp"] * all_gam_results["shipped_units"]
        return all_gam_results


# --------------------------- Optimizer (plain) ---------------------------
class Optimizer:
    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        def pick_best(col):
            if col not in all_gam_results or all_gam_results[col].isna().all():
                return pd.DataFrame()
            return all_gam_results.loc[all_gam_results.groupby("product")[col].idxmax()]
        return {
            "best_avg": pick_best("revenue_pred_avg"),
            "best50":   pick_best("revenue_pred_0.5"),
            "best975":  pick_best("revenue_pred_0.975"),
            "best25":   pick_best("revenue_pred_0.025"),
            "all_gam_results": all_gam_results,
        }


# --------------------------- Pipeline ---------------------------
class PricingPipeline:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.engineer = DataEngineer(pricing_df, product_df, top_n)

    def _build_curr_price_df(self) -> pd.DataFrame:
        """Compute current price as latest (by order_date) ASP per product_key."""
        pricing = self.engineer.pricing_df.copy()
        product = self.engineer.product_df.copy()

        # normalize columns
        pricing.columns = pricing.columns.str.strip().str.lower()
        product.columns = product.columns.str.strip().str.lower()

        # detect join key
        candidate_keys = ["asin", "sku", "item_sku", "product_id", "parent_asin"]
        common_keys = [k for k in candidate_keys if k in pricing and k in product]
        if not common_keys:
            # always return a well-formed empty frame
            return pd.DataFrame(columns=["product_key", "product", "current_price"])

        join_key = common_keys[0]
        pricing[join_key] = pricing[join_key].astype(str).str.strip()
        product[join_key] = product[join_key].astype(str).str.strip()

        df = pricing.merge(product, how="left", on=join_key, validate="m:1")

        required = ["order_date", "shipped_units", "revenue_share_amt"]
        if any(c not in df for c in required):
            return pd.DataFrame(columns=["product_key", "product", "current_price"])

        # ensure product_key & product label
        if "product_key" not in df.columns:
            df["product_key"] = df[join_key].astype(str)
        if "tag" in df.columns and "weight" in df.columns:
            df["product"] = (df["tag"] + " " + df["weight"].astype(str)).str.upper()
        else:
            df["product"] = df["product_key"]

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        grp = (df.groupby(["product_key", "product", "order_date"], as_index=False)
                [["shipped_units", "revenue_share_amt"]].sum())
        grp = grp[grp["shipped_units"] > 0]
        if grp.empty:
            return pd.DataFrame(columns=["product_key", "product", "current_price"])

        grp["asp"] = grp["revenue_share_amt"] / grp["shipped_units"]

        grp = grp.sort_values(["product_key", "order_date"])
        latest = grp.drop_duplicates("product_key", keep="last")

        curr_price_df = latest[["product_key", "product", "asp"]].rename(
            columns={"asp": "current_price"}
        )
        # consistent dtype for downstream joins
        curr_price_df["product_key"] = curr_price_df["product_key"].astype(str)
        return curr_price_df.reset_index(drop=True)


    def assemble_dashboard_frames(self) -> dict:
        topsellers = self.engineer.prepare()                      # has product & product_key
        elasticity_df = ElasticityAnalyzer.compute(topsellers)    # UI only
        all_gam_results = GAMModeler(topsellers, tail_strength=0.6, tail_p=1.0).run()

        # --- Optimizer outputs ---
        bests = Optimizer.run(all_gam_results)
        best_avg = bests["best_avg"].copy()

        # --- Ensure product_key exists on best_avg (fallback merge if missing) ---
        if "product_key" not in best_avg.columns:
            pk_map = topsellers[["product", "product_key"]].drop_duplicates()
            best_avg = best_avg.merge(pk_map, on="product", how="left")

        # --- Ensure revenue_actual exists on best_avg (recompute/merge if missing) ---
        if "revenue_actual" not in best_avg.columns:
            if {"asp", "shipped_units"}.issubset(best_avg.columns):
                best_avg["revenue_actual"] = best_avg["asp"] * best_avg["shipped_units"]
            else:
                ra = (all_gam_results[["product", "asp", "revenue_actual"]]
                        .drop_duplicates())
                best_avg = best_avg.merge(ra, on=["product", "asp"], how="left")

        # --- Current prices from raw data ---
        curr_price_df = self._build_curr_price_df()

        # --- Normalize key dtype across frames (prevents silent empty slices) ---
        for df in (best_avg, all_gam_results, curr_price_df, topsellers):
            if "product_key" in df.columns:
                df["product_key"] = df["product_key"].astype(str)

        # --- Build frames dict ---
        frames = {
            "price_quant_df": (topsellers.groupby(["asp", "product"])["shipped_units"]
                               .sum().reset_index()),
            "best_avg": best_avg,
            "all_gam_results": all_gam_results,
            "best_optimal_pricing_df": best_avg[
                ["product", "product_key", "asp", "revenue_pred_avg", "revenue_actual"]
            ].copy(),
            "elasticity_df": elasticity_df[["product", "ratio", "pct"]],
            "curr_opt_df": best_avg,
            "curr_price_df": curr_price_df,
        }
        return frames

    @classmethod
    def from_csv_folder(cls, base_dir, data_folder="data",
                        pricing_file="730d.csv", product_file="products.csv", top_n=10):
        pricing_df = pd.read_csv(os.path.join(base_dir, data_folder, pricing_file))
        product_df = pd.read_csv(os.path.join(base_dir, data_folder, product_file))
        return cls(pricing_df, product_df, top_n).assemble_dashboard_frames()


# --------------------------- viz ---------------------------
class viz:
    def __init__(self, template="lux"):
        templates = [
            "bootstrap", "minty", "pulse", "flatly", "quartz",
            "cyborg", "darkly", "vapor", "lux",
        ]
        load_figure_template(templates)
        self.template = template

    def gam_results(self, all_gam_results: pd.DataFrame):
        product_lst = all_gam_results["product"].unique()
        pltly_qual = px.colors.qualitative.Dark24
        pltly_qual.extend(px.colors.qualitative.Vivid)
        colors = random.sample(pltly_qual, len(product_lst))
        color_dct = {p: colors[i] for i, p in enumerate(product_lst)}

        fig = go.Figure()
        for group_name, group_df in all_gam_results.groupby("product"):
            best_50 = group_df[group_df["revenue_pred_0.5"] == group_df["revenue_pred_0.5"].max()].reset_index(drop=True)
            best_025 = group_df[group_df["revenue_pred_0.025"] == group_df["revenue_pred_0.025"].max()].reset_index(drop=True)
            best_975 = group_df[group_df["revenue_pred_0.975"] == group_df["revenue_pred_0.975"].max()].reset_index(drop=True)

            fig.add_trace(
                go.Scatter(
                    name=f"group {group_name} band",
                    x=group_df["asp"].tolist() + group_df["asp"].tolist()[::-1],
                    y=group_df["revenue_pred_0.975"].tolist() + group_df["revenue_pred_0.025"].tolist()[::-1],
                    fill="toself",
                    fillcolor="#cbcbcb",
                    line=dict(color="#cbcbcb"),
                    legendgroup=group_name,
                    showlegend=False,
                    opacity=0.4,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group_df["asp"],
                    y=group_df["revenue_actual"],
                    mode="markers",
                    name="Revenue Actual",
                    marker=dict(symbol="x", color=color_dct[group_name], size=10),
                    opacity=0.5,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=best_50["asp"],
                    y=best_50["revenue_pred_0.5"],
                    mode="markers",
                    marker=dict(color="#B82132", size=18),
                    name="Rec. Price",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group_df["asp"],
                    y=group_df["revenue_pred_0.5"],
                    mode="lines",
                    marker=dict(color="#B82132"),
                    name="Expected Revenue Prediction",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=best_025["asp"],
                    y=best_025["revenue_pred_0.025"],
                    mode="markers",
                    marker=dict(color="#AAB396", size=18),
                    name="Conservative Prediction",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=best_975["asp"],
                    y=best_975["revenue_pred_0.975"],
                    mode="markers",
                    marker=dict(color="#F2B28C", size=18),
                    name="Optimistic Prediction",
                )
            )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            margin=dict(r=8),
        )
        return fig

    def elast_dist(self, elast_df: pd.DataFrame):
        fig = (
            px.histogram(elast_df, x="ratio", width=1200, height=600, template=self.template)
            .update_xaxes(title_text="Elasticity")
            .update_yaxes(title_text="Product Count")
        )
        return fig

    def opportunity_chart(elast_df, best50_df, curr_df, all_gam):
        import plotly.express as px
        for df in [elast_df, best50_df, curr_df, all_gam]:
            if df is None or getattr(df, "empty", True):
                return {}
        if ("revenue_pred_0.5" not in all_gam.columns) or ("revenue_pred_0.5" not in best50_df.columns):
            return {}
        prods = sorted(set(all_gam["product"]) & set(best50_df["product"]) & set(curr_df["product"]))
        if not prods:
            return {}
        rows = []
        for p in prods:
            try:
                curr_price = curr_df.loc[curr_df["product"] == p, "current_price"]
                if curr_price.empty or pd.isna(curr_price.iloc[0]):
                    continue
                curr_price = float(curr_price.iloc[0])
                prod = all_gam[
                    (all_gam["product"] == p)
                    & pd.notna(all_gam["asp"])
                    & pd.notna(all_gam["revenue_pred_0.5"])
                ]
                if prod.empty:
                    continue
                idx = (prod["asp"] - curr_price).abs().idxmin()
                rev_curr = float(prod.loc[idx, "revenue_pred_0.5"])
                rec = best50_df.loc[best50_df["product"] == p]
                if rec.empty:
                    continue
                rev_best = float(rec["revenue_pred_0.5"].iloc[0])
                upside = rev_best - rev_curr
                e = elast_df.loc[elast_df["product"] == p, "ratio"]
                elast_val = float(e.iloc[0]) if len(e) else np.nan
                rows.append({"product": p, "upside": upside, "elasticity": elast_val})
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        df = df.sort_values("upside", ascending=False).head(12)
        fig = px.bar(
            df,
            x="product",
            y="upside",
            hover_data=["elasticity"],
            height=380,
        )
        fig.update_yaxes(
            title_text="Upside (Expected Revenue Δ)",
            tickprefix="$",
            separatethousands=True,
        )
        fig.update_traces(
            text=df["upside"].map(lambda x: f"${x:,.0f}"),
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=60),
            uniformtext_minsize=10,
            uniformtext_mode="hide",
            yaxis={"categoryorder": "total descending"},
        )
        return fig

    def empty_fig(self, title="No data"):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template=self.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=60, b=40),
        )
        fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        return fig
