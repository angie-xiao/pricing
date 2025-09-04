# --------- find avg rev, not median
# add RMSE/eval score

# built_in_logic.py
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

        # auto-detect a join key
        candidate_keys = ["asin", "sku", "item_sku", "product_id", "parent_asin"]
        common_keys = [
            k for k in candidate_keys if k in self.pricing_df and k in self.product_df
        ]
        if not common_keys:
            raise KeyError(
                f"No common join key. Tried {candidate_keys}. "
                f"pricing_df cols={list(self.pricing_df.columns)}, "
                f"product_df cols={list(self.product_df.columns)}"
            )
        join_key = common_keys[0]
        self.pricing_df[join_key] = self.pricing_df[join_key].astype(str).str.strip()
        self.product_df[join_key] = self.product_df[join_key].astype(str).str.strip()

        df_tags = self.pricing_df.merge(
            self.product_df, how="left", on=join_key, validate="m:1"
        )

        # ensure we always have a product_key
        if "product_key" not in df_tags.columns:
            # fallback: use the join key as a stable identifier
            df_tags["product_key"] = df_tags[join_key].astype(str)

        # required for product label
        missing = [
            c
            for c in [
                "tag",
                "weight",
                "order_date",
                "shipped_units",
                "revenue_share_amt",
            ]
            if c not in df_tags
        ]
        if missing:
            raise KeyError(f"Missing required columns after merge: {missing}")

        df_tags["product"] = (
            df_tags["tag"] + " " + df_tags["weight"].astype(str)
        ).str.upper()
        df_tags["order_date"] = pd.to_datetime(df_tags["order_date"])
        df_tags["week_num"] = df_tags["order_date"].dt.isocalendar().week
        df_tags["year"] = df_tags["order_date"].dt.year

        # group by using the detected join_key (not hard-coded 'asin')
        group_dims = [
            "week_num",
            "year",
            join_key,
            "item_name",
            "tag",
            "product",
            "event_name",
            "product_key",
        ]
        weekly_df = (
            df_tags.groupby(group_dims)[["shipped_units", "revenue_share_amt"]]
            .sum()
            .reset_index()
        )

        weekly_df["asp"] = weekly_df["revenue_share_amt"] / weekly_df["shipped_units"]
        weekly_df = weekly_df[weekly_df["asp"] != 0]

        asp_product = weekly_df.copy()
        asp_product["asp"] = asp_product["asp"].round(1)

        group_cols = ["tag", "product", "product_key", "asp", "event_name"]
        asp_product_df = (
            asp_product.groupby(group_cols)[["shipped_units", "revenue_share_amt"]]
            .sum()
            .reset_index()
        )
        asp_product_df = asp_product_df[asp_product_df["shipped_units"] >= 10]

        top_n_products = (
            asp_product_df.groupby("product")["revenue_share_amt"]
            .sum()
            .reset_index()
            .sort_values("revenue_share_amt", ascending=False)["product"]
            .head(self.top_n)
            .tolist()
        )

        return asp_product_df[
            asp_product_df["product"].isin(top_n_products)
            & (asp_product_df["event_name"] == "NO DEAL")
        ]


class ElasticityAnalyzer:
    @staticmethod
    def compute(topsellers: pd.DataFrame) -> pd.DataFrame:
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
        elasticity["pct_change_price"] = (
            (elasticity["asp_max"] - elasticity["asp_min"])
            / elasticity["asp_min"]
            * 100
        )
        elasticity["pct_change_qty"] = (
            (elasticity["shipped_units_max"] - elasticity["shipped_units_min"])
            / elasticity["shipped_units_min"]
            * 100
        )
        elasticity["ratio"] = (
            elasticity["pct_change_qty"] / elasticity["pct_change_price"]
        )
        elasticity["pct"] = elasticity["ratio"].rank(pct=True) * 100
        return elasticity.sort_values("ratio", ascending=False).reset_index(drop=True)


class GAMTuner:
    def __init__(self, expectile=0.5):
        self.expectile = expectile

    def fit(self, X, y, expectile=None):
        expt = self.expectile if expectile is None else expectile

        # scale X, y
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_mean, y_std = y.mean(), y.std()
        y_scaled = (y - y_mean) / (y_std if y_std != 0 else 1)

        # build terms safely
        nfeat = X_scaled.shape[1]
        terms = s(0) if nfeat == 1 else reduce(op.add, (s(i) for i in range(nfeat)))

        # model + grids
        gam = ExpectileGAM(terms, expectile=expt)
        lam_values = np.logspace(-2, 2, 5)  # 0.01, 0.1, 1, 10, 100
        spline_values = [5, 10, 20]
        order_values = [2, 3]

        try:
            gam = gam.gridsearch(
                X_scaled,
                y_scaled,
                lam=lam_values,
                n_splines=spline_values,
                spline_order=order_values,
            )
        except Exception as e:
            print(f"Gridsearch failed, falling back: {e}")
            gam = ExpectileGAM(s(0, n_splines=5), expectile=expt).fit(
                X_scaled, y_scaled
            )

        # stash scalers
        gam._scaler_X = scaler_X
        gam._y_mean = y_mean
        gam._y_std = y_std
        return gam


class GAMModeler:

    def __init__(self, topsellers: pd.DataFrame):
        self.topsellers = topsellers

    def run(self) -> pd.DataFrame:
        all_results = []
        for product, sub in self.topsellers.groupby("product"):
            X, y = sub[["asp"]], sub["shipped_units"]
            out = {}
            for q in [0.025, 0.5, 0.975]:
                gam = GAMTuner(expectile=q).fit(X, y)
                pred = (
                    gam.predict(gam._scaler_X.transform(X)) * gam._y_std + gam._y_mean
                )
                out[f"pred_{q}"] = pred
            preds = pd.DataFrame(out, index=sub.index)
            results = pd.concat(
                [sub[["asp", "product", "shipped_units"]], preds], axis=1
            )
            all_results.append(results)

        all_gam_results = pd.concat(all_results, axis=0)

        for col in [c for c in all_gam_results if c.startswith("pred_")]:
            all_gam_results[f"revenue_{col}"] = (
                all_gam_results["asp"] * all_gam_results[col]
            )

        rev_cols = [c for c in all_gam_results if c.startswith("revenue_pred_")]
        all_gam_results["revenue_pred_avg"] = all_gam_results[rev_cols].mean(axis=1)
        all_gam_results["revenue_actual"] = (
            all_gam_results["asp"] * all_gam_results["shipped_units"]
        )
        return all_gam_results


class Optimizer:
    """ """

    @staticmethod
    def run(all_gam_results: pd.DataFrame) -> dict:
        def pick_best(col):
            if col not in all_gam_results:
                return pd.DataFrame()
            return all_gam_results.loc[all_gam_results.groupby("product")[col].idxmax()]

        return {
            "best_avg": pick_best("revenue_pred_avg"),
            "best50": pick_best("revenue_pred_0.5"),
            "best975": pick_best("revenue_pred_0.975"),
            "best25": pick_best("revenue_pred_0.025"),
            "all_gam_results": all_gam_results,
        }


class PricingPipeline:
    def __init__(self, pricing_df, product_df, top_n=10):
        self.engineer = DataEngineer(pricing_df, product_df, top_n)

    def _build_curr_price_df(self) -> pd.DataFrame:
        """Compute current price as latest (by order_date) ASP per product_key."""
        # reuse same cleaning + join-key detection as DataEngineer
        pricing = self.engineer.pricing_df.copy()
        product = self.engineer.product_df.copy()

        # columns are already lower-cased by DataEngineer.prepare(), but be safe:
        pricing.columns = pricing.columns.str.strip().str.lower()
        product.columns = product.columns.str.strip().str.lower()

        candidate_keys = ["asin", "sku", "item_sku", "product_id", "parent_asin"]
        common_keys = [k for k in candidate_keys if k in pricing and k in product]
        if not common_keys:
            # schema-safe empty frame so UI won’t crash
            return pd.DataFrame(columns=["product_key", "product", "current_price"])

        join_key = common_keys[0]
        pricing[join_key] = pricing[join_key].astype(str).str.strip()
        product[join_key] = product[join_key].astype(str).str.strip()

        df = pricing.merge(product, how="left", on=join_key, validate="m:1")

        # ensure fields
        required = ["order_date", "shipped_units", "revenue_share_amt"]
        missing = [c for c in required if c not in df]
        if missing:
            return pd.DataFrame(columns=["product_key", "product", "current_price"])

        # stable id: use product_key if present, else fallback to join_key
        if "product_key" not in df.columns:
            df["product_key"] = df[join_key].astype(str)

        # product label for display
        if "tag" in df and "weight" in df:
            df["product"] = (df["tag"] + " " + df["weight"].astype(str)).str.upper()
        else:
            # fallback to product_key as name
            df["product"] = df["product_key"]

        df["order_date"] = pd.to_datetime(df["order_date"])

        # compute ASP at the raw row level; if you have multiple rows per day/key, group then ASP
        grp = df.groupby(["product_key", "product", "order_date"], as_index=False)[
            ["shipped_units", "revenue_share_amt"]
        ].sum()
        grp = grp[grp["shipped_units"] > 0]
        grp["asp"] = grp["revenue_share_amt"] / grp["shipped_units"]

        # pick latest date per product_key
        grp = grp.sort_values(["product_key", "order_date"])
        latest = grp.drop_duplicates("product_key", keep="last")

        curr_price_df = latest[["product_key", "product", "asp"]].rename(
            columns={"asp": "current_price"}
        )
        return curr_price_df.reset_index(drop=True)

    def assemble_dashboard_frames(self) -> dict:
        topsellers = self.engineer.prepare()
        elasticity_df = ElasticityAnalyzer.compute(topsellers)
        all_gam_results = GAMModeler(topsellers).run()
        bests = Optimizer.run(all_gam_results)

        best_avg = bests["best_avg"].merge(
            topsellers[["product", "product_key"]].drop_duplicates(),
            on="product",
            how="left",
        )

        # >>> NEW: compute current prices from raw data
        curr_price_df = self._build_curr_price_df()

        return {
            "price_quant_df": topsellers.groupby(["asp", "product"])["shipped_units"]
            .sum()
            .reset_index(),
            "best_avg": best_avg,
            "all_gam_results": all_gam_results,
            "best_optimal_pricing_df": best_avg[
                ["product", "product_key", "asp", "revenue_pred_avg", "revenue_actual"]
            ],
            "elasticity_df": elasticity_df[["product", "ratio", "pct"]],
            "curr_opt_df": best_avg,  # (your placeholder stays)
            "curr_price_df": curr_price_df,  # >>> now present with 'current_price'
        }

    @classmethod
    def from_csv_folder(
        cls,
        base_dir,
        data_folder="data",
        pricing_file="730d.csv",
        product_file="products.csv",
        top_n=10,
    ):
        """Convenience loader: read pricing + product CSVs and run the pipeline."""
        pricing_df = pd.read_csv(os.path.join(base_dir, data_folder, pricing_file))
        product_df = pd.read_csv(os.path.join(base_dir, data_folder, product_file))
        return cls(pricing_df, product_df, top_n).assemble_dashboard_frames()


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
        pred graph
        """
        product_lst = all_gam_results["product"].unique()
        pltly_qual = px.colors.qualitative.Dark24
        pltly_qual.extend(px.colors.qualitative.Vivid)
        colors = random.sample(pltly_qual, len(product_lst))
        color_dct = {p: colors[i] for i, p in enumerate(product_lst)}

        fig = go.Figure()
        for group_name, group_df in all_gam_results.groupby("product"):
            best_50 = group_df[
                group_df["revenue_pred_0.5"] == group_df["revenue_pred_0.5"].max()
            ].reset_index(drop=True)
            best_025 = group_df[
                group_df["revenue_pred_0.025"] == group_df["revenue_pred_0.025"].max()
            ].reset_index(drop=True)
            best_975 = group_df[
                group_df["revenue_pred_0.975"] == group_df["revenue_pred_0.975"].max()
            ].reset_index(drop=True)

            fig.add_trace(
                go.Scatter(
                    name=f"group {group_name} error",
                    x=group_df["asp"].tolist() + group_df["asp"].tolist()[::-1],
                    y=group_df["revenue_pred_0.975"].tolist()
                    + group_df["revenue_pred_0.025"].tolist()[::-1],
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
            px.histogram(
                elast_df, x="ratio", width=1200, height=600, template=self.template
            )
            .update_xaxes(title_text="Elasticity")
            .update_yaxes(title_text="Product Count")
        )
        return fig

    def opportunity_chart(elast_df, best50_df, curr_df, all_gam):
        """
        Bar chart of upside per product:
        upside = Expected revenue at recommended price - Expected revenue at current price
        """
        import plotly.express as px

        # quick guardrails
        for df in [elast_df, best50_df, curr_df, all_gam]:
            if df is None or getattr(df, "empty", True):
                return {}

        # ensure required columns exist
        if (
            "revenue_pred_0.5" not in all_gam.columns
            or "revenue_pred_0.5" not in best50_df.columns
        ):
            return {}

        # candidate product list
        prods = sorted(
            set(all_gam["product"])
            & set(best50_df["product"])
            & set(curr_df["product"])
        )
        if not prods:
            return {}

        rows = []
        for p in prods:
            try:
                # current price
                curr_price = curr_df.loc[curr_df["product"] == p, "current_price"]
                if curr_price.empty or pd.isna(curr_price.iloc[0]):
                    continue
                curr_price = float(curr_price.iloc[0])

                # expected curve points
                prod = all_gam[
                    (all_gam["product"] == p)
                    & pd.notna(all_gam["asp"])
                    & pd.notna(all_gam["revenue_pred_0.5"])
                ]
                if prod.empty:
                    continue

                # revenue at current price (nearest ASP)
                idx = (prod["asp"] - curr_price).abs().idxmin()
                rev_curr = float(prod.loc[idx, "revenue_pred_0.5"])

                # revenue at recommended price (from best50 table)
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
            # title="Upside vs Elasticity (Top Opportunities)",
        )
        fig.update_yaxes(
            title_text="Upside (Expected Revenue Δ)",
            tickprefix="$",
            separatethousands=True,
        )
        # fig.update_xaxes(title_text="")
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
        fig.add_annotation(
            text=title, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper"
        )
        return fig
