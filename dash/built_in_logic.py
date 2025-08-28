# built_in_logic.py

import os
import random
import pandas as pd
import numpy as np

from pygam import ExpectileGAM, s
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template


class output_key_dfs:
    """
    Encapsulates:
      - data engineering
      - GAM modeling
      - optimization outputs
      - normalized, dashboard-ready frames
    """

    # ---------- init & utilities ----------
    def __init__(
        self, pricing_df: pd.DataFrame, product_df: pd.DataFrame, top_n: int = 10
    ):
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n

    def _key_series(self, s):
        return (
            s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
        )

    def _add_key(self, df):
        if df is None or "product" not in df.columns:
            return df
        df = df.copy()
        df["product_key"] = self._key_series(df["product"])
        return df

    @staticmethod
    def _normalize_product_col(df: pd.DataFrame, col: str = "product") -> pd.DataFrame:
        if df is None or col not in df.columns:
            return df
        out = df.copy()
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )
        return out

    # ---------- file helpers (new) ----------
    @classmethod
    def from_csv_folder(
        cls,
        base_dir: str,
        data_folder: str = "data",
        pricing_file: str = "730d.csv",
        product_file: str = "products.csv",
        top_n: int = 10,
    ) -> dict:
        """
        Read CSVs and return the dashboard-ready dictionary (same structure app expects).
        """
        data_dir = os.path.join(base_dir, data_folder)
        pricing_path = os.path.join(data_dir, pricing_file)
        product_path = os.path.join(data_dir, product_file)

        pricing_df = pd.read_csv(pricing_path)
        product_df = pd.read_csv(product_path)

        return cls(pricing_df, product_df, top_n=top_n).assemble_dashboard_frames()

    # ---------- core pipeline ----------
    def data_engineer(self) -> pd.DataFrame:
        self.pricing_df.columns = [x.lower() for x in self.pricing_df.columns]
        self.product_df.columns = [x.lower() for x in self.product_df.columns]

        df_tags = self.pricing_df.merge(self.product_df, how="left", on="asin")
        df_tags["product"] = df_tags["tag"] + " " + df_tags["weight"].astype(str)
        df_tags["product"] = df_tags["product"].str.upper()

        df_tags["order_date"] = pd.to_datetime(df_tags["order_date"])
        df_tags["week_num"] = df_tags["order_date"].dt.isocalendar().week
        df_tags["year"] = df_tags["order_date"].dt.year

        weekly_df = (
            df_tags[
                [
                    "week_num",
                    "year",
                    "asin",
                    "item_name",
                    "tag",
                    "event_name",
                    "product",
                    "shipped_units",
                    "revenue_share_amt",
                ]
            ]
            .groupby(
                [
                    "week_num",
                    "year",
                    "asin",
                    "item_name",
                    "tag",
                    "product",
                    "event_name",
                ]
            )
            .sum()
            .reset_index()
        )

        weekly_df["asp"] = weekly_df["revenue_share_amt"] / weekly_df["shipped_units"]
        weekly_df = weekly_df[weekly_df["asp"] != 0]

        asp_product = weekly_df.copy()
        asp_product["asp"] = round(asp_product["asp"], 1)

        asp_product_df = (
            asp_product[
                [
                    "product",
                    "asp",
                    "tag",
                    "event_name",
                    "shipped_units",
                    "revenue_share_amt",
                ]
            ]
            .groupby(["tag", "product", "asp", "event_name"])
            .sum()
            .reset_index()
        )

        asp_product_df = asp_product_df[asp_product_df["shipped_units"] >= 10]

        top_n_product_lst = (
            asp_product_df[["product", "revenue_share_amt"]]
            .groupby("product")
            .sum()
            .reset_index()
            .sort_values(by=["revenue_share_amt"], ascending=False)["product"][
                : self.top_n
            ]
            .tolist()
        )

        topsellers = asp_product_df[asp_product_df["product"].isin(top_n_product_lst)]
        topsellers = topsellers[topsellers["event_name"] == "NO DEAL"]
        return topsellers

    def elasticity(self) -> pd.DataFrame:
        topsellers = self.data_engineer()
        elasticity = (
            topsellers.groupby("product")
            .agg(
                {
                    "asp": ["max", "min"],
                    "shipped_units": ["max", "min"],
                    "product": "count",
                }
            )
            .reset_index()
            .pipe(
                lambda d: d.set_axis(
                    [
                        "product",
                        "asp_max",
                        "asp_min",
                        "shipped_units_max",
                        "shipped_units_min",
                        "product_count",
                    ],
                    axis=1,
                )
            )
            .assign(
                pct_change_price=lambda d: (d["asp_max"] - d["asp_min"])
                / d["asp_min"]
                * 100,
                pct_change_qty=lambda d: (
                    d["shipped_units_max"] - d["shipped_units_min"]
                )
                / d["shipped_units_min"]
                * 100,
            )
            .assign(ratio=lambda d: d["pct_change_qty"] / d["pct_change_price"])
        )
        elasticity.sort_values(by="ratio", ascending=False).reset_index(drop=True)

        # + percentile col
        elasticity['pct'] = elasticity['ratio'].rank(pct=True) * 100

        return elasticity

    def price_quant(self) -> pd.DataFrame:
        topsellers = self.data_engineer()
        return (
            topsellers[["asp", "shipped_units", "product"]]
            .groupby(["asp", "product"])
            .sum()
            .reset_index()
        )

    def model(self) -> pd.DataFrame:
        df = self.data_engineer()
        unique_prod = df["product"].unique()
        all_gam_results = pd.DataFrame()

        for product in unique_prod:
            if not product or product != str(product):
                continue
            sub = df[df["product"] == product]

            X = sub[["asp"]]
            y = sub["shipped_units"]
            qs = [0.025, 0.5, 0.975]
            out = {}
            for q in qs:
                gam = ExpectileGAM(s(0), expectile=q)
                gam.fit(X, y)
                out[f"pred_{q}"] = gam.predict(X)

            preds = pd.DataFrame(out, index=X.index)
            results = pd.concat(
                [sub[["asp", "product", "shipped_units"]], preds], axis=1
            )
            all_gam_results = pd.concat([all_gam_results, results], axis=0)

        # revenues (pred + actual)
        for col in all_gam_results.columns:
            if col.startswith("pred_"):
                all_gam_results["revenue_" + col] = (
                    all_gam_results["asp"] * all_gam_results[col]
                )
        all_gam_results["revenue_actual"] = (
            all_gam_results["asp"] * all_gam_results["shipped_units"]
        )
        return all_gam_results

    def optimization(self) -> dict:
        all_gam_results = self.model()

        def pick_best(col):
            return (
                all_gam_results.groupby("product")
                .apply(lambda x: x[x[col] == x[col].max()].head(1))
                .reset_index(level=0, drop=True)
            )

        best_50 = pick_best("revenue_pred_0.5")
        best_975 = pick_best("revenue_pred_0.975")
        best_025 = pick_best("revenue_pred_0.025")

        for df in [best_50, best_975, best_025]:
            for c in [
                "pred_0.025",
                "pred_0.5",
                "pred_0.975",
                "revenue_pred_0.025",
                "revenue_pred_0.5",
                "revenue_pred_0.975",
                "revenue_actual",
            ]:
                if c in df.columns:
                    df[c] = df[c].round(2)

        return {
            "best50": best_50,
            "best975": best_975,
            "best25": best_025,
            "all_gam_results": all_gam_results,
        }

    def initial_dfs(self) -> dict:
        price_quant_df = self.price_quant()
        d = self.optimization()
        best50 = d["best50"]
        all_gam_results = d["all_gam_results"]

        # normalize join keys
        price_quant_df = self._normalize_product_col(price_quant_df)
        best50 = self._normalize_product_col(best50)
        all_gam_results = self._normalize_product_col(all_gam_results)

        return {
            "price_quant_df": price_quant_df,
            "best50": best50,
            "all_gam_results": all_gam_results,
        }

    # ---------- one call used by app ----------
    def assemble_dashboard_frames(self) -> dict:
        """ """
        base = self.initial_dfs()
        price_quant_df = self._add_key(base["price_quant_df"])
        best50 = self._add_key(base["best50"])
        all_gam_results = self._add_key(base["all_gam_results"])

        # current price (keep display name; add key)
        tmp = self.product_df.copy()
        tmp["product"] = tmp["tag"] + " " + tmp["weight"].astype(str)
        curr_price_df = tmp[["product", "current_price"]].copy()
        curr_price_df = self._add_key(curr_price_df)

        # product lookup for UI (display label + key)
        products_lookup = (
            curr_price_df[["product_key", "product"]]
            .drop_duplicates()
            .sort_values("product")
            .reset_index(drop=True)
        )

        # best50 we need for upside
        best50_optimal_pricing_df = best50[
            ["product", "product_key", "asp", "revenue_pred_0.5", "revenue_actual"]
        ].copy()

        # elasticity — do NOT trim here; keep full distribution you computed
        elasticity_df = self.elasticity()[["product", "ratio", "pct"]].copy()
        elasticity_df = (
            self._add_key(elasticity_df)
            .sort_values("ratio", ascending=False)
            .reset_index(drop=True)
        )

        # opportunities table
        curr_opt_df = curr_price_df.merge(
            best50_optimal_pricing_df,
            on=["product_key"],
            how="left",
            suffixes=("", "_b50"),
        )
        curr_opt_df["price_gap"] = (
            curr_opt_df["asp"] - curr_opt_df["current_price"]
        ).round(2)
        curr_opt_df["revenue_gap"] = (
            curr_opt_df["revenue_pred_0.5"] - curr_opt_df["revenue_actual"]
        ).round(2)
        curr_opt_df.rename(
            columns={"asp": "rec_price", "revenue_pred_0.5": "avg_pred_revenue"},
            inplace=True,
        )
        keep_cols = [
            "product",
            "product_key",
            "revenue_gap",
            "avg_pred_revenue",
            "revenue_actual",
            "price_gap",
            "rec_price",
            "current_price",
        ]
        curr_opt_df = (
            curr_opt_df[[c for c in keep_cols if c in curr_opt_df.columns]]
            .sort_values("revenue_gap", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "products_lookup": products_lookup,
            "price_quant_df": price_quant_df,
            "best50": best50,
            "all_gam_results": all_gam_results,
            "best50_optimal_pricing_df": best50_optimal_pricing_df,
            "elasticity_df": elasticity_df,
            "curr_price_df": curr_price_df,
            "curr_opt_df": curr_opt_df,
        }


# ---------- Plot helpers (unchanged from your working version) ----------
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
        '''
        pred graph
        '''
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
            legend=dict(
                orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0                
            ),
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
            yaxis={'categoryorder': 'total descending'}
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


# """
# # in terminal:

# .venv\Scripts\activate.bat

# pip3 install scikit-learn
# pip3 install plotnine
# """
