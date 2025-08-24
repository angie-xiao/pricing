# """
# # in terminal:

# .venv\Scripts\activate.bat

# pip3 install scikit-learn
# pip3 install plotnine
# """

import os
import random

# Data Wrangling
import pandas as pd
import numpy as np

# Modeling
from sklearn.preprocessing import LabelEncoder
from pygam import GAM, ExpectileGAM, s, l, f
import statsmodels.api as sm

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
from plotnine import (
    ggplot,
    aes,
    geom_ribbon,
    geom_line,
    facet_wrap,
    labs,
    theme,
    geom_point,
)
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly as ggplotly
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# import plotly.io as pio


class output_key_dfs:
    """ """

    def __init__(self, pricing_df, product_df, top_n=10):
        """
        params
            pricing_df: df
            product_df: df
            top_n: int
        """
        self.pricing_df = pricing_df
        self.product_df = product_df
        self.top_n = top_n

    def data_engineer(self):
        """
        Execute following steps:
            1) formatting (lower str, date, product variation, 1 decimal)
            2) merge
            3) aggregation
            4) calculate asp
            5) filter -
                5.1) exclude where shipped units < 10
                5.2) focus on top N top-sellers
                5.3) focus on BAU

        return
            df: asp_product_topsellers
        """

        # formatting
        self.pricing_df.columns = [x.lower() for x in self.pricing_df.columns]
        self.product_df.columns = [x.lower() for x in self.product_df.columns]

        # merge
        df_tags = self.pricing_df.merge(self.product_df, how="left", on="asin")

        # add a product tag, + weight
        df_tags["product"] = df_tags["tag"] + " " + df_tags["weight"].astype(str)

        # date
        df_tags["order_date"] = pd.to_datetime(df_tags["order_date"])
        df_tags["week_num"] = df_tags["order_date"].dt.isocalendar().week
        df_tags["year"] = df_tags["order_date"].dt.year

        # aggregate on weekly basis
        weekly_df_tags = (
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

        # calculate aps
        weekly_df_tags["asp"] = (
            weekly_df_tags["revenue_share_amt"] / weekly_df_tags["shipped_units"]
        )
        weekly_df_tags = weekly_df_tags[weekly_df_tags["asp"] != 0]

        # 1 decimal
        asp_product = weekly_df_tags.copy()
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

        # data points where shipped units < 10 (too little evidence)
        asp_product_df = asp_product_df[asp_product_df["shipped_units"] >= 10]

        # only focus on top sellers
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

        # filter for topsellers
        asp_product_topsellers = asp_product_df[
            asp_product_df["product"].isin(top_n_product_lst)
        ]

        asp_product_topsellers = asp_product_topsellers[
            asp_product_topsellers["event_name"] == "NO DEAL"
        ]

        # print('-' * 10, ' finished data engineering', '-'*10, '\n')

        return asp_product_topsellers

    def elasticity(self):
        """
        Calculate the price elasticity of each of our products:
            - % change in quantity demanded / % change in price
            - Unscented 40lb shows the highest elasticity (i.e., largest lift in units with each dollar price decreased)

        return
            df: elasticity
        """
        asp_product_topsellers = self.data_engineer()

        # Calculating Price-Demand elasticity
        elasticity = (
            asp_product_topsellers.groupby("product")
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
            .assign(
                ratio=lambda d: d["pct_change_qty"] / d["pct_change_price"]
            )  # demand elasticity
        )

        elasticity.sort_values(by="ratio", ascending=False).reset_index(drop=True)

        return elasticity

    def price_quant(self):
        """
        output df for price vs shipped units by product type

        return
            df
        """
        asp_product_topsellers = self.data_engineer()
        price_quantity = (
            asp_product_topsellers[["asp", "shipped_units", "product"]]
            .groupby(["asp", "product"])
            .sum()
            .reset_index()
        )

        return price_quantity

    def model(self):
        """
        fit GAM for each product, find [2.5%, 97.5%] CI

        return
            df
        """
        df_tags = self.data_engineer()

        unique_prod = df_tags["product"].unique()

        all_gam_results = pd.DataFrame()
        for product in unique_prod:  # Loop through products

            if product and str(product) == product:  # skip nan/none

                product_data = df_tags[
                    df_tags["product"] == product
                ]  # Filter for current product

                # Predictors & target split
                X = product_data[["asp"]]
                y = product_data["shipped_units"]

                # List of quantiles for modeling
                quantiles = [0.025, 0.5, 0.975]
                gam_results = {}

                # Fit the GAM model
                for q in quantiles:
                    gam = ExpectileGAM(s(0), expectile=q)  # initiate the model
                    gam.fit(X, y)  # fit
                    gam_results[f"pred_{q}"] = gam.predict(
                        X
                    )  # predict for that quantile
                    # print(q, "|", product, "|", gam.deviance_residuals(X,y).mean())
                # print("-----------\n")

                # Store the results in a DF
                predictions_gam = pd.DataFrame(gam_results).set_index(X.index)
                predictions_gam_df = pd.concat(
                    [
                        product_data[["asp", "product", "shipped_units"]],
                        predictions_gam,
                    ],
                    axis=1,
                )
                all_gam_results = pd.concat(
                    [all_gam_results, predictions_gam_df], axis=0
                )

        # print("-"*10, ' finished modeling ', "-"*10, "\n")

        return all_gam_results

    def optimization(self):
        """
        return
            dct: a dictionary of best 2.5%, 50%, 97.5% dfs
        """
        all_gam_results = self.model()

        # Calculate Revenue for each predicted price band
        for col in all_gam_results.columns:
            if col.startswith("pred"):
                all_gam_results["revenue_" + col] = (
                    all_gam_results["asp"] * all_gam_results[col]
                )

        # Actual revenue
        all_gam_results["revenue_actual"] = (
            all_gam_results["asp"] * all_gam_results["shipped_units"]
        )

        # View
        # all_gam_results.sample(2)

        # Calculating where the predicted median revenue is the max
        best_50 = (
            all_gam_results.groupby("product")
            .apply(
                lambda x: x[x["revenue_pred_0.5"] == x["revenue_pred_0.5"].max()].head(
                    1
                )
            )
            .reset_index(level=0, drop=True)
        )

        # Calculating where the predicted 97.5% percentile revenue is the max
        best_975 = (
            all_gam_results.groupby("product")
            .apply(
                lambda x: x[
                    x["revenue_pred_0.975"] == x["revenue_pred_0.975"].max()
                ].head(1)
            )
            .reset_index(level=0, drop=True)
        )

        # Calculating where the predicted 2.5% percentile revenue is the max
        best_025 = (
            all_gam_results.groupby("product")
            .apply(
                lambda x: x[
                    x["revenue_pred_0.025"] == x["revenue_pred_0.025"].max()
                ].head(1)
            )
            .reset_index(level=0, drop=True)
        )

        # only keep 2 decimals
        for df in [best_50, best_975, best_025]:
            for col in [
                "pred_0.025",
                "pred_0.5",
                "pred_0.975",
                "revenue_pred_0.025",
                "revenue_pred_0.5",
                "revenue_pred_0.975",
                "revenue_actual",
            ]:
                df[col] = round(df[col], 2)

        d = {
            "best50": best_50,
            "best975": best_975,
            "best25": best_025,
            "all_gam_results": all_gam_results,
        }

        # print("-"*10, ' finished pricing optimization ', "-"*10, "\n")

        return d

    def initial_dfs(self):
        """
        return a dictionary of dfs needed to get modeling steps started
        """
        price_quant_df = output_key_dfs(
            self.pricing_df, self.product_df, 10
        ).price_quant()  # key df #1
        d = output_key_dfs(
            self.pricing_df, self.product_df, 10
        ).optimization()  # modeling + optimization
        best50 = d["best50"]  # key df #2
        all_gam_results = d["all_gam_results"]

        dct = {
            "price_quant_df": price_quant_df,
            "best50": best50,
            "all_gam_results": all_gam_results,
        }

        return dct

    def grid_search(self):
        return


class viz:
    def __init__(self, template="lux"):
        """ """
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

    def price_quantity(self, price_quant_df):
        """
        Draw price & quantity scatterplot graph

        params
            folder: (sub) folder where data file is
            filename: data file to graph
        """

        # plot
        fig = (
            px.scatter(
                price_quant_df,
                x="asp",
                y="shipped_units",
                log_y=True,
                color="product",
                # opacity=.5,
                width=1200,
                height=600,
                trendline="lowess",  # used when the relationship is curved
                trendline_color_override="#CD9C20",
                title="Product Sales: Price vs Shipped Units",
                template=self.template,
            )
            .update_traces(marker=dict(size=7))
            .update_layout(legend_title_text="Product", yaxis_range=[0, None])
            .update_xaxes(title_text="Price")
            .update_yaxes(title_text="Shipped Units")
        )

        return fig

    def gam_results(self, all_gam_results):
        """
        params
            df: pricing df
            tags: product df

        return
            ggplot obj
        """

        # map color to product
        product_lst = all_gam_results["product"].unique()

        # color
        pltly_qual = px.colors.qualitative.Dark24  # color palette
        pltly_qual.extend(px.colors.qualitative.Vivid)
        colors = random.sample(pltly_qual, len(product_lst))

        color_dct = dict()

        i = 0
        while i < len(product_lst):
            color_dct[product_lst[i]] = colors[i]
            i += 1

        # plot
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

            # rev_actual_name = f"Revenue Actual - {}"

            # error band
            fig.add_trace(
                go.Scatter(
                    name=f"group {group_name} error",
                    x=group_df["asp"].tolist()
                    + group_df["asp"].tolist()[::-1],  # x, then x reversed
                    y=group_df["revenue_pred_0.975"].tolist()
                    + group_df["revenue_pred_0.025"].tolist()[
                        ::-1
                    ],  # upper, then lower reversed
                    fill="toself",
                    fillcolor="#cbcbcb",
                    line=dict(color="#cbcbcb"),
                    legendgroup=group_name,
                    showlegend=False,
                    opacity=0.4,
                )
            )

            # scatter plot (actual)
            fig.add_trace(
                go.Scatter(
                    x=group_df["asp"],
                    y=group_df["revenue_actual"],
                    mode="markers",
                    name="Revenue Actual",  # Name for the legend
                    marker=dict(symbol="x", color=color_dct[group_name], size=10),
                    # legendgroup=group_name,
                    opacity=0.5,
                )
            )

            # Adding the expected mean
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

            # conservative
            fig.add_trace(
                go.Scatter(
                    x=best_025["asp"],
                    y=best_025["revenue_pred_0.025"],
                    mode="markers",
                    marker=dict(color="#AAB396", size=18),
                    name="Conservative Prediction",
                )
            )

            # optimistic
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
            yaxis_type="log",
            title="BAU GAM Results",
            width=1200,
            height=650,
            template=self.template,
        )

        return fig

    def elasticity(self, asp_product_topsellers):
        """
        actually need to calculate for discount here
        """
        fig = (
            px.scatter(
                asp_product_topsellers,
                x="asp",
                y="shipped_units",
                facet_col="product",
                category_orders={
                    # "product": [
                    #     'Unscented 6.5','Unscented 28.0', 'Unscented 40.0',
                    #     'Probiotic 16.0', 'Probiotic 28.0', 'Probiotic 40.0',
                    #     'Extra Strength 16.0','Extra Strength 28.0', 'Extra Strength 40.0',
                    # ]
                },
                # facet_col_spacing=0.1,
                facet_row_spacing=0.15,
                facet_col_wrap=3,
                # # color='current_discount_percent',
                color_continuous_scale=px.colors.sequential.Oryel,
                trendline="lowess",
                title="Elasticiy Analysis",
                width=1200,
                height=600,
                template="plotly",
                log_y=True,
            )
            .update_traces(marker=dict(size=7))
            .update_layout(legend_title_text="Product", title_font=dict(size=16))
            .update_xaxes(
                title_text="Price", title_font=dict(size=10), tickfont=dict(size=10)
            )
            .update_yaxes(
                title_text="Shipped Units",
                title_font=dict(size=10),
                tickfont=dict(size=10),
            )
        )

        fig.update_layout(xaxis=dict(type="category"))

        for annotation in fig["layout"]["annotations"]:
            annotation["font"] = dict(size=10)

        fig.for_each_yaxis(
            lambda yaxis: yaxis.update(showticklabels=True)
        )  # show ticks
        fig.for_each_xaxis(
            lambda xaxis: xaxis.update(showticklabels=True)
        )  # show ticks
        fig.update_yaxes(matches=None)  # don't share y axis
        fig.update_xaxes(matches=None)  # don't share x axis

        return fig
