# app

# make a graph for elasticity to give the user a better sense in terms of distribution among their products
# top opps for asins w greatest gaps bw rec price & curr price
# take outliers into account


import os
import warnings
from dash import Dash, html, dash_table, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import random
from datetime import datetime
import dash_bootstrap_components as dbc

# scripts from app
from built_in_logic import output_key_dfs, viz
from navbar import get_navbar
from home import Homepage
import overview, descriptive


warnings.filterwarnings("ignore")


class get_dfs:

    def __init__(
        self, data_folder="data", pricing_file="730d.csv", product_file="products.csv"
    ):
        """
        must have the 2 files to start with
        """
        # read
        dash_folder = os.path.dirname(os.path.realpath(__file__))
        pricing_folder = os.path.dirname(dash_folder)
        data_folder = os.path.join(pricing_folder, data_folder)
        pricing_path, product_path = os.path.join(
            data_folder, pricing_file
        ), os.path.join(data_folder, product_file)
        # pd dfs
        self.pricing_df, self.product_df = pd.read_csv(pricing_path), pd.read_csv(
            product_path
        )

    def return_dct(self):
        """
        return
            a dictionary of dfs needed to get Dash running
        """
        dct = output_key_dfs(self.pricing_df, self.product_df, 10).initial_dfs()
        price_quant_df, best50, all_gam_results = (
            dct["price_quant_df"],
            dct["best50"],
            dct["all_gam_results"],
        )
        best50_optimal_pricing_df = best50[["product", "asp"]]
        asp_product_topsellers = output_key_dfs(
            self.pricing_df, self.product_df, 10
        ).data_engineer()

        tmp_curr_price = self.product_df.copy()
        tmp_curr_price["product"] = (
            self.product_df["tag"] + " " + self.product_df["weight"].astype(str)
        )
        curr_price_df = tmp_curr_price[["product", "current_price"]]

        elasticity_df = (
            output_key_dfs(self.pricing_df, self.product_df, 10)
            .elasticity()
            .rename(columns={"ratio": "ratio"})[["product", "ratio"]]
            .sort_values(by=["ratio"], ascending=False)
        )
        elasticity_df["ratio"] = round(elasticity_df["ratio"], 2)

        dct_output = {
            "pricing_df": self.pricing_df,
            "product_df": self.product_df,
            "price_quant_df": price_quant_df,
            "best50": best50,
            "all_gam_results": all_gam_results,
            "best50_optimal_pricing_df": best50_optimal_pricing_df,
            "asp_product_topsellers": asp_product_topsellers,
            "elasticity_df": elasticity_df,
            "curr_price_df": curr_price_df,
        }

        return dct_output


#################################### GET DFS ####################################

d = get_dfs(
    data_folder="data", pricing_file="730d.csv", product_file="products.csv"
).return_dct()
pricing_df = d["pricing_df"]
product_df = d["product_df"]
price_quant_df = d["price_quant_df"]
best50 = d["best50"]
all_gam_results = d["all_gam_results"]
best50_optimal_pricing_df = d["best50_optimal_pricing_df"]
asp_product_topsellers = d["asp_product_topsellers"]
elasticity_df = d["elasticity_df"]
curr_price_df = d["curr_price_df"]


################################# INITIATE DASH ##################################

app = Dash(
    __name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True
)
server = app.server

products = sorted(price_quant_df["product"].unique().tolist())

# Full tree for validation only (not shown to users)
app.validation_layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        Homepage(),
        overview.layout(products),
        # predictive.layout(products),
        descriptive.layout(products),
        html.Div(id="page-content"),
    ]
)

# app.py (layout section)
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        dcc.Loading(
            id="router-loader",
            type="circle",  # options: "graph", "cube", "circle", "dot", "default"
            children=html.Div(
                id="page-content", style={"minHeight": "65vh", "padding": "12px"}
            ),
        ),
    ]
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(path):
    print(">> pathname =", path)  # DEBUG: watch your terminal
    if path in ["/", "", None]:
        return Homepage()
    if path == "/overview":
        return overview.layout(products)
    
    if path == "/descriptive":
        return descriptive.layout(products)
    
    # if path == "/predictive":
    #     return predictive.layout(products)
    
    if path == "/faq":
        return html.Div("FAQ page TBD", className="p-4")
    return html.Div("404 - Not found", className="p-4")


# Register per-page callbacks (after app is created)
overview.register_callbacks(
    app, price_quant_df, best50_optimal_pricing_df, curr_price_df, elasticity_df, all_gam_results, viz
)
# predictive.register_callbacks(app, all_gam_results, viz)
descriptive.register_callbacks(app, price_quant_df, viz)


print(
    "\n",
    "-" * 10,
    datetime.now().strftime("%H:%M:%S"),
    " Page Updated " + "-" * 10,
    "\n",
)


if __name__ == "__main__":
    app.run(debug=True)


"""
ref:
https://dash.plotly.com/tutorial
https://medium.com/@wolfganghuang/advanced-dashboards-with-plotly-dash-things-to-consider-before-you-start-9754ac91fd10

# activate pyvenv
 ../.venv/Scripts/activate   # windows
pyenv shell pricing-venv    # mac

# in terminal:
cd dash


# call the app
py app.py # winddows
python3 app.py  # mac


host:
http://127.0.0.1:8050/
"""
