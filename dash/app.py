# app.py

import os
import warnings
from dash import Dash, html, dash_table, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import random
from datetime import datetime
import dash_bootstrap_components as dbc

# local imports
from built_in_logic import output_key_dfs, viz
from navbar import get_navbar
from home import Homepage
import overview, opps, faq

warnings.filterwarnings("ignore")


# ---------- load data & build frames ----------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# If /dash/app.py sits one level below project root, go up one level to find /data:
PROJECT_BASE = os.path.dirname(BASE_DIR)

d = output_key_dfs.from_csv_folder(
    base_dir=PROJECT_BASE,
    data_folder="data",
    pricing_file="730d.csv",
    product_file="products.csv",
    top_n=10,
)
products_lookup = d["best50_optimal_pricing_df"][['product', 'product_key']].drop_duplicates().reset_index(drop=True)

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server

# STRICT validation layout — pass lookup, not a list of names
app.validation_layout = html.Div([
    dcc.Location(id="url"),
    get_navbar(),
    Homepage(),
    overview.layout(products_lookup),                    # << here
    # descriptive.layout(products_lookup["product"].tolist()),
    opps.layout(d["curr_opt_df"]),
    faq.faq_section(),
    html.Div(id="page-content"),
])

# Main layout with simple router area
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        dcc.Loading(
            id="router-loader",
            type="circle",
            children=html.Div(id="page-content",
                              style={"minHeight": "65vh", "padding": "12px"}),
        ),
    ]
)


# Router
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(path):
    if path in ["/", "", None]:
        return Homepage()
    elif path == "/overview":
        return overview.layout(products_lookup)          # << here
    elif path == "/faq":
        return faq.faq_section()
    elif path == "/opps":
        return opps.layout(d["curr_opt_df"])
    return html.Div("404 - Not found", className="p-4")

# Register callbacks
overview.register_callbacks(
    app,
    d["price_quant_df"],
    d["best50_optimal_pricing_df"],
    d["curr_price_df"],
    d["elasticity_df"],
    d["all_gam_results"],
    products_lookup,   # << pass lookup to callbacks too
    viz,
)
 

# descriptive expects price_quant_df
# descriptive.register_callbacks(app, d["price_quant_df"], viz)

# opps expects the opportunity table df
opps.register_callbacks(app, d["curr_opt_df"])

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


# in terminal:

# ----------- dev -----------
cd dash

# activate pyvenv
../.venv/Scripts/activate                # windows
source ../.pricing-venv/bin/activate     # mac

# run script
py app.py                               # windows
python3 app.py                          # mac

host:
http://127.0.0.1:8050/
"""
