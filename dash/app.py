# app.py
from datetime import datetime
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# local modules
from built_in_logic import viz as Viz
from navbar import get_navbar
from home import Homepage
import overview, opps, faq
from helpers import Paths, Cache, DataEng

import warnings

warnings.filterwarnings("ignore")


# ---------- Load data (cached) ----------
BASE_DIR = Paths.BASE_DIR
PROJECT_BASE = Paths.PROJECT_BASE


frames = Cache.build_frames_with_cache(PROJECT_BASE)
price_quant_df = frames["price_quant_df"]
best_avg_df = frames["best_avg"]
all_gam_results = frames["all_gam_results"]
best_optimal_pricing = frames["best_optimal_pricing_df"]
elasticity_df = frames["elasticity_df"]
curr_opt_df = frames["curr_opt_df"]
curr_price_df = frames["curr_price_df"]
opps_summary = frames["opps_summary"]
meta = frames["meta"]  # {data_start, data_end, days_covered, annual_factor}

# --- Build lookups ---
lookup_all = DataEng.make_products_lookup(
    best_optimal_pricing, best_avg_df, curr_price_df, all_gam_results
)

dropdown_lookup = (
    best_optimal_pricing[["product", "asin"]]
    .dropna(subset=["asin"])
    .drop_duplicates(subset=["product"])
    .astype({"asin": str})
    .reset_index(drop=True)
)

if "revenue_pred_0.5" in all_gam_results.columns:
    idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
    best50_optimal_pricing = (
        all_gam_results.loc[
            idx, ["product", "asp", "revenue_pred_0.5", "units_pred_0.5"]
        ]
        .reset_index(drop=True)
        .merge(dropdown_lookup, on="product", how="left")[
            ["product", "asin", "asp", "revenue_pred_0.5", "units_pred_0.5"]
        ]
    )

else:
    best50_optimal_pricing = (
        best_optimal_pricing[["product", "asin", "asp"]]
        .merge(
            all_gam_results[["product", "asp", "revenue_pred_0.5", "pred_0.5"]],
            on=["product", "asp"],
            how="left",
        )
        .drop_duplicates(subset=["product"])
        .reset_index(drop=True)
    )

# ---------- app ----------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
)
server = app.server

app.validation_layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        Homepage(),
        overview.layout(dropdown_lookup),
        opps.layout(curr_opt_df),
        faq.faq_section(),
        html.Div(id="page-content"),
    ]
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        dcc.Loading(
            id="router-loader",
            type="circle",
            children=html.Div(
                id="page-content", style={"minHeight": "65vh", "padding": "12px"}
            ),
        ),
    ]
)


# ---------- Router ----------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(path):
    if path in ["/", "", None]:
        return Homepage()
    elif path == "/overview":
        return overview.layout(dropdown_lookup)
    elif path == "/faq":
        return faq.faq_section()
    elif path == "/opps":
        return opps.layout(curr_opt_df)
    return html.Div("404 - Not found", className="p-4")


# ---------- Register callbacks ----------
overview.register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing,
    curr_price_df,
    elasticity_df,
    all_gam_results,
    dropdown_lookup,
    meta,
    Viz,
)

opps.register_callbacks(
    app,
    {
        "elasticity_df": elasticity_df,
        "best50_optimal_pricing_df": best50_optimal_pricing,
        "curr_price_df": curr_price_df,
        "all_gam_results": all_gam_results,
    },
    curr_opt_df,
    Viz,
)

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
---------- Instructions to run app.py ----------
# always pull from Github first 
git pull origin main        # mac
git pull origin master      # windows

# in terminal:
cd dash


# ------------- mac -------------
# install pyenv
pyenv install 3.11.7
pyenv global 3.11.7
python -m venv .pricing-venv
source ../pricing-venv/bin/activate      # activate
python3 app.py                           # run

# ------------- winsows -------------
../.venv/Scripts/activate                 # activate
 py app.py                                # run

# ------- optional:  formatter -------

python3 -m black helpers.py

"""
