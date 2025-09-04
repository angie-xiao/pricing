# app.py
import os
import warnings
from datetime import datetime

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

# local imports
from built_in_logic import PricingPipeline, viz as Viz
from navbar import get_navbar
from home import Homepage
import overview, opps, faq


warnings.filterwarnings("ignore")

# ---------- load data & build frames ----------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))  # .../pricing/dash
PROJECT_BASE = os.path.dirname(BASE_DIR)                # .../pricing

d = PricingPipeline.from_csv_folder(PROJECT_BASE)
price_quant_df        = d["price_quant_df"]
best_avg_df           = d["best_avg"]
all_gam_results       = d["all_gam_results"]
best_optimal_pricing  = d["best_optimal_pricing_df"]
elasticity_df         = d["elasticity_df"]
curr_opt_df           = d["curr_opt_df"]
curr_price_df         = d["curr_price_df"]

# --- Debug logging for sanity check ---
print("\n--- Current Price DF ---")
print("Shape:", curr_price_df.shape)
print("Columns:", curr_price_df.columns.tolist())
print(curr_price_df.head().to_string(index=False))
print("--- End Current Price DF ---\n")

# product ↔ key lookup for dropdowns / joins
products_lookup = (
    best_optimal_pricing[["product", "product_key"]]
    .drop_duplicates()
    .reset_index(drop=True)
)


# 1) Enrich frames with product_key BEFORE any selections that need it
all_gam_results = all_gam_results.merge(
    products_lookup, on="product", how="left"
)
elasticity_df = elasticity_df.merge(
    products_lookup, on="product", how="left"
)[["product", "product_key", "ratio", "pct"]]

# 2) Build true best50 from P50 curve (now product_key exists)
if "revenue_pred_0.5" in all_gam_results.columns:
    idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
    best50_optimal_pricing = (
        all_gam_results.loc[idx, ["product", "product_key", "asp", "revenue_pred_0.5"]]
        .reset_index(drop=True)
    )
else:
    # Fallback: derive from best_optimal_pricing (avg table), then merge to get P50 if present
    best50_optimal_pricing = (
        best_optimal_pricing[["product", "product_key", "asp"]]
        .merge(
            all_gam_results[["product", "asp", "revenue_pred_0.5"]],
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

# STRICT validation layout — register all pages/components that appear via routing
app.validation_layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        Homepage(),
        overview.layout(products_lookup),  # pass lookup df
        opps.layout(curr_opt_df),
        faq.faq_section(),
        html.Div(id="page-content"),
    ]
)

# Main layout with router area
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        dcc.Loading(
            id="router-loader",
            type="circle",
            children=html.Div(id="page-content", style={"minHeight": "65vh", "padding": "12px"}),
        ),
    ]
)

# ---------- Router ----------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(path):
    if path in ["/", "", None]:
        return Homepage()
    elif path == "/overview":
        return overview.layout(products_lookup)
    elif path == "/faq":
        return faq.faq_section()
    elif path == "/opps":
        return opps.layout(curr_opt_df)
    return html.Div("404 - Not found", className="p-4")

# ---------- Register callbacks ----------
# Signature: (app, price_quant_df, best_optimal_pricing_df, curr_price_df, elasticity_df, all_gam_results, products_lookup, viz_cls)
overview.register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing,  # <- use best50 table with revenue_pred_0.5
    curr_price_df,
    elasticity_df,
    all_gam_results,
    products_lookup,
    Viz,   
)

opps.register_callbacks(app, curr_opt_df)

print("\n", "-" * 10, datetime.now().strftime("%H:%M:%S"), " Page Updated " + "-" * 10, "\n")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

"""
ref:
https://dash.plotly.com/tutorial
https://medium.com/@wolfganghuang/advanced-dashboards-with-plotly-dash-things-to-consider-before-you-start-9754ac91fd10

# in terminal:
# ----------- dev -----------
cd dash
# windows: ..\.venv\Scripts\activate
# mac:     source ../pricing-venv/bin/activate

# windows:  py app.py
# mac:      python3 app.py

"""