# app.py
import os, pickle, hashlib, inspect
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# local modules
import built_in_logic
from built_in_logic import PricingPipeline, viz as Viz
from navbar import get_navbar
from home import Homepage
import overview, opps, faq

# ---------- paths ----------
BASE_DIR     = os.path.dirname(os.path.realpath(__file__))   # .../pricing/dash
PROJECT_BASE = os.path.dirname(BASE_DIR)                     # .../pricing

# ---------- cache helpers ----------
def _files_sig(paths, top_n=10, version="v1"):
    """Hash file metadata + 'version' string (fast even for big CSVs)."""
    parts = []
    for p in paths:
        ap = os.path.abspath(p)
        try:
            st = os.stat(ap)
            parts.append(f"{ap}:{st.st_mtime_ns}:{st.st_size}")
        except FileNotFoundError:
            parts.append(f"{ap}:NA")
    sig_str = "|".join(parts) + f":top{top_n}:ver:{version}"
    return hashlib.sha1(sig_str.encode()).hexdigest()

def _code_sig():
    """Hash current modeling code to bust cache whenever built_in_logic.py changes."""
    try:
        src = inspect.getsource(built_in_logic)
        return hashlib.sha1(src.encode()).hexdigest()
    except Exception:
        try:
            p = Path(built_in_logic.__file__)
            return hashlib.sha1(p.read_bytes()).hexdigest()
        except Exception:
            return "nocode"

def build_frames_with_cache(
    base_dir,
    data_folder="data",
    pricing_file="730d.csv",
    product_file="products.csv",
    top_n=10,
):
    cache_dir = os.path.join(base_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    pricing_path = os.path.join(base_dir, data_folder, pricing_file)
    product_path = os.path.join(base_dir, data_folder, product_file)

    sig = _files_sig([pricing_path, product_path], top_n=top_n, version=_code_sig())
    force = os.environ.get("PRICING_FORCE_REBUILD") == "1"
    cache_fp = None if force else os.path.join(cache_dir, f"frames_{sig}.pkl")

    if cache_fp and os.path.exists(cache_fp):
        with open(cache_fp, "rb") as f:
            return pickle.load(f)

    frames = PricingPipeline.from_csv_folder(
        base_dir, data_folder, pricing_file, product_file, top_n
    )

    if cache_fp:
        with open(cache_fp, "wb") as f:
            pickle.dump(frames, f)

    return frames

# ---------- key normalization ----------
def make_products_lookup(*dfs):
    """
    Build a robust product ↔ key lookup from any frames that already contain both.
    """
    pieces = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        cols = set(df.columns)
        if {"product", "product_key"}.issubset(cols):
            pieces.append(df[["product", "product_key"]].copy())
    if not pieces:
        raise KeyError("No source frame had both ['product','product_key'].")
    lookup = (
        pd.concat(pieces, ignore_index=True)
        .dropna(subset=["product", "product_key"])
        .astype({"product_key": str})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return lookup

def ensure_product_key(df, lookup):
    """
    Ensure a single 'product_key' column (str) exists by merging-once to lookup and
    coalescing any suffixes from prior merges.
    """
    if df is None or len(df) == 0:
        return df
    if "product_key" not in df.columns:
        df = df.merge(lookup, on="product", how="left")
    for c in ("product_key_x", "product_key_y"):
        if c in df.columns:
            if "product_key" not in df.columns:
                df["product_key"] = df[c]
            else:
                df["product_key"] = df["product_key"].fillna(df[c])
    df = df.drop(columns=[c for c in ("product_key_x", "product_key_y") if c in df.columns], errors="ignore")
    if "product_key" in df.columns:
        df["product_key"] = df["product_key"].astype(str)
    return df

# ---------- Load data (cached) ----------
frames = build_frames_with_cache(PROJECT_BASE)
price_quant_df       = frames["price_quant_df"]
best_avg_df          = frames["best_avg"]
all_gam_results      = frames["all_gam_results"]
best_optimal_pricing = frames["best_optimal_pricing_df"]
elasticity_df        = frames["elasticity_df"]
curr_opt_df          = frames["curr_opt_df"]
curr_price_df        = frames["curr_price_df"]
opps_summary         = frames["opps_summary"]
meta                 = frames["meta"]  # {data_start, data_end, days_covered, annual_factor}

# --- Build lookups ---
# Full lookup for merges (keeps keys intact)
lookup_all = make_products_lookup(best_optimal_pricing, best_avg_df, curr_price_df, all_gam_results)

# Top-N only lookup for the Overview dropdown (best_optimal_pricing is Top-N by construction)
dropdown_lookup = (
    best_optimal_pricing[["product", "product_key"]]
    .dropna(subset=["product_key"])
    .drop_duplicates(subset=["product"])
    .astype({"product_key": str})
    .reset_index(drop=True)
)

# --- Ensure 'product_key' ONCE (using full lookup) ---
all_gam_results      = ensure_product_key(all_gam_results,      lookup_all)
elasticity_df        = ensure_product_key(elasticity_df,        lookup_all)
best_optimal_pricing = ensure_product_key(best_optimal_pricing, lookup_all)
best_avg_df          = ensure_product_key(best_avg_df,          lookup_all)
curr_opt_df          = ensure_product_key(curr_opt_df,          lookup_all)
curr_price_df        = ensure_product_key(curr_price_df,        lookup_all)

# Build best50 from P50 curve (guarantee product_key present)
if "revenue_pred_0.5" in all_gam_results.columns:
    idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
    best50_optimal_pricing = (
        all_gam_results.loc[idx, ["product", "asp", "revenue_pred_0.5", "pred_0.5"]]
        .reset_index(drop=True)
        .merge(dropdown_lookup, on="product", how="left")
        [["product", "product_key", "asp", "revenue_pred_0.5", "pred_0.5"]]
    )
else:
    best50_optimal_pricing = (
        best_optimal_pricing[["product", "product_key", "asp"]]
        .merge(all_gam_results[["product", "asp", "revenue_pred_0.5", "pred_0.5"]],
               on=["product", "asp"], how="left")
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

# STRICT validation layout — include all routes’ components
app.validation_layout = html.Div(
    [
        dcc.Location(id="url"),
        get_navbar(),
        Homepage(),
        overview.layout(dropdown_lookup),  # pass Top-N only
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
        return overview.layout(dropdown_lookup)  # Top-N only
    elif path == "/faq":
        return faq.faq_section()
    elif path == "/opps":
        return opps.layout(curr_opt_df)
    return html.Div("404 - Not found", className="p-4")

# ---------- Register callbacks ----------
# Signature: (app, price_quant_df, best50_optimal_pricing_df, curr_price_df, elasticity_df,
#             all_gam_results, products_lookup, meta, viz_cls)
overview.register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing,
    curr_price_df,
    elasticity_df,
    all_gam_results,
    dropdown_lookup,   # Top-N mapping used by dropdown + titles
    meta,
    Viz,
)

# opps page (unchanged)
opps.register_callbacks(
    app,
    {
        "elasticity_df": elasticity_df,
        "best50_optimal_pricing_df": best50_optimal_pricing,
        "curr_price_df": curr_price_df,
        "all_gam_results": all_gam_results,
    },
    curr_opt_df,
)

print("\n", "-" * 10, datetime.now().strftime("%H:%M:%S"), " Page Updated " + "-" * 10, "\n")

if __name__ == "__main__":
    app.run(debug=True)
