# app.py
import os
import pickle
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
import inspect

import pandas as pd
import numpy as np

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# local modules
import built_in_logic  # IMPORTANT: importing this is part of code signature
from built_in_logic import PricingPipeline, viz as Viz
from navbar import get_navbar
from home import Homepage
import overview, opps, faq

warnings.filterwarnings("ignore")

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))   # .../pricing/dash
PROJECT_BASE = os.path.dirname(BASE_DIR)                 # .../pricing


# ---------- cache helpers ----------
def _files_sig(paths, top_n=10, version="v1"):
    """
    Build a stable signature from file metadata + an extra 'version' string.
    Pass ABSOLUTE paths. Uses mtime_ns and size so it’s fast even for big CSVs.
    """
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
    """
    Hash the *current* modeling code. If you edit built_in_logic.py,
    this changes immediately and busts the cache on next run.
    """
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

    # include code signature so model edits rebuild frames
    sig = _files_sig([pricing_path, product_path], top_n=top_n, version=_code_sig())

    # Dev kill-switch
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

    # coalesce suffixes if they exist from earlier merges
    for c in ("product_key_x", "product_key_y"):
        if c in df.columns:
            if "product_key" not in df.columns:
                df["product_key"] = df[c]
            else:
                df["product_key"] = df["product_key"].fillna(df[c])

    # drop any residue
    df = df.drop(
        columns=[c for c in ("product_key_x", "product_key_y") if c in df.columns],
        errors="ignore",
    )

    # normalize dtype
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

# --- Build a robust lookup, using whatever already has both product + product_key
products_lookup = make_products_lookup(best_optimal_pricing, best_avg_df, curr_price_df, all_gam_results)

# --- Ensure 'product_key' ONCE (no double-merge later) ---
all_gam_results      = ensure_product_key(all_gam_results,      products_lookup)
elasticity_df        = ensure_product_key(elasticity_df,        products_lookup)
best_optimal_pricing = ensure_product_key(best_optimal_pricing, products_lookup)
best_avg_df          = ensure_product_key(best_avg_df,          products_lookup)
curr_opt_df          = ensure_product_key(curr_opt_df,          products_lookup)
curr_price_df        = ensure_product_key(curr_price_df,        products_lookup)

# Keep only the elasticity columns we need
if {"product", "product_key", "ratio", "pct"}.issubset(elasticity_df.columns):
    elasticity_df = elasticity_df[["product", "product_key", "ratio", "pct"]].copy()
else:
    # be defensive if pipeline changed
    needed = ["product", "product_key"]
    for c in ["ratio", "pct"]:
        if c in elasticity_df.columns:
            needed.append(c)
    elasticity_df = elasticity_df[needed].copy()

# --- Build true best-50 table directly from P50 curve (has product_key) ---
if "revenue_pred_0.5" in all_gam_results.columns:
    idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
    best50_optimal_pricing = (
        all_gam_results.loc[idx, ["product", "product_key", "asp", "revenue_pred_0.5"]]
        .drop_duplicates(subset=["product"])
        .reset_index(drop=True)
    )
else:
    # Fallback: merge from avg table to attach P50 revenue if present
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

# --- Debug fingerprints so you can see the model actually changed ---
# print(
#     "[frames]",
#     f"all_gam_results rows={len(all_gam_results)}",
#     f"prods={all_gam_results['product'].nunique()}",
#     f"has_p50={'pred_0.5' in all_gam_results.columns}",
# )
# print("[frames] all_gam_results cols:", sorted(all_gam_results.columns))
# print("[frames] best50 cols:", best50_optimal_pricing.columns.tolist())
# print("[frames] elasticity_df cols:", sorted(elasticity_df.columns))

# ---------- Dash app ----------
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
        return overview.layout(products_lookup)
    elif path == "/faq":
        return faq.faq_section()
    elif path == "/opps":
        return opps.layout(curr_opt_df)
    return html.Div("404 - Not found", className="p-4")


# ---------- Register callbacks ----------
# overview.register_callbacks signature:
# (app, price_quant_df, best_optimal_pricing_df, curr_price_df, elasticity_df, all_gam_results, products_lookup, viz_cls)
overview.register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing,  # use P50 table
    curr_price_df,
    elasticity_df,
    all_gam_results,
    products_lookup,
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
    curr_opt_df,  # table data
)

print(
    "\n",
    "-" * 10,
    datetime.now().strftime("%H:%M:%S"),
    " Page Updated " + "-" * 10,
    "\n",
)

if __name__ == "__main__":
    # Dev: force rebuild with
    #   PRICING_FORCE_REBUILD=1 python app.py
    app.run(debug=True)
