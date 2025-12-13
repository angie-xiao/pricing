import os
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# Local Modules
from built_in_logic import viz as Viz
from built_in_logic import PricingPipeline
from helpers import Paths
from navbar import get_navbar
from home import Homepage
import overview, opps, faq

# Note: Removed 'Cache' import. We only use DataEng for the lookup helper.
from helpers import DataEng

import warnings

warnings.filterwarnings("ignore")

# ========================================================================
#                         DATA LOADING
# ========================================================================

# 1. Define Path (Points to .../pricing/data based on your screenshot)
DATA_PATH = os.path.join(Paths.PROJECT_BASE, "data")
print(f"📂 Loading data from: {DATA_PATH}")

# 2. Instantiate & Load
#    We start with empty None values, then load from folder
pipeline = PricingPipeline(pricing_df=None, product_df=None)
pipeline.load_from_folder(DATA_PATH)

# 3. Run Logic Once (Global State)
#    This generates all your dashboards frames (topsellers, best50, etc.)
GLOBAL_DATA = pipeline.assemble_dashboard_frames()

# 4. Set Defaults
DEFAULT_PRODUCT_LINE = "Standard"
available_product_lines = [DEFAULT_PRODUCT_LINE]

print("✓ Pipeline run complete. Data ready.")

# ========================================================================
#                    HELPER FUNCTIONS (Simplified)
# ========================================================================


def prepare_product_line_data(product_line: str):
    """
    Since we only have one dataset loaded, we just return the global data.
    The 'product_line' argument is kept for compatibility but ignored.
    """
    current_frames = GLOBAL_DATA

    # Safety Check
    if current_frames.get("all_gam_results") is None:
        return current_frames, None, None

    # Extract Best 50 table for the dropdowns
    best50_optimal_pricing = current_frames.get("best_optimal_pricing_df")

    # Build Dropdown Lookup
    dropdown_lookup = (
        best50_optimal_pricing[["product", "asin", "revenue_pred_0.5"]]
        .sort_values(by="revenue_pred_0.5", ascending=False)
        .dropna(subset=["asin"])
        .astype({"asin": str})
        .reset_index(drop=True)
    )

    return current_frames, best50_optimal_pricing, dropdown_lookup


# ========================================================================
#                    DASH APP SETUP
# ========================================================================

# Initialize Data
current_frames, best50_optimal_pricing, dropdown_lookup = prepare_product_line_data(
    DEFAULT_PRODUCT_LINE
)

# Unpack frames for specific pages
price_quant_df = current_frames["price_quant_df"]
best_avg_df = current_frames["best_avg"]
all_gam_results = current_frames["all_gam_results"]
curr_price_df = current_frames["curr_price_df"]
elasticity_df = current_frames["elasticity_df"]
curr_opt_df = current_frames["curr_opt_df"]
meta = current_frames["meta"]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
)
server = app.server


# Validation layout (for callback validation)
app.validation_layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="selected-product-line", data=DEFAULT_PRODUCT_LINE),
        dcc.Store(id="product-line-data", data={}),
        get_navbar(),
        Homepage(),
        overview.layout(dropdown_lookup),
        opps.layout(curr_opt_df),
        faq.faq_section(),
        html.Div(id="page-content"),
    ]
)

# Main app layout
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="selected-product-line", data=DEFAULT_PRODUCT_LINE),
        dcc.Store(id="product-line-data", data={}),
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


# ========================================================================
#                         CALLBACKS
# ========================================================================


@app.callback(
    [Output("selected-product-line", "data"), Output("product-line-data", "data")],
    Input("product-line-selector", "value"),
    prevent_initial_call=False,
)
def update_product_line_selection(selected_line):
    """
    Update the selected product line and prepare its data.

    This callback is triggered when user selects a different product line
    from the dropdown. It loads the corresponding data and stores it.
    """
    if not selected_line:
        selected_line = DEFAULT_PRODUCT_LINE

    print(f"→ Switching to product line: {selected_line}")

    # Prepare data for selected product line
    line_frames, best50, dropdown = prepare_product_line_data(selected_line)

    # Store essential data for other callbacks
    product_data = {"product_line": selected_line, "has_data": best50 is not None}

    return selected_line, product_data


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"), Input("selected-product-line", "data")],
)
def route(path, selected_product_line):
    """
    Route to different pages based on URL path.

    Args:
        path: URL pathname
        selected_product_line: Currently selected product line

    Returns:
        Page layout component
    """
    # Use selected product line or default
    product_line = selected_product_line or DEFAULT_PRODUCT_LINE

    # Prepare data for the selected product line
    line_frames, best50, dropdown = prepare_product_line_data(product_line)

    # Get curr_opt_df for opportunities page
    curr_opt = line_frames.get("curr_opt_df")

    # Route to appropriate page
    if path in ["/", "", None]:
        return Homepage()
    elif path == "/overview":
        return (
            overview.layout(dropdown)
            if dropdown is not None
            else html.Div("No data available")
        )
    elif path == "/faq":
        return faq.faq_section()
    elif path == "/opps":
        return (
            opps.layout(curr_opt)
            if curr_opt is not None
            else html.Div("No data available")
        )

    return html.Div("404 - Not found", className="p-4")


# ========================================================================
#                    REGISTER PAGE CALLBACKS
# ========================================================================

# Register overview page callbacks
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

# Register opportunities page callbacks
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


# ========================================================================
#                         RUN APP
# ========================================================================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


""" 
---------- Instructions to run app.py ----------
# always pull from Github first 
git pull origin master         # mac
git pull origin master         # windows

# in terminal:
cd dash


# ------------- mac -------------
# install pyenv
pyenv install 3.11.7
pyenv global 3.11.7
python -m venv .pricing-venv
source ../.venv/bin/activate              # activate
python3 app.py                            # run

# ------------- windows -------------
python -m venv .venv                      # create venv (run once)
../.venv/Scripts/activate                 # activate
py app.py                                 # run

# ------- optional:  formatter -------
python3 -m black helpers.py

"""
