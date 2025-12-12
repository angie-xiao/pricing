
import os
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

FORCE_REBUILD = os.getenv("OPTIMA_FORCE_REBUILD", "0") in ("1", "true", "True")

# ========================================================================
#                         DATA LOADING
# ========================================================================

BASE_DIR = Paths.BASE_DIR
PROJECT_BASE = Paths.PROJECT_BASE

# Load all product lines at once using pattern matching
frames = Cache.build_frames_with_cache(
    base_dir=".",
    data_folder="data",
    pricing_pattern="_pricing",  # Matches files with _pricing
    product_pattern="_product",   # Matches files with _product or _products
    top_n=10,
    force_rebuild=False
)
# Discover available product lines from loaded frames
available_product_lines = sorted(set(k.split('_')[0] for k in frames.keys() if '_' in k))
print(f"✓ Available product lines: {available_product_lines}")

# Set default product line
DEFAULT_PRODUCT_LINE = available_product_lines[0] if available_product_lines else "boxie"
print(f"✓ Default product line: {DEFAULT_PRODUCT_LINE}")


# ========================================================================
#                    HELPER FUNCTIONS
# ========================================================================

def get_frames_for_product_line(product_line: str):
    """
    Extract all dataframes for a specific product line.
    
    Args:
        product_line: Product line identifier (e.g., 'boxie', 'chd')
    
    Returns:
        Dictionary of dataframes for the specified product line
    """
    return {
        "price_quant_df": frames.get(f"{product_line}_price_quant_df"),
        "best_avg": frames.get(f"{product_line}_best_avg"),
        "all_gam_results": frames.get(f"{product_line}_all_gam_results"),
        "best_optimal_pricing_df": frames.get(f"{product_line}_best_optimal_pricing_df"),
        "elasticity_df": frames.get(f"{product_line}_elasticity_df"),
        "curr_opt_df": frames.get(f"{product_line}_curr_opt_df"),
        "curr_price_df": frames.get(f"{product_line}_curr_price_df"),
        "opps_summary": frames.get(f"{product_line}_opps_summary"),
        "meta": frames.get(f"{product_line}_meta"),
        "best_weighted": frames.get(f"{product_line}_best_weighted"),
    }


def prepare_product_line_data(product_line: str):
    """
    Prepare all necessary dataframes and lookups for a product line.
    
    Args:
        product_line: Product line identifier
    
    Returns:
        Tuple of (current_frames, best50_optimal_pricing, dropdown_lookup)
    """
    # Get frames for this product line
    current_frames = get_frames_for_product_line(product_line)
    
    # Extract individual dataframes
    all_gam_results = current_frames["all_gam_results"]
    
    if all_gam_results is None or all_gam_results.empty:
        print(f"⚠ Warning: No GAM results for {product_line}")
        return current_frames, None, None
    
    # Create best50_optimal_pricing
    idx = all_gam_results.groupby("product")["revenue_pred_0.5"].idxmax()
    best50_optimal_pricing = (
        all_gam_results.loc[
            idx, ["product", "asin", "asp", "revenue_pred_0.5", "units_pred_0.5"]
        ]
        .reset_index(drop=True)
        .drop_duplicates(subset=["product"])
    )
    
    # Create dropdown_lookup
    dropdown_lookup = (
        best50_optimal_pricing[["product", "asin", "revenue_pred_0.5"]]
        .sort_values(by="revenue_pred_0.5", ascending=False)
        .dropna(subset=["asin"])
        .astype({"asin": str})
        .reset_index(drop=True)
    )
    
    return current_frames, best50_optimal_pricing, dropdown_lookup


# ========================================================================
#                    INITIALIZE DEFAULT PRODUCT LINE
# ========================================================================

# Load default product line data
current_frames, best50_optimal_pricing, dropdown_lookup = prepare_product_line_data(DEFAULT_PRODUCT_LINE)

# Extract individual dataframes for backward compatibility
price_quant_df = current_frames["price_quant_df"]
best_avg_df = current_frames["best_avg"]
all_gam_results = current_frames["all_gam_results"]
best_optimal_pricing = current_frames["best_optimal_pricing_df"]
elasticity_df = current_frames["elasticity_df"]
curr_opt_df = current_frames["curr_opt_df"]
curr_price_df = current_frames["curr_price_df"]
opps_summary = current_frames["opps_summary"]
meta = current_frames["meta"]
best_weighted_df = current_frames["best_weighted"]

# Build lookups
lookup_all = DataEng.make_products_lookup(
    best_optimal_pricing, best_avg_df, curr_price_df, all_gam_results
)


# ========================================================================
#                         DASH APP SETUP
# ========================================================================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
)
server = app.server

# Product line selector component
product_line_selector = html.Div(
    [
        html.Label(
            "Product Line:",
            style={
                'display': 'inline-block',
                'marginRight': '10px',
                'fontWeight': '600',
                'color': '#495057'
            }
        ),
        dcc.Dropdown(
            id='product-line-selector',
            options=[
                {'label': line.upper(), 'value': line}
                for line in available_product_lines
            ],
            value=DEFAULT_PRODUCT_LINE,
            clearable=False,
            style={'width': '200px', 'display': 'inline-block'}
        ),
    ],
    style={
        'padding': '12px 20px',
        'backgroundColor': '#f8f9fa',
        'borderBottom': '1px solid #dee2e6',
        'display': 'flex',
        'alignItems': 'center'
    }
)

# Validation layout (for callback validation)
app.validation_layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id='selected-product-line', data=DEFAULT_PRODUCT_LINE),
    dcc.Store(id='product-line-data', data={}),
    get_navbar(),
    product_line_selector,
    Homepage(),
    overview.layout(dropdown_lookup),
    opps.layout(curr_opt_df),
    faq.faq_section(),
    html.Div(id="page-content"),
])

# Main app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id='selected-product-line', data=DEFAULT_PRODUCT_LINE),
    dcc.Store(id='product-line-data', data={}),
    get_navbar(),
    product_line_selector,
    dcc.Loading(
        id="router-loader",
        type="circle",
        children=html.Div(
            id="page-content",
            style={"minHeight": "65vh", "padding": "12px"}
        ),
    ),
])


# ========================================================================
#                         CALLBACKS
# ========================================================================

@app.callback(
    [
        Output('selected-product-line', 'data'),
        Output('product-line-data', 'data')
    ],
    Input('product-line-selector', 'value'),
    prevent_initial_call=False
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
    product_data = {
        'product_line': selected_line,
        'has_data': best50 is not None
    }
    
    return selected_line, product_data


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"), Input('selected-product-line', 'data')]
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
        return overview.layout(dropdown) if dropdown is not None else html.Div("No data available")
    elif path == "/faq":
        return faq.faq_section()
    elif path == "/opps":
        return opps.layout(curr_opt) if curr_opt is not None else html.Div("No data available")
    
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
    print(" " + "="*70)
    print(f"  🚀 Starting Pricing Optimization Dashboard")
    print(f"  📊 Product Lines: {', '.join(available_product_lines)}")
    print(f"  🎯 Default: {DEFAULT_PRODUCT_LINE}")
    print("="*70 + "")
    
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
