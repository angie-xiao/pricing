# overview.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

ACCENT = {"color": "#DAA520"}

def layout(products_lookup: pd.DataFrame):
    """Pass a Top-N-only DataFrame with columns ['asin','product'].""" 
    return dbc.Container(
        [
            html.H1(
                "Overview",
                className="display-5",
                style={"textAlign": "center", "padding": "58px 0 8px"},
            ),

            # Product selector (Top-N only)
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            "Select a Product:",
                            style={"fontWeight": 600, "textAlign": "right", "marginRight": "10px"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="product_dropdown_snap",
                            options=[
                                {"label": r["product"], "value": r["asin"]}
                                for _, r in products_lookup.iterrows()
                            ],
                            value=(
                                products_lookup["asin"].iloc[0]
                                if len(products_lookup)
                                else None
                            ),
                            style={"width": "300px"},
                            clearable=False,
                        ),
                        width="auto",
                    ),
                ],
                justify="center",
                align="center",
                style={"padding": "10px 0 20px"},
            ),

            # KPI row (exact order requested)
            dbc.Row(
                [
                    _kpi_card("card_title_date_range", "Number of Days", "date_range_value", bg="#eef2fa", id_subtext="date-range-subtext"),
                    _kpi_card("card_title_curr_price_snap", "Current Price", "curr_price_snap", bg="#eef2fa"),
                    _kpi_card("card_title_snap", "Recommended Price", "card_asp_snap", bg="#F5E8D8"),
                    _kpi_card("card_title_elasticity_snap", "Elasticity", "elasticity_ratio_snap", bg="#eef2fa", id_subtext="elasticity-subtext"),
                    _kpi_card("card_title_units_opp_ann", "Annualized Units Sold Opportunity", "units_opp_ann_value", bg="#eef8f0"),
                    _kpi_card("card_title_rev_best_ann", "Annualized Revenue Opportunity", "rev_best_ann_value", bg="#eef8f0"),
                    _kpi_card("card_title_fit_snap", "Model Fit (Daily Revenue)", "fit_value_snap", bg="#eef2fa", id_subtext="fit-subtext"),
                ],
                className="g-4 align-items-stretch",
                justify="center",
                align="center",
                style={"padding": "10px 0 10px"},
            ),
            html.Br(),
            html.Hr(className="my-4", style={"padding":"20px",}),

            # Predictive graph + scenario table + explainer
            dbc.Row(
                [   
                    # title
                    html.H3(
                        "Predictive Graph",
                        className="mt-3",
                        style={"margin-left": "50px", "marginTop": "190px", "color": "#DAA520"},
                    ),
                    # graph and coverage note
                    dbc.Col(
                        [
                            dcc.Loading(
                                type="circle",
                                children=dcc.Graph(
                                    id="gam_results_pred",
                                    style={"height": "560px"},
                                    config={"displaylogo": False},
                                ),
                            ),
                        ],
                        md=8, xs=12, className="mb-0",  # Changed mb-3 to mb-0
                    ),
                    
                    # scenario table and reading notes
                    dbc.Col(
                        [
                            html.H6("Scenario Summary", className="mb-2", style={"textAlign": "center", "marginTop": "40px"}),
                            dash_table.DataTable(
                                id="scenario_table",
                                columns=[
                                    {"name": "Case", "id": "case"},
                                    {"name": "Price", "id": "price"},
                                    {"name": "Revenue", "id": "revenue"},
                                ],
                                data=[],
                                style_table={"border": "none", "marginBottom": "12px"},
                                style_cell={"textAlign": "center", "border": "none", "fontSize": "14px", "padding": "6px"},
                                style_header={"fontWeight": 600, "border": "none", "backgroundColor": "#f6f6f6"},
                            ),
                            dbc.Card(
                                dbc.CardBody([
                                    html.H6("How to read this", className="mb-2 fw-bold text-uppercase"),
                                    html.Ul(
                                        [
                                            html.Li([html.B("Goal: "), "Pick the price where the central curve's expected revenue is highest."]),
                                            html.Li([html.B("Range: "), "Conservative and optimistic curves show how outcomes may vary."]),
                                            html.Li([html.B("Confidence: "), "Stronger when all curves peak around a similar price and there's lots of nearby data."]),
                                        ],
                                        className="predictive-explainer-list",
                                    ),
                                ]),
                                className="shadow-sm",
                                style={"marginTop": "70px"},
                            ),
                        ],
                        md=4, xs=12,
                    ),
                ],
                className="g-3 mb-2",  # Reduced bottom margin
            ),

            # Line and coverage note (moved outside the main row)
            # html.Hr(className="mt-0 mb-2"),  # Added explicit margin classes
            html.Div(
                html.Span(html.I(id="coverage_note")),
                style={
                    "textAlign": "center",
                    "color": "#5f6b7a",
                    "fontSize": "0.9em",
                    "marginBottom": "50px",  # Reduced from default
                }
            ),

            # Footer
            html.Div(
                [html.Span("made with ♥️ | "), html.Span(html.I("@aqxiao")), html.P("github.com/angie-xiao")],
                className="text-center py-3",
                style={
                    "fontSize": "0.8em",
                    "color": "#ac274f",
                    "textAlign": "center",
                    "backgroundColor": "#f3f3f3",
                    "margin": "20px auto 0 auto",  # Reduced top margin
                    "borderRadius": "6px"
                },
            ),
        ],
        fluid=True,
    )


def _kpi_card(id_title, title, id_value, bg="#f3f0f0", id_subtext=None):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        id=id_title, className="kpi-title",
                        style={
                            "color": "#121212", "textAlign": "center",
                            "marginBottom": "10px", "marginTop": "10px",
                            "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
                        },
                    ),
                    html.Div(
                        style={
                            "height": "4px", "width": "44px", "margin": "2px auto 8px",
                            "borderRadius": "999px", "background": "linear-gradient(90deg,#DAA520,#F0C64C)",
                            "opacity": 0.9,
                        }
                    ),
                    html.H2(
                        title, className="kpi-eyebrow",
                        style={"color": "#121212", "textAlign": "center", "fontSize": "18px", "letterSpacing": ".14em", "fontWeight": 700},
                    ),
                    html.H1(
                        id=id_value, className="kpi-value",
                        style={"color": "#DAA520", "textAlign": "center", "fontSize": "44px", "fontWeight": 800},
                    ),
                    html.Div(
                        id=id_subtext if id_subtext else f"{id_value}-subtext",
                        className="kpi-subtext text-muted",
                        style={"textAlign": "center", "fontSize": "15px", "marginTop": "6px", "lineHeight": "1.25", "minHeight": "34px"},
                    ),
                ],
                className="d-flex flex-column justify-content-start",
                style={"gap": "6px", "padding": "22px 18px"},
            ),
            className="h-100 shadow-sm border rounded-4",
            style={
                "backgroundColor": bg, "padding": "12px 0",
                "borderColor": "rgba(17,24,39,0.08)",
                "boxShadow": "0 1px 2px rgba(16,24,40,.03), 0 4px 8px rgba(16,24,40,.04)",
                "backgroundImage": "radial-gradient(160% 80% at 50% 0%, rgba(218,165,32,.03), transparent 45%)",
            },
        ),
        width=3,
        className="kpi-card",
    )

# ---------- helpers (formatting & metrics) ----------
def _format_money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"

def _format_units(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "—"

def _format_signed_units(x):
    if x is None or not np.isfinite(float(x)):
        return "—"
    x = float(x)
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):,.0f}"

def _format_signed_money(x):
    if x is None or not np.isfinite(float(x)):
        return "—"
    x = float(x)
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.0f}"

def _format_date(dt):
    try:
        dt = pd.to_datetime(dt)
        if pd.isna(dt):
            return "—"
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "—"


def _model_fit_units(prod_df: pd.DataFrame):
    """
    Return (value_text, subtext) for model accuracy on daily revenue (P50).
    Shows typical error in dollars per day.
    """
    if prod_df is None or prod_df.empty:
        return "—", ""
    
    # Check for required columns
    need = {"daily_rev", "revenue_pred_0.5", "asp"}
    if not need.issubset(prod_df.columns):
        return "—", ""
        
    # Get clean data
    df = prod_df[["asp", "daily_rev", "revenue_pred_0.5"]].dropna()
    if df.empty:
        return "—", ""
        
    # Aggregate by price point for stable evaluation
    df = df.groupby("asp", as_index=False).agg(
        daily_rev=("daily_rev", "mean"),
        pred_rev=("revenue_pred_0.5", "mean")
    )
    
    y_true = df["daily_rev"].to_numpy(float)
    y_pred = df["pred_rev"].to_numpy(float)
    
    if y_true.size == 0:
        return "—", ""
        
    # Calculate RMSE in dollars
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate percentage error
    avg_rev = float(np.mean(y_true)) if y_true.size else np.nan
    pct_err = (rmse_val / avg_rev * 100.0) if avg_rev else np.nan
    
    # Format output with dollar sign since we're measuring revenue
    return f"±${rmse_val:,.0f}", (f"≈{pct_err:.1f}% typical error" if np.isfinite(pct_err) else "")


def _update_elasticity_kpi_by_product(product_name: str, elast_df: pd.DataFrame):
    try:
        row = elast_df.loc[elast_df["product"] == product_name]
        if row.empty or "ratio" not in row or "pct" not in row:
            return "—", ""
        ratio = float(row["ratio"].iloc[0])
        pct = float(row["pct"].iloc[0])
        value_text = f"{ratio:,.2f}"
        pct_round = int(round(pct))
        top_share = max(1, 100 - pct_round)
        subtext = ("Top ~{0}% most ELASTIC".format(top_share) if pct >= 50
                   else "Top ~{0}% most INELASTIC".format(top_share))
        return value_text, subtext
    except Exception:
        return "—", ""

def _scenario_table(prod_df: pd.DataFrame) -> pd.DataFrame:
    if prod_df.empty:
        return pd.DataFrame(columns=["case", "price", "revenue"])
    rows = []
    for col in ["revenue_pred_0.025", "revenue_pred_0.5", "revenue_pred_0.975"]:
        if col not in prod_df:
            continue
        row = prod_df.loc[prod_df[col] == prod_df[col].max()].head(1)
        if not row.empty:
            rows.append(
                {
                    "case": {
                        "revenue_pred_0.025": "Conservative",
                        "revenue_pred_0.5": "Expected",
                        "revenue_pred_0.975": "Optimistic",
                    }[col],
                    "price": f"${row['asp'].iloc[0]:,.2f}",
                    "revenue": f"${row[col].iloc[0]:,.0f}",
                }
            )
    return pd.DataFrame(rows)

def _annualized_kpis_signed(asin, best50_df, curr_price_df, all_gam, annual_factor):
    """
    Returns formatted strings:
      +Annualized Δ Units  and  +$Annualized Potential Revenue
    Based on daily revenue predictions.
    """
    try:
        # Get best price point predictions
        best = best50_df[best50_df["asin"] == asin]
        if best.empty:
            return "—", "—"
            
        # Get daily values at best price
        daily_units_best = float(best.get("pred_0.5", np.nan).iloc[0]) if "pred_0.5" in best else np.nan
        daily_rev_best = float(best.get("revenue_pred_0.5", np.nan).iloc[0]) if "revenue_pred_0.5" in best else np.nan

        # Get current price
        cp = curr_price_df.loc[curr_price_df["asin"] == asin, "current_price"]
        curr_price = float(cp.iloc[0]) if len(cp) else np.nan

        # Get predictions at current price
        prod = all_gam[
            (all_gam["asin"] == asin)
            & pd.notna(all_gam["asp"])
            & pd.notna(all_gam["pred_0.5"])
        ]
        
        if prod.empty:
            daily_units_diff = np.nan
        else:
            idx = (prod["asp"] - curr_price).abs().idxmin() if pd.notna(curr_price) else None
            if idx is not None:
                daily_units_curr = float(prod.loc[idx, "pred_0.5"])
                daily_units_diff = daily_units_best - daily_units_curr
            else:
                daily_units_diff = np.nan

        # Annualize the differences
        units_diff_annual = daily_units_diff * 365.0 if pd.notna(daily_units_diff) else np.nan
        rev_best_annual = daily_rev_best * 365.0 if pd.notna(daily_rev_best) else np.nan

        return _format_signed_units(units_diff_annual), _format_signed_money(rev_best_annual)
    except Exception as e:
        print(f"Error in annualized KPIs: {e}")
        return "—", "—"

def _robustness_badge(prod_df):
    if prod_df is None or prod_df.empty:
        return ""
    def _peak_asp(col):
        if col not in prod_df or prod_df[col].isna().all():
            return np.nan
        idx = prod_df[col].idxmax()
        try:
            return float(prod_df.loc[idx, "asp"])
        except Exception:
            return np.nan

    p_low  = _peak_asp("revenue_pred_0.025")
    p_mid  = _peak_asp("revenue_pred_0.5")
    p_high = _peak_asp("revenue_pred_0.975")

    if np.isnan([p_low, p_mid, p_high]).any() or (not p_mid):
        spread_score = 0.0
    else:
        align_spread = max(p_low, p_mid, p_high) - min(p_low, p_mid, p_high)
        spread_score = float(np.exp(-align_spread / (0.1 * p_mid)))

    elasticity_score = 0.0
    if "elasticity" in prod_df.columns and p_mid and not np.isnan(p_mid):
        try:
            row = prod_df.loc[(prod_df["asp"] == p_mid) & prod_df["elasticity"].notna()]
            if not row.empty:
                el_mid = float(row["elasticity"].iloc[0])
                el_min = float(prod_df["elasticity"].min())
                el_max = float(prod_df["elasticity"].max())
                if el_max > el_min:
                    elasticity_score = 1.0 - (el_mid - el_min) / (el_max - el_min)
                else:
                    elasticity_score = 1.0
        except Exception:
            pass

    try:
        n_distinct_prices = int(prod_df["asp"].nunique(dropna=True))
    except Exception:
        n_distinct_prices = prod_df.shape[0]
    data_strength = 1.0 - float(np.exp(-n_distinct_prices / 6.0))
    credibility_multiplier = 0.6 + 0.4 * data_strength

    base_score  = 0.4 * spread_score + 0.6 * elasticity_score
    final_score = base_score * credibility_multiplier

    label, color = ("Weak", "danger")
    if final_score >= 0.70:
        label, color = ("Strong", "success")
    elif final_score >= 0.45:
        label, color = ("Medium", "warning")

    return dbc.Badge(f"Confidence: {label}", color=color, pill=True, className="px-3 py-2")

def _coverage_note(prod_df):
    if prod_df is None or prod_df.empty:
        return ""
    n_points = int(len(prod_df))
    if {"revenue_actual", "revenue_pred_0.025", "revenue_pred_0.975"}.issubset(prod_df.columns):
        within = (
            (prod_df["revenue_actual"] >= prod_df["revenue_pred_0.025"]) &
            (prod_df["revenue_actual"] <= prod_df["revenue_pred_0.975"])
        ).mean()
    else:
        within = np.nan
    return html.Div(
        [
            html.Div(
                [
                    "Based on ",
                    html.Span(f"{n_points}", style={"color": ACCENT["color"], "fontWeight": 600}),
                    " historical points; ",
                    html.Span(f"{within*100:,.0f}%" if np.isfinite(within) else "—", style={"color": ACCENT["color"], "fontWeight": 600}),
                    " of actual revenue outcomes fall within the shown range.",
                ],
                style={"textAlign": "center"},
            ),
        ],
        style={"marginTop": "8px"},
    )

# ---------- callbacks ----------
def register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing_df,
    curr_price_df,
    elasticity_df,
    all_gam_results,
    products_lookup,
    meta,
    viz_cls,
):
    viz = viz_cls()
    
    @app.callback(
        # Updated KPI cards outputs
        Output("card_title_date_range", "children"),
        Output("date_range_value", "children"),
        Output("date-range-subtext", "children"),
        Output("card_title_curr_price_snap", "children"),
        Output("curr_price_snap", "children"),
        Output("card_title_snap", "children"),
        Output("card_asp_snap", "children"),
        Output("card_title_elasticity_snap", "children"),
        Output("elasticity_ratio_snap", "children"),
        Output("elasticity-subtext", "children"),
        Output("card_title_units_opp_ann", "children"),
        Output("units_opp_ann_value", "children"),
        Output("card_title_rev_best_ann", "children"),
        Output("rev_best_ann_value", "children"),
        Output("card_title_fit_snap", "children"),
        Output("fit_value_snap", "children"),
        Output("fit-subtext", "children"),
        # rest of the page
        Output("gam_results_pred", "figure"),
        Output("scenario_table", "data"),
        Output("coverage_note", "children"),
        Input("product_dropdown_snap", "value"),
    )
    
    def overview(asin):
        # Empty selection -> placeholders
        if not asin:
            empty_fig = viz.empty_fig("Select a product")
            return (
                "", "—", "", "", "—", "", "—", "", "—", "",
                "", "—", "", "—", "", "—", "",
                empty_fig, [], ""
            )

        # Resolve display name (optional; we keep titles static)
        try:
            display_name = products_lookup.loc[
                products_lookup["asin"] == asin, "product"
            ].iloc[0]
        except Exception:
            display_name = ""

        # Current price
        cp = curr_price_df.loc[curr_price_df["asin"] == asin, "current_price"]
        curr_price_val = f"${float(cp.iloc[0]):,.2f}" if len(cp) else "—"

        # Recommended price (P50 best)
        best_row = best50_optimal_pricing_df.loc[
            best50_optimal_pricing_df["asin"] == asin
        ]
        asp_val = f"${float(best_row['asp'].iloc[0]):,.2f}" if len(best_row) else "—"

        # Elasticity KPI (lookup by product name)
        elast_val, elast_subtext = _update_elasticity_kpi_by_product(display_name, elasticity_df)

        # Product slice for plots/KPIs
        filt = all_gam_results[all_gam_results["asin"] == asin]
        pred_graph = viz.gam_results(filt) if len(filt) else viz.empty_fig("No model data")

        # Scenario summary
        scenario_df = _scenario_table(filt) if len(filt) else pd.DataFrame([{"case": "—", "price": "—", "revenue": "—"}])
        scenario_data = scenario_df.to_dict("records")

        # Annualized KPIs (signed, with + prefixes where applicable)
        du_ann, rev_best_ann = _annualized_kpis_signed(
            asin, best50_optimal_pricing_df, curr_price_df, all_gam_results, meta.get("annual_factor", 1.0)
        )

        # Fit (units RMSE)
        fit_val, fit_sub = _model_fit_units(filt)

        # # Robustness & coverage
        # badge = _robustness_badge(filt)
        coverage = _coverage_note(filt)

        # # Date range (from meta)
        # start_txt = _format_date(meta.get("data_start"))
        # end_txt   = _format_date(meta.get("data_end"))

        # Date range calculation
        start_date = pd.to_datetime(meta.get("data_start"))
        end_date = pd.to_datetime(meta.get("data_end"))
        if pd.notna(start_date) and pd.notna(end_date):
            num_days = (end_date - start_date).days + 1
            date_range_val = f"{num_days:,}"
            date_range_subtext = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            date_range_val = "—"
            date_range_subtext = "—"
        return (
            "", date_range_val, date_range_subtext,  # Date Range
            "", curr_price_val,                      # Current Price
            "", asp_val,                             # Recommended Price
            "", elast_val, elast_subtext,            # Elasticity
            "", du_ann,                              # Annualized Potential Units Sold
            "", rev_best_ann,                        # Annualized Potential Revenue
            "", fit_val, fit_sub,                    # Model Fit
            pred_graph,                              # graph
            scenario_data,                           # scenario
            coverage,                                # coverage
        )

        