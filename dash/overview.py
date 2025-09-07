# overview.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

ACCENT = {"color": "#DAA520"}

def layout(products_lookup: pd.DataFrame):
    """Pass a Top-N-only DataFrame with columns ['product_key','product'].""" 
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
                                {"label": r["product"], "value": r["product_key"]}
                                for _, r in products_lookup.iterrows()
                            ],
                            value=(
                                products_lookup["product_key"].iloc[0]
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
                    _kpi_card("card_title_start_date", "Date Start", "start_date_value", bg="#f3f0f0"),
                    _kpi_card("card_title_end_date", "Date End", "end_date_value", bg="#f3f0f0"),
                    _kpi_card("card_title_curr_price_snap", "Current Price", "curr_price_snap", bg="#f3f0f0"),
                    _kpi_card("card_title_snap", "Recommended Price", "card_asp_snap", bg="#F5E8D8"),
                    _kpi_card("card_title_elasticity_snap", "Elasticity", "elasticity_ratio_snap", bg="#f3f0f0", id_subtext="elasticity-subtext"),
                    _kpi_card("card_title_units_opp_ann", "Annualized Potential Units Sold", "units_opp_ann_value", bg="#eef8f0"),
                    _kpi_card("card_title_rev_best_ann", "Annualized Potential Revenue", "rev_best_ann_value", bg="#eef8f0"),
                    _kpi_card("card_title_fit_snap", "Model Fit", "fit_value_snap", bg="#eef2fa", id_subtext="fit-subtext"),
                ],
                className="g-4 align-items-stretch",
                justify="center",
                align="center",
                style={"padding": "10px 0 10px"},
            ),

            # Confidence / Robustness badge
            dbc.Row(
                dbc.Col(
                    html.Div(id="robustness_badge", style={"textAlign": "center", "padding": "25px"}),
                    width=12,
                )
            ),

            html.Hr(className="my-4"),

            # Predictive graph + scenario table + explainer
            dbc.Row(
                [   
                    # title
                    html.H3(
                        "Predictive Graph",
                        className="mt-3",
                        style={"margin-left": "50px", "marginTop": "190px", "color": "#DAA520"},
                    ),
                    # graph
                    dbc.Col(
                        dcc.Loading(
                            type="circle",
                            children=dcc.Graph(
                                id="gam_results_pred",
                                style={"height": "560px"},
                                config={"displaylogo": False},
                            ),
                        ),
                        md=8, xs=12, className="mb-3",
                    ),
                    
                    dbc.Col(
                        [
                            # scenario table
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
                                style_header={"fontWeight": 600, "border": "none", "backgroundColor": "#f6f6f6", "marginTop": "400px"},
                            ),
                            # reading notes
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6("How to read this", className="mb-2 fw-bold text-uppercase"),
                                        html.Ul(
                                            [
                                                html.Li([html.B("Goal: "), "Pick the price where the central curve's expected revenue is highest."]),
                                                html.Li([html.B("Range: "), "Conservative and optimistic curves show how outcomes may vary."]),
                                                html.Li([html.B("Confidence: "), "Stronger when all curves peak around a similar price and there's lots of nearby data."]),
                                            ],
                                            className="predictive-explainer-list",
                                        ),
                                    ]
                                ),
                                className="shadow-sm",
                                style={"marginTop": "70px"},
                            ),
                        ],
                        md=4, xs=12,
                    ),
                ],
                className="g-3 mb-4",
            ),

            html.Div(style={"height": "16px"}),

            # Coverage note
            dbc.Row(
                dbc.Col(
                    html.Span(html.I(id="coverage_note")),
                    width=12,
                    style={"textAlign": "center", "color": "#5f6b7a", "fontSize": "0.9em"},
                )
            ),

            # Footer
            html.Div(
                [html.Span("made with ♥️ | "), html.Span(html.I("@aqxiao")), html.P("github.com/angie-xiao")],
                className="text-center py-3",
                style={"fontSize": "0.8em", "color": "#ac274f", "textAlign": "center",
                       "backgroundColor": "#f3f3f3", "margin": "40px auto 0 auto", "borderRadius": "6px"},
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
    Return (value_text, subtext) for model accuracy on shipped units (P50).
    Aggregates to one row per ASP for stable eval; shows 1 decimal place.
    """
    if prod_df is None or prod_df.empty:
        return "—", ""
    need = {"shipped_units", "pred_0.5", "asp"}
    if not need.issubset(prod_df.columns):
        return "—", ""
    df = prod_df[["asp", "shipped_units", "pred_0.5"]].dropna()
    if df.empty:
        return "—", ""
    df = df.groupby("asp", as_index=False).agg(
        shipped_units=("shipped_units", "sum"),
        pred_0_5=("pred_0.5", "mean"),
    )
    y_true = df["shipped_units"].to_numpy(float)
    y_pred = df["pred_0_5"].to_numpy(float)
    if y_true.size == 0:
        return "—", ""
    rmse_val = mean_squared_error(y_true, y_pred, squared=False)
    avg_units = float(np.mean(y_true)) if y_true.size else np.nan
    pct_err = (rmse_val / avg_units * 100.0) if avg_units else np.nan
    return f"±{rmse_val:,.1f}", (f"≈{pct_err:.1f}% typical error" if np.isfinite(pct_err) else "")

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

def _annualized_kpis_signed(product_key, best50_df, curr_price_df, all_gam, annual_factor):
    """
    Returns formatted strings:
      +Annualized Δ Units  and  +$Annualized Potential Revenue
    """
    try:
        best = best50_df[best50_df["product_key"] == product_key]
        if best.empty:
            return "—", "—"
        units_best = float(best.get("pred_0.5", np.nan).iloc[0]) if "pred_0.5" in best else np.nan
        rev_best   = float(best.get("revenue_pred_0.5", np.nan).iloc[0]) if "revenue_pred_0.5" in best else np.nan

        cp = curr_price_df.loc[curr_price_df["product_key"] == product_key, "current_price"]
        curr_price = float(cp.iloc[0]) if len(cp) else np.nan

        prod = all_gam[
            (all_gam["product_key"] == product_key)
            & pd.notna(all_gam["asp"])
            & pd.notna(all_gam["pred_0.5"])
        ]
        if prod.empty:
            du_ann = np.nan
        else:
            idx = (prod["asp"] - curr_price).abs().idxmin() if pd.notna(curr_price) else None
            units_curr = float(prod.loc[idx, "pred_0.5"]) if idx is not None else np.nan
            du = (units_best - units_curr) if (pd.notna(units_best) and pd.notna(units_curr)) else np.nan
            du_ann = du * float(annual_factor) if pd.notna(du) else np.nan

        rev_best_ann = rev_best * float(annual_factor) if pd.notna(rev_best) else np.nan
        return _format_signed_units(du_ann), _format_signed_money(rev_best_ann)
    except Exception:
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
        # 8 KPI cards (title/value pairs + the one subtext for elasticity and fit)
        Output("card_title_start_date", "children"),
        Output("start_date_value", "children"),
        Output("card_title_end_date", "children"),
        Output("end_date_value", "children"),
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
        Output("robustness_badge", "children"),
        Output("gam_results_pred", "figure"),
        Output("scenario_table", "data"),
        Output("coverage_note", "children"),
        Input("product_dropdown_snap", "value"),
    )
    def overview(product_key):
        # Empty selection -> placeholders
        if not product_key:
            empty_fig = viz.empty_fig("Select a product")
            return (
                "", "—", "", "—", "", "—", "", "—", "", "—", "",
                "", "—", "", "—", "", "—", "",
                "", empty_fig, [], ""
            )

        # Resolve display name (optional; we keep titles static)
        try:
            display_name = products_lookup.loc[
                products_lookup["product_key"] == product_key, "product"
            ].iloc[0]
        except Exception:
            display_name = ""

        # Current price
        cp = curr_price_df.loc[curr_price_df["product_key"] == product_key, "current_price"]
        curr_price_val = f"${float(cp.iloc[0]):,.2f}" if len(cp) else "—"

        # Recommended price (P50 best)
        best_row = best50_optimal_pricing_df.loc[
            best50_optimal_pricing_df["product_key"] == product_key
        ]
        asp_val = f"${float(best_row['asp'].iloc[0]):,.2f}" if len(best_row) else "—"

        # Elasticity KPI (lookup by product name)
        elast_val, elast_subtext = _update_elasticity_kpi_by_product(display_name, elasticity_df)

        # Product slice for plots/KPIs
        filt = all_gam_results[all_gam_results["product_key"] == product_key]
        pred_graph = viz.gam_results(filt) if len(filt) else viz.empty_fig("No model data")

        # Scenario summary
        scenario_df = _scenario_table(filt) if len(filt) else pd.DataFrame([{"case": "—", "price": "—", "revenue": "—"}])
        scenario_data = scenario_df.to_dict("records")

        # Annualized KPIs (signed, with + prefixes where applicable)
        du_ann, rev_best_ann = _annualized_kpis_signed(
            product_key, best50_optimal_pricing_df, curr_price_df, all_gam_results, meta.get("annual_factor", 1.0)
        )

        # Fit (units RMSE)
        fit_val, fit_sub = _model_fit_units(filt)

        # Robustness & coverage
        badge = _robustness_badge(filt)
        coverage = _coverage_note(filt)

        # Date range (from meta)
        start_txt = _format_date(meta.get("data_start"))
        end_txt   = _format_date(meta.get("data_end"))

        return (
            "", start_txt,                 # Date Start (title text blank)
            "", end_txt,                   # Date End   (title text blank)
            "", curr_price_val,            # Current Price
            "", asp_val,                   # Recommended Price
            "", elast_val, elast_subtext,  # Elasticity
            "", du_ann,                    # Annualized Potential Units Sold
            "", rev_best_ann,              # Annualized Potential Revenue
            "", fit_val, fit_sub,          # Model Fit
            badge,                         # robustness
            pred_graph,                    # graph
            scenario_data,                 # scenario
            coverage,                      # coverage
        )
