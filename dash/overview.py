# annualize rev opp
# overview.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

ACCENT = {"color": "#DAA520"}


def layout(products_lookup: pd.DataFrame):
    """Pass the DataFrame with columns ['product_key','product'].""" 
    return dbc.Container(
        [
            html.H1(
                "Overview",
                className="display-5",
                style={"textAlign": "center", "padding": "58px 0 8px"},
            ),

            # Product selector (centered)
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            "Select a Product:",
                            style={
                                "fontWeight": 600,
                                "textAlign": "right",
                                "marginRight": "10px",
                            },
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
                            value=products_lookup["product_key"].iloc[0]
                            if len(products_lookup)
                            else None,
                            style={"width": "280px"},
                        ),
                        width="auto",
                    ),
                ],
                justify="center",
                align="center",
                style={"padding": "10px 0 20px"},
            ),

            # KPI row
            dbc.Row(
                [
                    _kpi_card("card_title_curr_price_snap", "Current Price", "curr_price_snap", bg="#f3f0f0"),
                    _kpi_card("card_title_snap", "Recommended Price", "card_asp_snap", bg="#F5E8D8"),
                    _kpi_card("card_title_elasticity_snap", "Elasticity", "elasticity_ratio_snap", bg="#f3f0f0", id_subtext="elasticity-subtext"),
                    _kpi_card("card_title_upside_snap", "Revenue Opportunity", "upside_value_snap", bg="#eef8f0", id_subtext="upside-subtext"),
                    _kpi_card("card_title_fit_snap", "Model Fit Evaluation (Shipped Units)", "fit_value_snap", bg="#eef2fa", id_subtext="fit-subtext"),
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

            # Coverage note (centered, two lines)
            dbc.Row(
                dbc.Col(
                    html.Div(id="coverage_note"),
                    width=12,
                    style={"textAlign": "center", "color": "#5f6b7a", "fontSize": "0.9em"},
                )
            ),
            html.Hr(className="my-4"),

            # Predictive graph + mini scenario table + explainer
            dbc.Row(
                [
                    # pred graph title
                    html.H3(
                        "Predictive Graph",
                        className="mt-3",
                        style={"margin-left": "50px", "marginTop": "190px", "color": "#DAA520"},
                    ),

                    # pred graph
                    dbc.Col(
                        dcc.Loading(
                            type="circle",
                            children=dcc.Graph(
                                id="gam_results_pred",
                                style={"height": "560px"},
                                config={"displaylogo": False},
                            ),
                        ),
                        md=8,
                        xs=12,
                        className="mb-3",
                    ),

                    # scenario summary & sticky note
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
                                style_header={"fontWeight": 600, "border": "none", "backgroundColor": "#f6f6f6", "marginTop": "400px"},
                            ),
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
                        md=4,
                        xs=12,
                    ),
                ],
                className="g-3 mb-4",
            ),

            html.Div(style={"height": "16px"}),

            # Footer
            html.Div(
                [html.Span("made with ♥️ | "), html.Span(html.I("@aqxiao")), html.P("github.com/angie-xiao")],
                className="text-center py-3",
                style={
                    "fontSize": "0.8em",
                    "color": "#ac274f",
                    "textAlign": "center",
                    "backgroundColor": "#f3f3f3",
                    "margin": "40px auto 0 auto",
                    "borderRadius": "6px",
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
                        id=id_title,
                        className="kpi-title",
                        style={
                            "color": "#121212",
                            "textAlign": "center",
                            "marginBottom": "10px",
                            "marginTop": "10px",
                            "whiteSpace": "nowrap",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                    ),
                    html.Div(
                        style={
                            "height": "4px",
                            "width": "44px",
                            "margin": "2px auto 8px",
                            "borderRadius": "999px",
                            "background": "linear-gradient(90deg,#DAA520,#F0C64C)",
                            "opacity": 0.9,
                        }
                    ),
                    html.H2(
                        title,
                        className="kpi-eyebrow",
                        style={"color": "#121212", "textAlign": "center", "fontSize": "18px", "letterSpacing": ".14em", "fontWeight": 700},
                    ),
                    html.H1(
                        id=id_value,
                        className="kpi-value",
                        style={"color": "#DAA520", "textAlign": "center", "fontSize": "50px", "fontWeight": 800},
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
                "backgroundColor": bg,
                "padding": "12px 0",
                "borderColor": "rgba(17,24,39,0.08)",
                "boxShadow": "0 1px 2px rgba(16,24,40,.03), 0 4px 8px rgba(16,24,40,.04)",
                "backgroundImage": "radial-gradient(160% 80% at 50% 0%, rgba(218,165,32,.03), transparent 45%)",
            },
        ),
        width=3,
        className="kpi-card",
    )

def _model_fit_units(prod_df: pd.DataFrame):
    """
    Return (value_text, subtext) for model accuracy on shipped units.
    - Computes RMSE on the *current product slice* only.
    - Aggregates by ASP to avoid double-counting repeated ASP rows (optional but helpful).
    - Shows one decimal place so small changes are visible.
    """
    if prod_df is None or prod_df.empty:
        return "—", ""

    # must have both actual and prediction
    need = {"shipped_units", "pred_0.5", "asp"}
    if not need.issubset(prod_df.columns):
        return "—", ""

    # OPTIONAL: aggregate to one row per ASP to stabilize eval when there are duplicates
    df = prod_df[["asp", "shipped_units", "pred_0.5"]].dropna()
    if df.empty:
        return "—", ""

    df = (
        df.groupby("asp", as_index=False)
          .agg(shipped_units=("shipped_units", "sum"),
               pred_0_5=("pred_0.5", "mean"))
    )

    y_true = df["shipped_units"].to_numpy(dtype=float)
    y_pred = df["pred_0_5"].to_numpy(dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        return "—", ""

    rmse_val = mean_squared_error(y_true, y_pred, squared=False)  # in units
    avg_units = float(np.mean(y_true)) if y_true.size else np.nan
    pct_err = (rmse_val / avg_units * 100.0) if avg_units else np.nan

    # 1 decimal so you can actually see changes between products/model tweaks
    value_text = f"±{rmse_val:,.1f}"
    subtext = f"≈{pct_err:.1f}% typical error" if np.isfinite(pct_err) else ""

    # print(
    #     "[fit_kpi]",
    #     "n_rows:", len(prod_df),
    #     "n_unique_asp:", prod_df["asp"].nunique() if "asp" in prod_df else None,
    #     "rmse_example:", f"{rmse_val:.3f}" if 'rmse_val' in locals() else None
    # )

    return value_text, subtext


def _upside_vs_current_parts(curr_df, best50_df, product_key, all_gam):
    """
    Returns (value_text, subtext_text):
      value_text = signed $ delta between revenue at recommended vs current price
      subtext_text = signed (% delta) in parentheses, or "" if N/A
    """
    try:
        # current price for this product
        cp = curr_df.loc[curr_df["product_key"] == product_key, "current_price"]
        if cp.empty or pd.isna(cp.iloc[0]):
            return "—", ""
        curr_price = float(cp.iloc[0])

        # expected revenue curve for this product (P50)
        prod = all_gam[
            (all_gam["product_key"] == product_key)
            & pd.notna(all_gam["asp"])
            & pd.notna(all_gam["revenue_pred_0.5"])
        ]
        if prod.empty:
            return "—", ""

        # revenue at current price (nearest ASP on the curve)
        idx = (prod["asp"] - curr_price).abs().idxmin()
        rev_at_curr = float(prod.loc[idx, "revenue_pred_0.5"])

        # revenue at recommended price
        best = best50_df[best50_df["product_key"] == product_key]
        if best.empty or pd.isna(best["revenue_pred_0.5"].iloc[0]):
            return "—", ""
        rev_at_best = float(best["revenue_pred_0.5"].iloc[0])

        # deltas
        delta = rev_at_best - rev_at_curr
        sign = "+" if delta >= 0 else "-"
        value_text = f"{sign}${abs(delta):,.0f}"
        pct = (delta / rev_at_curr * 100) if rev_at_curr else np.nan
        subtext_text = f"({sign}{abs(pct):,.1f}%)" if pd.notna(pct) else ""
        return value_text, subtext_text
    except Exception:
        return "—", ""


def register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing_df,
    curr_price_df,
    elasticity_df,
    all_gam_results,
    products_lookup,
    viz_cls,
):
    viz = viz_cls()

    @app.callback(
        Output("card_title_curr_price_snap", "children"),
        Output("curr_price_snap", "children"),
        Output("card_title_snap", "children"),
        Output("card_asp_snap", "children"),
        Output("card_title_elasticity_snap", "children"),
        Output("elasticity_ratio_snap", "children"),
        Output("elasticity-subtext", "children"),
        Output("card_title_upside_snap", "children"),
        Output("upside_value_snap", "children"),
        Output("upside-subtext", "children"),
        Output("fit_value_snap", "children"),
        Output("fit-subtext", "children"),
        Output("robustness_badge", "children"),
        Output("gam_results_pred", "figure"),
        Output("scenario_table", "data"),
        Output("coverage_note", "children"),
        Input("product_dropdown_snap", "value"),
    )
    def overview(product_key):
        # Ensure percentile exists if ratio exists
        edf = elasticity_df.copy()
        if "pct" not in edf.columns and "ratio" in edf.columns:
            edf["pct"] = edf["ratio"].rank(pct=True) * 100

        # ----- No selection: return EXACTLY 16 placeholders (matches Outputs) -----
        if not product_key:
            empty_fig = viz.empty_fig("Select a product")
            return (
                "",   # title: current price (product name)
                "—",  # current price
                "",   # title: recommended price
                "—",  # recommended price
                "",   # title: elasticity
                "—",  # elasticity value
                "",   # elasticity subtext
                "",   # title: upside
                "—",  # upside value
                "",   # upside subtext
                "—",  # fit value
                "",   # fit subtext
                "",   # robustness badge
                empty_fig,  # graph
                [],   # scenario table
                "",   # coverage note
            )

        # ----- Resolve display name (product) from key -----
        try:
            display_name = products_lookup.loc[
                products_lookup["product_key"] == product_key, "product"
            ].iloc[0]
        except Exception:
            display_name = product_key

        # ----- Current price -----
        curr = curr_price_df.loc[curr_price_df["product_key"] == product_key, "current_price"]
        curr_price_val = f"${float(curr.iloc[0]):,.2f}" if len(curr) else "—"

        # ----- Recommended price (from best50 table) -----
        filt_opt = best50_optimal_pricing_df[best50_optimal_pricing_df["product_key"] == product_key]
        if len(filt_opt):
            try:
                asp_val = f"${float(filt_opt['asp'].iloc[0]):,.2f}"
            except Exception:
                asp_val = str(filt_opt["asp"].iloc[0])
        else:
            asp_val = "—"

        # ----- Elasticity KPI (lookup by product NAME, not key) -----
        elast_val, elast_subtext = _update_elasticity_kpi_by_product(display_name, edf)

        # ----- Product slice for plots/KPIs -----
        filt = all_gam_results[all_gam_results["product_key"] == product_key]
        pred_graph = viz.gam_results(filt) if len(filt) else viz.empty_fig("No model data")
        
        
        try:
            model_fingerprint = f"rows={len(all_gam_results)}, prods={all_gam_results['product_key'].nunique()}, rmse_col?={'pred_0.5' in all_gam_results}"
        except Exception:
            model_fingerprint = "n/a"

        pred_graph.update_layout(
            title_text=f"Predictive Graph  •  model={model_fingerprint}"
        )
        print("[model_version]", model_fingerprint, "id(all_gam_results)=", id(all_gam_results))



        # ----- Scenario summary -----
        scenario_df = _scenario_table(filt) if len(filt) else pd.DataFrame([{"case": "—", "price": "—", "revenue": "—"}])
        scenario_data = scenario_df.to_dict("records")

        # ----- Upside vs current -----
        upside_val, upside_sub = _upside_vs_current_parts(
            curr_price_df, best50_optimal_pricing_df, product_key, all_gam_results
        )

        # ----- Model fit (units) -----
        fit_val, fit_sub = _model_fit_units(filt)

        # ----- Robustness badge & coverage note -----
        badge = _robustness_badge(filt)
        coverage = _coverage_note(filt)

        # Return EXACTLY 16, in same order as Outputs
        return (
            display_name,     # title: current price
            curr_price_val,   # current price
            display_name,     # title: recommended price
            asp_val,          # recommended price
            display_name,     # title: elasticity
            elast_val,        # elasticity value
            elast_subtext,    # elasticity subtext
            display_name,     # title: upside
            upside_val,       # upside value
            upside_sub,       # upside subtext
            fit_val,          # fit value (units RMSE)
            fit_sub,          # fit subtext (% of avg units)
            badge,            # robustness
            pred_graph,       # graph
            scenario_data,    # scenario
            coverage,         # coverage
        )


# ---------- helpers ----------
def _update_elasticity_kpi_by_product(product_name: str, elast_df: pd.DataFrame):
    """Look up elasticity by product NAME (elasticity_df usually has 'product', not 'product_key')."""
    try:
        row = elast_df.loc[elast_df["product"] == product_name]
        if row.empty or "ratio" not in row or "pct" not in row:
            return "—", ""
        ratio = float(row["ratio"].iloc[0])
        pct = float(row["pct"].iloc[0])
        value_text = f"{ratio:,.2f}"
        pct_round = int(round(pct))
        top_share = max(1, 100 - pct_round)  # e.g., 90th pct -> Top 10%
        subtext = (
            f"Top ~{top_share}% most ELASTIC"
            if pct >= 50
            else f"Top ~{top_share}% most INELASTIC"
        )
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


def _robustness_badge(prod_df):
    """
    Confidence combines:
      (1) Spread alignment across scenarios (tighter = better)
      (2) Elasticity at the recommended price (lower magnitude = better) — optional
      (3) Data-volume credibility (more distinct ASPs = more believable)
    Returns a dbc.Badge (or "" if not enough data).
    """
    if prod_df is None or prod_df.empty:
        return ""

    # helper to get ASP at each scenario's revenue peak
    def _peak_asp(col):
        if col not in prod_df or prod_df[col].isna().all():
            return np.nan
        idx = prod_df[col].idxmax()
        try:
            return float(prod_df.loc[idx, "asp"])
        except Exception:
            return np.nan

    p_low = _peak_asp("revenue_pred_0.025")
    p_mid = _peak_asp("revenue_pred_0.5")
    p_high = _peak_asp("revenue_pred_0.975")

    # (1) spread alignment
    if np.isnan([p_low, p_mid, p_high]).any() or (not p_mid):
        spread_score = 0.0
    else:
        align_spread = max(p_low, p_mid, p_high) - min(p_low, p_mid, p_high)
        spread_score = float(np.exp(-align_spread / (0.1 * p_mid)))  # 10% of median price

    # (2) elasticity score (optional)
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

    # (3) credibility (distinct price coverage)
    try:
        n_distinct_prices = int(prod_df["asp"].nunique(dropna=True))
    except Exception:
        n_distinct_prices = prod_df.shape[0]
    data_strength = 1.0 - float(np.exp(-n_distinct_prices / 6.0))
    credibility_multiplier = 0.6 + 0.4 * data_strength

    base_score = 0.4 * spread_score + 0.6 * elasticity_score
    final_score = base_score * credibility_multiplier

    label, color = ("Weak", "danger")
    if final_score >= 0.70:
        label, color = ("Strong", "success")
    elif final_score >= 0.45:
        label, color = ("Medium", "warning")

    return dbc.Badge(f"Confidence: {label}", color=color, pill=True, className="px-3 py-2")


def _coverage_note(prod_df):
    """
    Centered, two-line coverage note with highlighted numbers.
    Expects columns: revenue_actual, revenue_pred_0.025, revenue_pred_0.975
    """
    if prod_df is None or prod_df.empty:
        return ""

    n_points = int(len(prod_df))
    if {"revenue_actual", "revenue_pred_0.025", "revenue_pred_0.975"}.issubset(prod_df.columns):
        within = (
            (prod_df["revenue_actual"] >= prod_df["revenue_pred_0.025"])
            & (prod_df["revenue_actual"] <= prod_df["revenue_pred_0.975"])
        ).mean()
    else:
        within = np.nan

    return html.Div(
        [
            html.Div(
                [
                    "Based on ",
                    html.Span(f"{n_points}", style={"color": ACCENT["color"], "fontWeight": 600}),
                    " historical points;",
                ],
                style={"textAlign": "center"},
            ),
            html.Div(
                [
                    html.Span(
                        f"{within*100:,.0f}%" if np.isfinite(within) else "—",
                        style={"color": ACCENT["color"], "fontWeight": 600},
                    ),
                    " of actual revenue outcomes fall within the shown range.",
                ],
                style={"textAlign": "center"},
            ),
        ],
        style={"marginTop": "8px"},
    )
