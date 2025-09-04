# annualize rev opp


# overview.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 

ACCENT = {"color": "#DAA520"}


def layout(products_lookup: pd.DataFrame):
    """Pass the DataFrame with columns ['product_key','product']."""
    return dbc.Container(
        [
            html.H1("Overview", 
                    className="display-5",
                    style={"textAlign": "center", "padding": "58px 0 8px"}),

            # Product selector (centered)
            dbc.Row(
                [
                    dbc.Col(
                        html.Label("Select a Product:",
                                   style={"fontWeight": 600, "textAlign": "right", "marginRight": "10px"}),
                        width="auto"
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="product_dropdown_snap",
                            options=[
                                {"label": r["product"], "value": r["product_key"]}
                                for _, r in products_lookup.iterrows()
                            ],
                            value=products_lookup["product_key"].iloc[0] if len(products_lookup) else None,
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
                ],
                className="g-4 align-items-stretch",          # <— key
                justify="center",
                align="center",
                style={"padding": "10px 0 10px"},
            ),

            # Confidence / Robustness badge
            dbc.Row(dbc.Col(html.Div(id="robustness_badge",
                                     style={"textAlign": "center", "paddingTop": "25px"}), width=12)),
            html.Hr(className="my-4"),

            # Predictive graph + mini scenario table + explainer
            dbc.Row(
                [
                    # pred graph title
                    html.H3("Predictive Graph", className="mt-3",
                        style={"margin-left": "50px", "marginTop": "190px", "color": "#DAA520",}),
                    # pred graph
                    dbc.Col(
                        dcc.Loading(
                            type="circle",
                            children=dcc.Graph(
                                id="gam_results_pred",
                                style={"height": "560px", 
                                    #    "padding":'10px'
                                       },
                                config={"displaylogo": False},
                            ),
                        ),
                        md=8, xs=12, className="mb-3"
                    ),
                    # sceario summary & sticky note
                    dbc.Col(
                        [
                            
                            html.H6("Scenario Summary", className="mb-2", style={"textAlign": "center",  "marginTop":"40px"}),
                            dash_table.DataTable(
                                id="scenario_table",
                                columns=[
                                    {"name": "Case", "id": "case"},
                                    {"name": "Price", "id": "price"},
                                    {"name": "Revenue", "id": "revenue"},
                                ],
                                data=[],
                                style_table={"border": "none",  "marginBottom": "12px"},
                                style_cell={"textAlign": "center", "border": "none", "fontSize": "14px", "padding": "6px"},
                                style_header={"fontWeight": 600, "border": "none", "backgroundColor": "#f6f6f6", "marginTop":"400px"},
                            ),
                            
                            # sticky note
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
                                style={"marginTop": "100px", },
                            ),                      

                        ],
                        md=4, xs=12
                    ),
                ],
                className="g-3 mb-4",
            ),

            html.Hr(className="my-4"),
            # Coverage note
            dbc.Row(dbc.Col(html.Div(id="coverage_note",
                                     style={"textAlign": "center", "color": "#5f6b7a", "fontSize": "0.9em", }), width=12)),


            html.Div(style={"height": "16px"}),

            # Footer
            html.Div([html.Span("made with ♥️ | "), html.Span(html.I("@aqxiao")), html.P("github.com/angie-xiao")],
                     className="text-center py-3",
                     style={"fontSize": "0.8em", "color": "#ac274f", "textAlign": "center",
                            "backgroundColor": "#f3f3f3", "margin": "40px auto 0 auto", "borderRadius": "6px"}),
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
                    # subtle gold accent bar (optional)
                    html.Div(
                        style={
                            "height": "4px", "width": "44px", "margin": "2px auto 8px",
                            "borderRadius": "999px",
                            "background": "linear-gradient(90deg,#DAA520,#F0C64C)",
                            "opacity": 0.9,
                        }
                    ),
                    html.H2(
                        title, className="kpi-eyebrow",
                        style={"color": "#121212", "textAlign": "center", "fontSize": "18px", "letterSpacing": ".14em", "fontWeight": 700},
                    ),
                    html.H1(
                        id=id_value, className="kpi-value",
                        style={"color": "#DAA520", "textAlign": "center", "fontSize": "50px", "fontWeight": 800},
                    ),
                    html.Div(
                        id=id_subtext if id_subtext else f"{id_value}-subtext",
                        className="kpi-subtext text-muted",
                        style={"textAlign": "center", "fontSize": "15px", "marginTop": "6px",
                               "lineHeight": "1.25", "minHeight": "34px"},
                    ),
                ],
                className="d-flex flex-column justify-content-start",
                style={"gap": "6px", "padding": "22px 18px"},
            ),
            # Bootstrap-only cosmetics + a touch of inline elevation
            # Subtle, balanced “card” look
            className="h-100 shadow-sm border rounded-4",
            style={
                "backgroundColor": bg,
                "padding": "12px 0",
                "borderColor": "rgba(17,24,39,0.08)",   # soft edge instead of heavy shadow
                # OPTIONAL: whisper-lift; comment out if you want only Bootstrap's shadow-sm
                "boxShadow": "0 1px 2px rgba(16,24,40,.03), 0 4px 8px rgba(16,24,40,.04)",
                # OPTIONAL: very faint header glow; remove if you want pure flat
                "backgroundImage": "radial-gradient(160% 80% at 50% 0%, rgba(218,165,32,.03), transparent 45%)",
            },
        ),
        width=3,
        className="kpi-card",
    )


def register_callbacks(app,
                       price_quant_df,
                       best50_optimal_pricing_df,
                       curr_price_df,
                       elasticity_df,
                       all_gam_results,
                       products_lookup,
                       viz_cls):
    viz = viz_cls()

    @app.callback(
        Output("card_title_curr_price_snap", "children"),
        Output("curr_price_snap", "children"),
        Output("card_title_snap", "children"),
        Output("card_asp_snap", "children"),
        Output("card_title_elasticity_snap", "children"),
        Output("elasticity_ratio_snap", "children"),
        Output("elasticity-subtext", "children"),    
        Output("card_title_upside_snap","children"),      
        Output("upside_value_snap", "children"),
        Output("upside-subtext", "children"),               
        Output("robustness_badge", "children"),
        Output("gam_results_pred", "figure"),
        Output("scenario_table", "data"),
        # Output("opportunity_chart", "figure"),
        Output("coverage_note", "children"),
        Input("product_dropdown_snap", "value"),
    )
    def overview(product_key):
        # Ensure percentile column exists (if you didn't add it upstream)
        edf = elasticity_df.copy()
        if "pct" not in edf.columns and "ratio" in edf.columns:
            edf["pct"] = edf["ratio"].rank(pct=True) * 100

        # ---- No selection: return exactly 13 placeholders in correct order
        if not product_key:
            empty_fig = viz.empty_fig("Select a product")
            return (
                "",                 # card_title_curr_price_snap
                "—",                # curr_price_snap
                "",                 # card_title_snap
                "—",                # card_asp_snap
                "",                 # card_title_elasticity_snap
                "—",                # elasticity_ratio_snap
                "",                 # elasticity-subtext   
                "",                 # card_title_upside_snap
                "—",                # upside_value_snap
                "",                 # upside-subtext             
                "",                 # robustness_badge
                empty_fig,          # gam_results_pred.figure
                [],                 # scenario_table.data
                # empty_fig,          # opportunity_chart.figure
                "",                 # coverage_note.children
            )

        # ----- Display label
        try:
            display_name = products_lookup.loc[
                products_lookup["product_key"] == product_key, "product"
            ].iloc[0]
        except Exception:
            display_name = product_key

        # ----- Current price
        curr = curr_price_df.loc[curr_price_df["product_key"] == product_key, "current_price"]
        curr_price_val = f"${float(curr.iloc[0]):,.2f}" if len(curr) else "—"

        # ----- Recommended price
        filt_opt = best50_optimal_pricing_df[best50_optimal_pricing_df["product_key"] == product_key]
        if len(filt_opt):
            try:
                asp_val = f"${float(filt_opt['asp'].iloc[0]):,.2f}"
            except Exception:
                asp_val = str(filt_opt["asp"].iloc[0])
        else:
            asp_val = "—"

        # ----- Elasticity (value + percentile)
        value_text, elast_subtext = _update_elasticity_kpi(product_key, edf[["product_key","ratio","pct"]].to_dict("records"))
        elast_val = value_text

        # ----- Prediction graph
        filt = all_gam_results[all_gam_results["product_key"] == product_key]
        pred_graph = viz.gam_results(filt) if len(filt) else viz.empty_fig("No model data")

        # ----- Scenario summary
        scenario_df = _scenario_table(filt) if len(filt) else pd.DataFrame(
            [{"case": "—", "price": "—", "revenue": "—"}]
        )
        scenario_data = scenario_df.to_dict("records")

        # ----- Upside (split into value + subtext)
        upside_val, upside_sub = _upside_vs_current_parts(
            curr_price_df, best50_optimal_pricing_df, product_key, all_gam_results
        )

        # ----- Robustness badge
        badge = _robustness_badge(filt)
 
        # ----- Coverage note
        coverage = _coverage_note(filt)

        # Return EXACTLY 14 in same order as Outputs
        return (
            display_name,         # 1
            curr_price_val,       # 2
            display_name,         # 3
            asp_val,              # 4
            display_name,         # 5
            elast_val,            # 6
            elast_subtext,        # 7
            display_name,
            upside_val,           # 8
            upside_sub,           # 9
            badge,                # 10
            pred_graph,           # 11
            scenario_data,        # 12
            coverage,             # 13
        )



# ---------- Helpers ----------
    
def _update_elasticity_kpi(product_key, elast_data):
    ''' '''
    # if not product_key or not elast_data:
    #     return no_update, no_update

    df = pd.DataFrame(elast_data)
    row = df.loc[df["product_key"] == product_key]
    if row.empty:
        return "—", ""

    ratio = float(row["ratio"].iloc[0])
    pct   = float(row["pct"].iloc[0])  # 0–100

    # Format displays
    value_text = f"{ratio:,.2f}"
    pct_round  = int(round(pct))

    # “Top X%” helper (e.g., 90th pct = Top 10%)
    top_share  = max(1, 100 - pct_round)

    if pct >= 50: 
        subtext = f"Top ~{top_share}% most ELASTIC"

    elif pct < 50:
        subtext = f"Top ~{top_share}% most INELASTIC"
 
    return value_text, subtext


def _scenario_table(prod_df: pd.DataFrame) -> pd.DataFrame:
    if prod_df.empty:
        return pd.DataFrame(columns=["case", "price", "revenue"])
    rows = []
    for col in ["revenue_pred_0.025", "revenue_pred_0.5", "revenue_pred_0.975"]:
        row = prod_df.loc[prod_df[col] == prod_df[col].max()].head(1)
        if not row.empty:
            rows.append({
                "case": {"revenue_pred_0.025": "Conservative",
                         "revenue_pred_0.5": "Expected",
                         "revenue_pred_0.975": "Optimistic"}[col],
                "price": f"${row['asp'].iloc[0]:,.2f}",
                "revenue": f"${row[col].iloc[0]:,.0f}",
            })
    return pd.DataFrame(rows)


def _upside_vs_current_parts(curr_df, best50_df, product_key, all_gam):
    """
    Returns (value_text, subtext_text):
      value_text = signed $ delta
      subtext_text = signed (% delta) wrapped in parentheses, or "" if N/A
    """
    try:
        cp = curr_df.loc[curr_df["product_key"] == product_key, "current_price"]
        if cp.empty or pd.isna(cp.iloc[0]):
            return "—", ""
        curr_price = float(cp.iloc[0])

        prod = all_gam[
            (all_gam["product_key"] == product_key)
            & pd.notna(all_gam["asp"])
            & pd.notna(all_gam["revenue_pred_0.5"])
        ]
        if prod.empty:
            return "—", ""

        # nearest ASP to current price on expected curve
        idx = (prod["asp"] - curr_price).abs().idxmin()
        rev_at_curr = float(prod.loc[idx, "revenue_pred_0.5"])

        best = best50_df[best50_df["product_key"] == product_key]
        if best.empty or pd.isna(best["revenue_pred_0.5"].iloc[0]):
            return "—", ""

        rev_at_best = float(best["revenue_pred_0.5"].iloc[0])

        delta = rev_at_best - rev_at_curr
        sign = "+" if delta >= 0 else "-"
        value_text = f"{sign}${abs(delta):,.0f}"

        # percent subtext
        pct = (delta / rev_at_curr * 100) if rev_at_curr else np.nan
        subtext_text = f"({sign}{abs(pct):,.1f}%)" if pd.notna(pct) else ""

        return value_text, subtext_text
    except Exception:
        return "—", ""


def _robustness_badge(prod_df: pd.DataFrame):
    """
    Confidence combines:
      (1) Spread alignment across scenarios (tighter = better)
      (2) Elasticity (lower magnitude = better)
      (3) Data-volume credibility (more distinct ASPs = more believable)

    param
        prod_df [df]
    
    return 
        dbc.Badge
    """

    if prod_df.empty:
        return ""

    def peak_x(col):
        try:
            return prod_df.loc[prod_df[col] == prod_df[col].max(), "asp"].iloc[0]
        except Exception:
            return np.nan

    # 1. Get peak ASPs at 3 revenue prediction percentiles
    p_low = peak_x("revenue_pred_0.025")        # worst
    p_mid = peak_x("revenue_pred_0.5")          # median
    p_high = peak_x("revenue_pred_0.975")       # best

    #------------------- 2. Spread Score (tighter spread = higher confidence) -------------------
    # measuring how consistent the model’s recommended prices are across three different forecast scenarios
    align_spread = np.nanmax([p_low, p_mid, p_high]) - np.nanmin([p_low, p_mid, p_high])  # smaller range = more trust

    # reward tight alignment between scenarios, using 10% of median price as benchmark, punishing exponentially
    spread_score = np.exp(               
        -align_spread /                 
        (
            0.1 *                       # …relative to 10% of the median price (reasonable tolerance)
            (p_mid if p_mid else 1)     # fallback to 1 to avoid division by zero
        )
    ) if p_mid else 0  # If p_mid is missing, we can’t assess spread confidence → return 0


    #------------------- 3. Elasticity Score (lower elasticity = higher confidence) -------------------
    # find elasticity at the ASP that gives max P50 revenue
    try:
        elasticity_at_mid = prod_df.loc[prod_df["asp"] == p_mid, "elasticity"].iloc[0]
    except Exception:
        elasticity_at_mid = np.nan

    if pd.isna(elasticity_at_mid):
        elasticity_score = 0
    else:
        # normalize using min-max across all observed elasticities in this product dataframe
        min_el = prod_df["elasticity"].min()
        max_el = prod_df["elasticity"].max()
        if max_el - min_el > 0:
            # higher elasticity → lower score, invert the scale
            elasticity_score = 1 - (elasticity_at_mid - min_el) / (max_el - min_el)
        else:
            elasticity_score = 1  # all elasticities are the same → full confidence

            
    #------------------- 4. Final Score: weight spread and elasticity -------------------
    base_score = 0.4 * spread_score + 0.6 * elasticity_score


    #------------------- 5) Data-volume credibility (more data = higher credibility) -------------------
    # Use the number of DISTINCT prices tested — variety matters more than raw row count
    try:
        n_points = int(prod_df["asp"].nunique(dropna=True))
    except Exception:
        n_points = prod_df.shape[0]

    # Smooth saturating curve in [0,1): rises quickly with first few distinct ASPs, then flattens
    data_strength = 1.0 - np.exp(-n_points / 6.0)  # ~0.80 by ~10 distinct ASPs

    # Convert to a gentle multiplier in [0.6, 1.0]: never crushes the score when data are light,
    # but rewards breadth of evidence (distinct price experiments).
    credibility_multiplier = 0.6 + 0.4 * data_strength
    
    final_score = base_score * credibility_multiplier

    # 6. Label logic
    label, color = ("Weak", "danger")
    if final_score >= 0.7:
        label, color = ("Strong", "success")
    elif final_score >= 0.45:
        label, color = ("Medium", "warning")

    return dbc.Badge(f"Confidence: {label}", color=color, pill=True, className="px-3 py-2")


def _coverage_note(prod_df: pd.DataFrame) -> str:
    if prod_df.empty:
        return ""
    n_points = len(prod_df)
    within = ((prod_df["revenue_actual"] >= prod_df["revenue_pred_0.025"]) &
              (prod_df["revenue_actual"] <= prod_df["revenue_pred_0.975"])).mean()
    return f"Based on {n_points} historical points;\n{within*100:,.0f}% of actual revenue outcomes fall within the shown range."
