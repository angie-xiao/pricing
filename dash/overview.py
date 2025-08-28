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

            # Coverage note
            dbc.Row(dbc.Col(html.Div(id="coverage_note",
                                     style={"textAlign": "center", "color": "#5f6b7a", "fontSize": "0.9em", }), width=12)),

            html.Hr(className="my-4"),

            # Elasticity section: dist + ranked table
            # dbc.Row(
            #     [
            #         dbc.Col(
            #             [
            #                 html.H4("Top Products Elasticity", style={"color": "#DAA520"}),
            #                 html.Span("Elasticity = % change in demand / % change in price",
            #                           style={"color": "#5f6b7a", "display": "block", "marginBottom": "6px"}),
            #                 dcc.Loading(
            #                     type="circle",
            #                     children=dcc.Graph(id="elast_dist", style={"height": "420px"},
            #                                        config={"displaylogo": False})
            #                 ),
            #             ],
            #             md=7, xs=12, className="mb-3"
            #         ),
            #         dbc.Col(
            #             dash_table.DataTable(
            #                 id="elast_table",
            #                 columns=[
            #                     {"name": "Rank", "id": "rank"},
            #                     {"name": "Product", "id": "product"},
            #                     {"name": "Elasticity", "id": "ratio", "type": "numeric"},
            #                 ],
            #                 data=[],
            #                 page_size=10,
            #                 sort_action="native",
            #                 style_table={"height": "420px", "overflowY": "auto", "border": "none"},
            #                 style_cell={"padding": "10px", "fontSize": "14px", "border": "none", "textAlign": "center"},
            #                 style_header={"fontWeight": 600, "border": "none", "backgroundColor": "#f6f6f6"},
            #             ),
            #             md=5, xs=12
            #         ),
            #     ],
            #     className="g-3"
            # ),

            # Portfolio upside chart
            dbc.Row(
                [
                    # pred graph title
                    html.H3("Top Revenue Opportunities", className="mt-3",
                        style={"margin-left": "50px", "marginTop": "190px", "color": "#DAA520",}
                    ),
                    
                    dbc.Col(
                        dcc.Loading(type="circle",
                                    children=dcc.Graph(id="opportunity_chart", config={"displaylogo": False})),
                        width=12, className="mt-3"
                    )
                ]
            ),

            html.Div(style={"height": "16px"}),

            # Footer
            html.Div([html.Span("made with ♥️ | "), html.Span(html.I("@aqxiao")), html.P("github.com/angie-xiao")],
                     className="text-center py-3",
                     style={"fontSize": "0.8em", "color": "#ac274f", "textAlign": "center",
                            "backgroundColor": "#f3f3f3", "margin": "40px auto 0 auto", "borderRadius": "6px"}),
        ],
        fluid=True,
    )

# _kpi_card: add id_subtext
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
        Output("opportunity_chart", "figure"),
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
                empty_fig,          # opportunity_chart.figure
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
        row = edf.loc[edf["product_key"] == product_key]
        if row.empty or pd.isna(row["ratio"].iloc[0]):
            elast_val, elast_subtext = "—", ""
        else:
            elast_val = f"{float(row['ratio'].iloc[0]):.2f}"
            pct_val = float(row["pct"].iloc[0]) if "pct" in row.columns else np.nan
            if np.isnan(pct_val):
                elast_subtext = ""
            else:
                pct_round = int(round(pct_val))
                top_share = max(1, 100 - pct_round)
                elast_subtext = f"Top ~{top_share}% among selected products"

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

        # ----- Opportunity chart
        opp_fig = _opportunity_chart(edf, best50_optimal_pricing_df, curr_price_df, all_gam_results)
        if opp_fig == {}:
            opp_fig = viz.empty_fig("No portfolio opportunities computed")

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
            upside_sub,           # 9  (new)
            badge,                # 10
            pred_graph,           # 11
            scenario_data,        # 12
            opp_fig,              # 13
            coverage,             # 14
        )



# ---------- Helpers (unchanged) ----------
    
def update_elasticity_kpi(product_key, elast_data):
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
    # Optional “Top X%” helper (e.g., 90th pct = Top 10%)
    top_share  = max(1, 100 - pct_round)

    subtext = f"Top ~{top_share}% among selected products"

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


def _upside_vs_current(curr_df, best50_df, product_key, all_gam):
    try:
        curr_price = curr_df.loc[curr_df["product_key"] == product_key, "current_price"].iloc[0]
        prod = all_gam[all_gam["product_key"] == product_key]
        if prod.empty:
            return "—"
        idx = (prod["asp"] - curr_price).abs().idxmin()
        rev_at_curr = prod.loc[idx, "revenue_pred_0.5"]
        best = best50_df[best50_df["product_key"] == product_key]
        if best.empty:
            return "—"
        rev_at_best = best["revenue_pred_0.5"].iloc[0]
        delta = rev_at_best - rev_at_curr
        pct = (delta / rev_at_curr * 100) if rev_at_curr else np.nan
        if pd.isna(pct):
            return f"{'+' if delta >= 0 else '−'}${abs(delta):,.0f}"
        sign = "+" if delta >= 0 else "−"
        return f"{sign}${abs(delta):,.0f} ({sign}{abs(pct):,.1f}%)"
    except Exception:
        return "—"

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
    if prod_df.empty:
        return ""
    def peak_x(col):
        try:
            return prod_df.loc[prod_df[col] == prod_df[col].max(), "asp"].iloc[0]
        except Exception:
            return np.nan
    p_low = peak_x("revenue_pred_0.025")
    p_mid = peak_x("revenue_pred_0.5")
    p_high = peak_x("revenue_pred_0.975")
    align_spread = np.nanmax([p_low, p_mid, p_high]) - np.nanmin([p_low, p_mid, p_high])
    if pd.isna(p_mid):
        density_score = 0
    else:
        tol = 0.05 * p_mid
        density_score = (prod_df["asp"].between(p_mid - tol, p_mid + tol)).mean()
    spread_score = np.exp(-align_spread / (0.1 * (p_mid if p_mid else 1))) if p_mid else 0
    score = 0.6 * spread_score + 0.4 * density_score
    label, color = ("Weak", "danger")
    if score >= 0.7:
        label, color = ("Strong", "success")
    elif score >= 0.45:
        label, color = ("Medium", "warning")
    return dbc.Badge(f"Confidence: {label}", color=color, pill=True, className="px-3 py-2")


def _elasticity_rank_table(elasticity_df: pd.DataFrame):
    if elasticity_df.empty:
        return []
    df = elasticity_df[["product", "ratio"]].copy()
    df["abs_ratio"] = df["ratio"].abs()
    df = df.sort_values("abs_ratio", ascending=False).drop(columns="abs_ratio").reset_index(drop=True)
    df["rank"] = df.index + 1
    try:
        df["ratio"] = df["ratio"].round(2)
    except Exception:
        pass
    return df[["rank", "product", "ratio"]].to_dict("records")


def _opportunity_chart(elast_df, best50_df, curr_df, all_gam):


    def _empty(title):
        fig = go.Figure()
        fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False))
        fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        return fig

    # guards
    for df in [elast_df, best50_df, curr_df, all_gam]:
        if df is None or getattr(df, "empty", True):
            return _empty("No data for opportunity chart")

    # must have product_key in all four
    if not all(("product_key" in df.columns) for df in [elast_df, best50_df, curr_df, all_gam]):
        return _empty("Missing product_key")

    # must have predictions
    if "revenue_pred_0.5" not in all_gam.columns or "revenue_pred_0.5" not in best50_df.columns:
        return _empty("Missing prediction columns")

    # ensure numeric where needed
    for frame, cols in [
        (all_gam, ["asp", "revenue_pred_0.5"]),
        (best50_df, ["revenue_pred_0.5"]),
        (curr_df, ["current_price"]),
        (elast_df, ["ratio"]),
    ]:
        for c in cols:
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors="coerce")

    # overlap on product_key (not product)
    # prods = sorted(
    #     set(all_gam["product_key"])
    #     & set(best50_df["product_key"])
    #     & set(curr_df["product_key"])
    # )
    
    # only use filtered top N prods
    prods = set(best50_df["product_key"])
    
    
    if not prods:
        return _empty("No overlapping products across inputs")

    rows = []
    for pk in prods:
        try:
            # current price
            cp = curr_df.loc[curr_df["product_key"] == pk, "current_price"]
            if cp.empty or pd.isna(cp.iloc[0]):
                continue
            curr_price = float(cp.iloc[0])

            # expected curve for this product
            prod = all_gam[
                (all_gam["product_key"] == pk)
                & pd.notna(all_gam["asp"])
                & pd.notna(all_gam["revenue_pred_0.5"])
            ]
            if prod.empty:
                continue

            # nearest ASP on expected curve
            idx = (prod["asp"] - curr_price).abs().idxmin()
            rev_curr = float(prod.loc[idx, "revenue_pred_0.5"])

            # revenue at recommended price (best50)
            rec = best50_df.loc[best50_df["product_key"] == pk]
            if rec.empty or pd.isna(rec["revenue_pred_0.5"].iloc[0]):
                continue
            rev_best = float(rec["revenue_pred_0.5"].iloc[0])

            # display label and elasticity (optional)
            label = curr_df.loc[curr_df["product_key"] == pk, "product"]
            label = label.iloc[0] if len(label) else pk

            e = elast_df.loc[elast_df["product_key"] == pk, "ratio"]
            elast_val = float(e.iloc[0]) if len(e) and pd.notna(e.iloc[0]) else np.nan

            rows.append({"product": label, "upside": rev_best - rev_curr, "elasticity": elast_val})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty("No computable upside")

    df = df.sort_values("upside", ascending=False).head(12)
    fig = px.bar(
        df, y="product", x="upside",
        hover_data=["elasticity"], height=380,
        # title="Upside vs Elasticity (Top Opportunities)"
    )
    fig.update_xaxes(title_text="Upside (Expected Revenue Δ)", tickprefix="$", separatethousands=True)
    fig.update_yaxes(title_text="")
    fig.update_traces(text=df["upside"].map(lambda x: f"${x:,.0f}"),
                      textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=60),
                      uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def _coverage_note(prod_df: pd.DataFrame) -> str:
    if prod_df.empty:
        return ""
    n_points = len(prod_df)
    within = ((prod_df["revenue_actual"] >= prod_df["revenue_pred_0.025"]) &
              (prod_df["revenue_actual"] <= prod_df["revenue_pred_0.975"])).mean()
    return f"Based on {n_points} historical points;\n{within*100:,.0f}% of actual revenue outcomes fall within the shown range."
