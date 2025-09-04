# opps.py

from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table import FormatTemplate
import plotly.express as px
import plotly.graph_objects as go

# --- spacing constants (tweak here) ---
LEFT_INSET = "50px"         # align table & chart title
GAP_H1_TO_TABLE = "32px"    # space after H1 before table
GAP_TABLE_TO_H3 = "36px"    # space between table and chart title
GAP_H3_TO_GRAPH = "8px"     # space between chart title and chart

def _build_cols(df: pd.DataFrame):
    cols = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append({
                    "name": c.replace("_", " ").title(),
                    "id": c,
                    "type": "numeric",
                    "format": FormatTemplate.money(2)
                })
            else:
                cols.append({"name": c.replace("_", " ").title(), "id": c})
    return cols


def layout(opp_table_df: pd.DataFrame):
    cols = _build_cols(opp_table_df)
    data = opp_table_df.to_dict("records") if isinstance(opp_table_df, pd.DataFrame) else []

    # Inner content with consistent left/right inset
    inner = html.Div(
        [

            # TABLE TITLE
            html.H3(
                "Revenue Prediction vs Actual Breakdown",
                style={
                    "color": "#DAA520",
                    "marginTop": "100px",    # more space from table
                    # "marginBottom": GAP_H3_TO_GRAPH,  # tight to graph
                },
            ),

            # TABLE
            dash_table.DataTable(
                id="opp_table",
                columns=cols,
                data=data,
                sort_action="native",
                page_size=25,  # <<< longer table
                style_table={
                    "width": "100%",
                    "overflowX": "auto",
                    "border": "none",
                },
                style_cell={
                    "padding": "12px",
                    "fontSize": "14px",
                    "border": "none",
                    "textAlign": "center",
                },
                style_header={
                    "fontWeight": 600,
                    "border": "none",
                    "backgroundColor": "#F5EFE7",
                    "textAlign": "center",
                },
                style_data_conditional=[
                    {"if": {"column_id": "product"}, "fontWeight": "bold"},
                    {"if": {"column_id": "revenue_gap"}, "backgroundColor": "#FFFBE9", "color": "black"},
                    {"if": {"column_id": "price_gap"}, "backgroundColor": "#FFFBE9", "color": "black"},
                ],
            ),

            # CHART TITLE
            html.H3(
                "Revenue Potential by Product",
                style={
                    "color": "#DAA520",
                    "marginTop": "100px",    # more space from table
                    # "marginBottom": GAP_H3_TO_GRAPH,  # tight to graph
                },
            ),

            # CHART
            dcc.Loading(
                type="circle",
                children=dcc.Graph(id="opportunity_chart", config={"displaylogo": False})
            ),
        ],
        style={"marginLeft": LEFT_INSET, "marginRight": LEFT_INSET, "marginTop": "30px"},
    )

    return html.Div(
        [
            dbc.Container(
                [
                    html.H1(
                        "Top Revenue Opportunities",
                        className="display-5",
                        style={"textAlign": "center", "padding": "58px 0 8px"},
                    ),
                    inner,
                    html.Div(style={"height": "16px"}),
                ],
                fluid=True, className="px-3",
                style={"maxWidth": "1500px", "margin": "0 auto"},
            ),
            # Full-width footer (stretched)
            html.Footer(
                dbc.Container(
                    [html.Span("made with ♥️ | "),
                     html.Span(html.I("@aqxiao")),
                     html.P("github.com/angie-xiao", className="mb-0")],
                    className="text-center py-3",
                ),
                style={
                    "width": "100%",
                    "backgroundColor": "#f3f3f3",
                    "color": "#ac274f",
                    "textAlign": "center",
                    "marginTop": "40px",
                    "padding": "10px 0",
                },
            ),
        ]
    )


def _empty(title):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    return fig


def _build_opp_chart(elast_df, best50_df, curr_df, all_gam):
    """ rev chart """
    for df in [elast_df, best50_df, curr_df, all_gam]:
        if df is None or getattr(df, "empty", True):
            return _empty("No data for opportunity chart")
    if not all(("product_key" in df.columns) for df in [elast_df, best50_df, curr_df, all_gam]):
        return _empty("Missing product_key")
    if "revenue_pred_0.5" not in all_gam.columns or "revenue_pred_0.5" not in best50_df.columns:
        return _empty("Missing prediction columns")

    # ensure numeric
    for frame, cols in [
        (all_gam, ["asp", "revenue_pred_0.5"]),
        (best50_df, ["revenue_pred_0.5"]),
        (curr_df, ["current_price"]),
        (elast_df, ["ratio"]),
    ]:
        for c in cols:
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors="coerce")

    prods = set(best50_df["product_key"])
    if not prods:
        return _empty("No overlapping products across inputs")

    rows = []
    for pk in prods:
        try:
            cp = curr_df.loc[curr_df["product_key"] == pk, "current_price"]
            if cp.empty or pd.isna(cp.iloc[0]): 
                continue
            curr_price = float(cp.iloc[0])

            prod = all_gam[
                (all_gam["product_key"] == pk) &
                pd.notna(all_gam["asp"]) &
                pd.notna(all_gam["revenue_pred_0.5"])
            ]
            if prod.empty:
                continue

            idx = (prod["asp"] - curr_price).abs().idxmin()
            rev_curr = float(prod.loc[idx, "revenue_pred_0.5"])

            rec = best50_df.loc[best50_df["product_key"] == pk]
            if rec.empty or pd.isna(rec["revenue_pred_0.5"].iloc[0]):
                continue
            rev_best = float(rec["revenue_pred_0.5"].iloc[0])

            label = curr_df.loc[curr_df["product_key"] == pk, "product"]
            label = label.iloc[0] if len(label) else pk

            e = elast_df.loc[elast_df["product_key"] == pk, "ratio"]
            elast_val = float(e.iloc[0]) if len(e) and pd.notna(e.iloc[0]) else None

            rows.append({"product": label, "upside": rev_best - rev_curr, "elasticity": elast_val})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty("No computable upside")

    df = df.sort_values("upside", ascending=True).head(12)
    fig = px.bar(df, y="product", x="upside", hover_data=["elasticity"], height=420)
    fig.update_xaxes(title_text="Expected Revenue Δ", tickprefix="$", separatethousands=True)
    fig.update_yaxes(title_text="")
    fig.update_traces(text=df["upside"].map(lambda x: f"${x:,.0f}"), textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=60), uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def register_callbacks(app, opp_inputs, opp_table_df: pd.DataFrame):
    @app.callback(Output("opp_table", "data"), Input("opp_table", "id"), prevent_initial_call=False)
    def _fill_table(_):
        return opp_table_df.to_dict("records")

    @app.callback(Output("opportunity_chart", "figure"), Input("opportunity_chart", "id"), prevent_initial_call=False)
    def _fill_chart(_):
        return _build_opp_chart(
            opp_inputs["elasticity_df"],
            opp_inputs["best50_optimal_pricing_df"],
            opp_inputs["curr_price_df"],
            opp_inputs["all_gam_results"],
        )
