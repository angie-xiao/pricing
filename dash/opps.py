# + for the table add in "number of days an asp was valid"

# opps.py

from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table import FormatTemplate
import plotly.express as px
import plotly.graph_objects as go

# --- spacing constants (tweak here) ---
LEFT_INSET = "50px"  # align table & chart title
GAP_H1_TO_TABLE = "32px"  # space after H1 before table
GAP_TABLE_TO_H3 = "36px"  # space between table and chart title
GAP_H3_TO_GRAPH = "8px"  # space between chart title and chart


def _build_cols(df: pd.DataFrame):
    """ Build columns for dash DataTable with formatting."""
    
    df = df[[
        'asin', 'product', 'asp', 'days_sold', 
        'shipped_units', 'daily_units', 'revenue_actual', 'daily_rev',
         'revenue_pred_0.025', 'revenue_pred_0.5', 'revenue_pred_0.975', 
         'pred_0.025', 'pred_0.5', 'pred_0.975' 
    ]]
    
    integer_columns = ['days_sold', 'shipped_units', 'daily_units', 'days_asp_valid']
    money_columns = ['asp', 'revenue_actual', 'daily_rev', 'revenue_pred_0.025', 
                    'revenue_pred_0.5', 'revenue_pred_0.975', 'pred_0.025', 
                    'pred_0.5', 'pred_0.975']
    
    cols = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        for c in df.columns:
            if c in integer_columns:
                cols.append({
                    "name": c.replace("_", " ").title(),
                    "id": c,
                    "type": "numeric",
                    "format": {"specifier": ",d"}  # Format as integer with thousands separator
                })
            elif c in money_columns:
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
    data = (
        opp_table_df.to_dict("records")
        if isinstance(opp_table_df, pd.DataFrame)
        else []
    )

    # Inner content with consistent left/right inset
    inner = html.Div(
        [
            # TABLE TITLE
            html.H3(
                "Table Breakdown",
                style={
                    "color": "#DAA520",
                    "marginTop": "60px",  # more space from table
                    "marginBottom": "30px",
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
                    {
                        "if": {"column_id": "revenue_gap"},
                        "backgroundColor": "#FFFBE9",
                        "color": "black",
                    },
                    {
                        "if": {"column_id": "price_gap"},
                        "backgroundColor": "#FFFBE9",
                        "color": "black",
                    },
                ],
            ),
            # CHART TITLE
            html.H3(
                "Bar Graph",
                style={
                    "color": "#DAA520",
                    "marginTop": "100px",  # more space from table
                    # "marginBottom": GAP_H3_TO_GRAPH,  # tight to graph
                },
            ),
            # CHART
            dcc.Loading(
                type="circle",
                children=dcc.Graph(
                    id="opportunity_chart", config={"displaylogo": False}
                ),
            ),
        ],
        style={
            "marginLeft": LEFT_INSET,
            "marginRight": LEFT_INSET,
            "marginTop": "30px",
        },
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
                fluid=True,
                className="px-3",
                style={"maxWidth": "1500px", "margin": "0 auto"},
            ),
            # Full-width footer (stretched)
            html.Footer(
                dbc.Container(
                    [
                        html.Span("made with ♥️ | "),
                        html.Span(html.I("@aqxiao")),
                        html.P("github.com/angie-xiao", className="mb-0"),
                    ],
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


def register_callbacks(app, opp_inputs, opp_table_df: pd.DataFrame, viz_cls):
    v = viz_cls()  # instantiate with default template ("lux")

    @app.callback(
        Output("opp_table", "data"),
        Input("opp_table", "id"),
        prevent_initial_call=False,
    )
    def _fill_table(_):
        return opp_table_df.to_dict("records")

    @app.callback(
        Output("opportunity_chart", "figure"),
        Input("opportunity_chart", "id"),
        prevent_initial_call=False,
    )
    def _fill_chart(_):
        return v.opportunity_chart(
            opp_inputs["elasticity_df"],
            opp_inputs["best50_optimal_pricing_df"],
            opp_inputs["curr_price_df"],
            opp_inputs["all_gam_results"],
        )
