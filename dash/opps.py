# opps.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from helpers import OppsTable,Style   # import locally

OPP_LEFT_INSET = Style.OPP_LEFT_INSET
OPP_GAP_H1_TO_TABLE = Style.OPP_GAP_H1_TO_TABLE
OPP_GAP_TABLE_TO_H3 = Style.OPP_GAP_TABLE_TO_H3
OPP_GAP_H3_TO_GRAPH = Style.OPP_GAP_H3_TO_GRAPH


def layout(opp_table_df: pd.DataFrame):
    # make sure 'days_asp_valid' exists
    table_df = OppsTable.ensure_days_valid_column(opp_table_df)

    # build columns
    cols = OppsTable.build_opp_table_columns(table_df)

    data = table_df.to_dict("records") if isinstance(table_df, pd.DataFrame) else []

    # Inner content with consistent left/right inset
    inner = html.Div(
        [
            # TABLE TITLE
            html.H3(
                "Table Breakdown",
                style={
                    "color": "#DAA520",
                    "marginTop": OPP_GAP_H1_TO_TABLE,
                    "marginBottom": "30px",
                },
            ),
            # TABLE
            dash_table.DataTable(
                id="opp_table",
                columns=cols,
                data=data,
                sort_action="native",
                page_size=25,
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
                    "marginTop": OPP_GAP_TABLE_TO_H3,
                    # "marginBottom": OPP_GAP_H3_TO_GRAPH,
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
            "marginLeft": OPP_LEFT_INSET,
            "marginRight": OPP_LEFT_INSET,
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
            # Footer
            html.Div(
                [
                    html.Div(
                        [html.Span("made with ♥️ | "), html.Span(html.I("@aqxiao"))],
                        style={"marginBottom": "4px"},
                    ),
                    html.A(
                        "github.com/angie-xiao",
                        href="https://github.com/angie-xiao",
                        style={"textDecoration": "none", "color": "#ac274f"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",  # horizontal center
                    "justifyContent": "center",  # vertical center
                    "height": "80px",
                    "fontSize": "0.8em",
                    "color": "#ac274f",
                    "backgroundColor": "#f3f3f3",
                    "borderRadius": "0",
                    "width": "100%",
                    "marginTop": "40px",
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
        df = OppsTable.ensure_days_valid_column(opp_table_df)
        return df.to_dict("records")

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
