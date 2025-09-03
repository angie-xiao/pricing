from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table import FormatTemplate


def layout(opp_table_df):
    # format numbers
    if isinstance(opp_table_df, pd.DataFrame):
        cols = []
        for c in opp_table_df.columns:
            if pd.api.types.is_numeric_dtype(opp_table_df[c]):
                cols.append({
                    "name": c.replace("_", " ").title(),
                    "id": c,
                    "type": "numeric",
                    "format": FormatTemplate.money(2)  # $ + commas + 2 decimals
                })
            else:
                cols.append({"name": c.replace("_", " ").title(), "id": c})
        data = opp_table_df.to_dict("records")

    # Page = content container (max width) + full-bleed footer
    return html.Div(
        [
            # ---- Content (constrained width) ----
            dbc.Container(
                [
                    html.H1(
                        "Top Opportunities",
                        className="display-5",
                        style={"textAlign": "center", "padding": "58px 0 8px"},
                    ),
                    dbc.Row(
                        dbc.Col(
                            dash_table.DataTable(
                                id="opp_table",
                                columns=cols,
                                data=data,
                                sort_action="native",
                                page_size=12,
                                style_table={
                                    "width": "100%",
                                    "overflowX": "auto",
                                    "border": "none"
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
                                    {
                                        "if": {"column_id": "product"},
                                        "fontWeight": "bold",
                                    },
                                    {
                                        "if": {"column_id": "revenue_gap"},
                                        "backgroundColor": "#FFFBE9",
                                        "color": "black",       # ensure text stays readable
                                    },
                                    {
                                        "if": {"column_id": "price_gap"},
                                        "backgroundColor": "#FFFBE9",
                                        "color": "black",
                                    },
                                ],
                            ),
                            width=12, lg=10, className="mx-auto",
                        ),
                    ),
                    html.Div(style={"height": "16px"}),
                ],
                fluid=True,
                className="px-3",
                style={"maxWidth": "1500px", "margin": "0 auto"},
            ),

            # ---- Full-width footer (background spans to edges) ----
            html.Footer(
                dbc.Container(
                    [
                        html.Span("made with ♥️ | "),
                        html.Span(html.I("@aqxiao")),
                        html.P("github.com/angie-xiao", className="mb-0"),
                    ],
                    className="text-center py-3",  # center text, vertical padding
                ),
                className="w-100",  # make footer full width
                style={
                    "font-size": "0.8em",
                    "color": "#ac274f",
                    "textAlign": "center",
                    "background-color": "#f3f3f3",
                    "margin": "40px auto 0 auto",  # <-- top margin added (40px); auto keeps centered
                    "borderRadius": "6px",
                    # "width":"100%"
                },
            ),
        ]
    )


def register_callbacks(app, opp_table_df):
    @app.callback(
        Output("opp_table", "data"),
        Input("opp_table", "id"),   # dummy input so callback runs once
        prevent_initial_call=False
    )
    def _fill_table(_):
        # convert df to records for DataTable
        return opp_table_df.to_dict("records")