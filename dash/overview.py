# overview
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc


def layout(products):
    return dbc.Container(
        [
            html.H2("Overview", className="mt-3", style={"textAlign": "center"}),
            # Row: Dropdown (col 1) + 3 KPI cards (cols 2-4)
            dbc.Row(
                [
                    # Col 1: Dropdown
                    dbc.Col(
                        [
                            html.Label(
                                "Select a Product:",
                                style={"margin-left": "80px", "display": "block"},
                            ),
                            dcc.Dropdown(
                                id="product_dropdown_snap",
                                options=[{"label": p, "value": p} for p in products],
                                value=products[0] if products else None,
                                style={"width": "80%", "margin-left": "20px"},
                            ),
                        ],
                        width=3,
                    ),
                    # Col 2: Current Price
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="card_title_curr_price_snap",
                                        className="kpi-title",
                                        style={
                                            "color": "#121212",
                                            "textAlign": "center",
                                            "margin-bottom": "10px",
                                            "margin-top": "10px",
                                        },
                                    ),
                                    html.H2(
                                        "Current Price",
                                        className="kpi-eyebrow",
                                        style={
                                            "color": "#121212",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.H1(
                                        id="curr_price_snap",
                                        className="kpi-value",
                                        style={
                                            "color": "#DAA520",
                                            "textAlign": "center",
                                        },
                                    ),
                                ]
                            ),
                            style={"backgroundColor": "#f3f0f0"},
                        ),
                        width=3,
                        className="kpi-card",
                    ),
                    # Col 3: Recommended Price
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="card_title_snap",
                                        className="kpi-title",
                                        style={
                                            "color": "#121212",
                                            "textAlign": "center",
                                            "margin-bottom": "10px",
                                            "margin-top": "10px",
                                        },
                                    ),
                                    html.H2(
                                        "Rec. Price (P50)",
                                        className="kpi-eyebrow",
                                        style={
                                            "color": "#121212",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.H1(
                                        id="card_asp_snap",
                                        className="kpi-value",
                                        style={
                                            "color": "#DAA520",
                                            "textAlign": "center",
                                        },
                                    ),
                                ]
                            ),
                            style={"backgroundColor": "#F5E8D8"},
                        ),
                        width=3,
                        className="kpi-card",
                    ),
                    # Col 4: Elasticity
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="card_title_elasticity_snap",
                                        className="kpi-title",
                                        style={
                                            "color": "#121212",
                                            "textAlign": "center",
                                            "margin-bottom": "10px",
                                            "margin-top": "10px",
                                        },
                                    ),
                                    html.H2(
                                        "Elasticity",
                                        className="kpi-eyebrow",
                                        style={
                                            "color": "#121212",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.H1(
                                        id="elasticity_ratio_snap",
                                        className="kpi-value",
                                        style={
                                            "color": "#DAA520",
                                            "textAlign": "center",
                                        },
                                    ),
                                ]
                            ),
                            style={"backgroundColor": "#f3f0f0"},
                        ),
                        width=3,
                        className="kpi-card",
                    ),
                ],
                style={
                    "margin-left": "20px",
                    "margin-right": "80px",
                    "margin-top": "20px",
                },
                align="center",
                justify="center",
            ),
            html.Hr(className="my-4"),
            # Main: left (graph) / right (sticky explainer)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Loading(
                                type="circle",
                                children=dcc.Graph(
                                    id="gam_results_pred",
                                    style={
                                        "marginTop": "8px",
                                        "height": "600px",
                                    },  # taller graph
                                    config={"displaylogo": False},
                                ),
                            )
                        ],
                        md=8,
                        xs=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "How to read this",
                                            className="mb-2 fw-bold text-uppercase",
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    [
                                                        html.B("Goal: "),
                                                        "Pick the price where expected (mean) revenue is highest.",
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.B("How: "),
                                                        "Learn price & demand relationship from historical data ",
                                                        html.I("('x' markers)."),
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.B("Range: "),
                                                        "Conservative → optimistic scenarios shown by the band ",
                                                        html.I("(P2.5-P97.5)."),
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.B("Recommended price: "),
                                                        "Peak of the P50 curve ",
                                                        html.I("(red dot)."),
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.B("Robustness: "),
                                                        "Strongest when peaks of P2.5, P50, & P97.5 align.",
                                                    ]
                                                ),
                                            ],
                                            className="predictive-explainer-list",
                                        ),
                                    ]
                                ),
                                className="shadow-sm predictive-explainer",
                                style={
                                    "margin-top": "100px",
                                    "margin-right": "20px",
                                },  # center inside the col
                            )
                        ],
                        md=4,
                        xs=12,
                        className="explainer-col",
                    ),
                ],
                className="g-3 mb-4",
            ),
            # dbc.Row(
            #     [
            #         dbc.Col(
            #             [
            #                 dcc.Loading(
            #                     type="circle",
            #                     children=dcc.Graph(
            #                         id="snapshot_spark",  # <-- present in layout now
            #                         style={"height": "420px"},
            #                         config={"displaylogo": False},
            #                     ),
            #                 )
            #             ],
            #             md=10,
            #             xs=12,
            #             className="mx-auto",
            #         )
            #     ],
            #     className="mb-4",
            # ),
            # breathing room bottom
            html.Div(style={"height": "16px"}),
        ],
        fluid=True,
    )


def register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing_df,
    curr_price_df,
    elasticity_df,
    all_gam_results,
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
        Output("gam_results_pred", "figure"),
        Input("product_dropdown_snap", "value"),  # <-- fixed id here too
    )
    def update_snapshot(product):
        if not product:
            return ("", "", "", "", "", "", {})

        # Filter data
        filt_pricing = price_quant_df[price_quant_df["product"] == product]
        filt_opt = best50_optimal_pricing_df[
            best50_optimal_pricing_df["product"] == product
        ]
        curr = curr_price_df[curr_price_df["product"] == product]["current_price"]
        elast = elasticity_df[elasticity_df["product"] == product]["ratio"]

        # Titles
        title = product
        title_curr = product
        title_elast = product

        # Values (format safely)
        asp_val = "—"
        if len(filt_opt):
            try:
                asp_val = f"${float(filt_opt['asp'].iloc[0]):,.2f}"
            except Exception:
                asp_val = f"${filt_opt['asp'].iloc[0]}"

        curr_price_val = f"${float(curr.iloc[0]):,.2f}" if len(curr) else "—"
        try:
            elast_val = f"{float(elast.iloc[0]):.2f}" if len(elast) else "—"
        except Exception:
            elast_val = str(elast.iloc[0]) if len(elast) else "—"

        # # Descriptive figure
        # spark = viz.price_quantity(filt_pricing) if len(filt_pricing) else {}
        
        # pred figure
        filt = all_gam_results[all_gam_results["product"] == product]
        pred_graph = viz.gam_results(filt) if len(filt) else {}


        return (
            title_curr,
            curr_price_val,
            title,
            asp_val,
            title_elast,
            elast_val,
            # spark,
            pred_graph
        )

