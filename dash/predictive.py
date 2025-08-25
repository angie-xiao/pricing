# predictive.py
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc


def layout(products):
    return dbc.Container(
        [
            html.H2(
                "Predictive Modeling", className="mt-3", style={"textAlign": "center"}
            ),
            # Top: product dropdown
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Select a Product:",
                                className="form-label",
                                style={
                                    "textAlign": "center",
                                    "margin-top": "20px",
                                    "margin-left": "110px",
                                },
                            ),
                            dcc.Dropdown(
                                id="product_dropdown_pred",
                                options=[{"label": p, "value": p} for p in products],
                                value=products[0] if products else None,
                                style={
                                    "width": "350px",  # wider dropdown
                                    "margin": "0 auto",  # keep centered
                                },
                            ),
                        ],
                        md="auto",
                    )
                ],
                justify="center",
                className="mt-2 mb-3",
            ),
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
            # footnote
            dbc.Container(
                html.Div(
                    [
                        html.Span("made with ♥️ | "),
                        html.Span(html.I("@aqxiao")),
                        html.P("github.com/angie-xiao"),
                    ],
                    className="text-center py-3",
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
                fluid=True,  # <— spans full viewport width
                className="px-0",  # <— no left/right padding
            ),
        ],
        fluid=True,
    )


def register_callbacks(app, all_gam_results, viz_cls):
    viz = viz_cls()

    @app.callback(
        Output("gam_results_pred", "figure"),
        Input("product_dropdown_pred", "value"),
    )
    def update_gam(product):
        if not product:
            return {}
        filt = all_gam_results[all_gam_results["product"] == product]
        return viz.gam_results(filt) if len(filt) else {}
