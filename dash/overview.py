# overview
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc


def layout(products):
    return dbc.Container(
        [
            html.H2(
                "Overview",
                className="mt-3",
                style={
                    "textAlign": "center",
                    "padding": "20px 0"
                },
            ),

            # Col 1: Dropdown (centered)
            # Row: Label + Dropdown (side by side, centered)
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            "Select a Product:",
                            style={"fontWeight": "600", "textAlign": "right", "marginRight": "10px"},
                        ),
                        width="auto",   # shrink to fit label
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="product_dropdown_snap",
                            options=[{"label": p, "value": p} for p in products],
                            value=products[0] if products else None,
                            style={"width": "250px"},  # fixed width looks cleaner
                        ),
                        width="auto",   # shrink to fit dropdown
                    ),
                ],
                justify="center",  # centers the two columns together
                align="center",    # vertical alignment
                # className="my-4",
                style={ "padding": "20px 0" },
            ),


            # KPI cards row (centered)
            dbc.Row(
                [
                    # Col 1: Current Price
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
                                        style={"color": "#121212", "textAlign": "center"},
                                    ),
                                    html.H1(
                                        id="curr_price_snap",
                                        className="kpi-value",
                                        style={"color": "#DAA520", "textAlign": "center"},
                                    ),
                                ]
                            ),
                            style={"backgroundColor": "#f3f0f0", "padding": "20px 0"},
                        ),
                        width=3,
                        className="kpi-card",
                    ),

                    # Col 2: Recommended Price
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
                                        style={"color": "#121212", "textAlign": "center"},
                                    ),
                                    html.H1(
                                        id="card_asp_snap",
                                        className="kpi-value",
                                        style={"color": "#DAA520", "textAlign": "center"},
                                    ),
                                ]
                            ),
                            style={"backgroundColor": "#F5E8D8", "padding": "20px 0"},
                        ),
                        width=3,
                        className="kpi-card",
                    ),

                    # Col 3: Elasticity
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
                                        style={"color": "#121212", "textAlign": "center"},
                                    ),
                                    html.H1(
                                        id="elasticity_ratio_snap",
                                        className="kpi-value",
                                        style={"color": "#DAA520", "textAlign": "center"},
                                    ),
                                ]
                            ),
                            style={"backgroundColor": "#f3f0f0", "padding": "20px 0"},
                        ),
                        width=3,
                        className="kpi-card",
                    ),
                ],
                justify="center",   # centers the set of cols in the row
                align="center",
                className="g-5",   # spacing between cols and top/bottom margin,
                style={ "padding": "20px 0" },

            ),


            html.Hr(className="my-5"),

            # Main: left (pred graph) / right (sticky explainer)
            dbc.Row(
                [
                    html.H2(
                        "Revenue Prediction with GAM",
                        style={
                            # "padding": "20px 0",
                            "color": "#DAA520",
                            "margin-left": "50px",
                            "margin-right": "30px",
                            "margin-bottom": "0px",
                            "font-size": "20px",
                        },
                    ),
                    html.Span(
                        "Price-Revenue Scenarios: Conservative, Expected, and Optimistic Projections",
                        style={
                            "color": "#5f6b7a",
                            "margin-left": "50px",
                            "display": "block",  # ensures margin-top works
                        },
                    ),
                    # pred graph
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
                    # sticky note
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
            html.Hr(className="my-5"),

            # ------- Elasticity Section -------
            # Header (full width)
            dbc.Row(
                [
                    html.H2(
                        "Top Products Elasticity",
                        style={
                            "color": "#DAA520",
                            "margin-left": "50px",
                            "margin-bottom": "0px",
                            "font-size": "20px",
                        },
                    ),
                    html.Span(
                        "Elasticity = % change in demand / % change in price",
                        style={"color": "#5f6b7a", "margin-left": "50px", "display": "block"},
                    ),
                ],
                className="g-3 mb-4",
            ),

            # Content Row: left = histogram, right = table
            dbc.Row(
                [
                    # elasticity graph
                    dbc.Col(
                        dcc.Loading(
                            type="circle",
                            children=dcc.Graph(
                                id="elast_dist",
                                style={"marginTop": "8px", "height": "600px", "width": "50%"},
                                config={"displaylogo": False},
                            ),
                        ),
                        md=8, xs=12, className="mb-3 pe-md-4",
                    ),

                    # elasticity table
                    dbc.Col(
                        dash_table.DataTable(
                            id="elast_table",
                            columns=[
                                {"name": "Product", "id": "product"},
                                {"name": "Elasticity", "id": "ratio", "type": "numeric"},
                            ],
                            data=[],
                            sort_action="native",
                            page_size=10,
                            style_table={
                                "height": "600px",
                                "overflowY": "auto",
                                "marginTop": "50px",
                                "width": "100%",
                                "border": "none",
                                "marginLeft":"80px"
                            },
                            style_cell={
                                "padding": "12px",
                                "fontSize": "14px",
                                "border": "none",
                                "textAlign": "center",     # center text
                            },
                            style_header={
                                "fontWeight": 600,
                                "border": "none",
                                "backgroundColor": "#e8e3e3",
                                "textAlign": "center",     # center header text too
                            },
                        ),
                        md=3,
                        xs=12,
                        # className="ps-md-15",   # <-- padding for left & right
                    ),
                ]),
            # breathing room bottom
            html.Div(style={"height": "16px"}),
            
            # footnote
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
        Output("elast_dist", "figure"),
        Output("elast_table", "data"),            
        Input("product_dropdown_snap", "value"),  # 
    )
    def overview(product):
        if not product:
            return ("", "", "", "", "", "", {}, {}, [])

        # Filter data
        filt_pricing = price_quant_df[price_quant_df["product"] == product]
        filt_elasticity_df = elasticity_df.copy()  # all products for the table + hist
        filt_opt = best50_optimal_pricing_df[
            best50_optimal_pricing_df["product"] == product
        ]

        # --------------------- KPI CARDS ---------------------
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

        # --------------------- PRED GRAPH ---------------------
        # pred figure
        filt = all_gam_results[all_gam_results["product"] == product]
        pred_graph = viz.gam_results(filt) if len(filt) else {}

        # ------------------- ELASTICITY DIST -------------------
        elast_dist_graph = (
            viz.elast_dist(filt_elasticity_df) if len(filt_elasticity_df) else {}
        )

        # --- Elasticity table (all products, numeric rounded) ---
        table_df = filt_elasticity_df[["product", "ratio"]].copy()
        # sort by absolute elasticity descending so the most sensitive are on top
        table_df["abs_ratio"] = table_df["ratio"].abs()
        table_df = table_df.sort_values("abs_ratio", ascending=False).drop(columns="abs_ratio")
        # round for display
        try:
            table_df["ratio"] = table_df["ratio"].round(2)
        except Exception:
            pass
        elast_table_data = table_df.to_dict("records")

        return (
            title_curr,
            curr_price_val,
            title,
            asp_val,
            title_elast,
            elast_val,
            pred_graph,
            elast_dist_graph,
            elast_table_data,   
        )

