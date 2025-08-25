# descriptive
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc


def layout(products):
    return dbc.Container(
        [
            html.H2("Descriptive Analyses", className="mt-3", style={"textAlign": "center"}),
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
            # Descriptive graph below (with loader)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Loading(
                                type="circle",
                                children=dcc.Graph(
                                    id="snapshot_spark",  # <-- present in layout now
                                    style={"height": "420px"},
                                    config={"displaylogo": False},
                                ),
                            )
                        ],
                        md=10,
                        xs=12,
                        className="mx-auto",
                    )
                ],
                className="mb-4",
            ),
            # breathing room bottom
            html.Div(style={"height": "16px"}),
        ],
        fluid=True,
    )


def register_callbacks(
    app,
    price_quant_df,
    viz_cls,
):
    viz = viz_cls()
    
    @app.callback(
        Output("snapshot_spark", "figure", ),
        Input("product_dropdown_snap", "value")
    )
    def update_descriptive_snapshot(product):
        if not product:
            return {}
        
        filt_pricing = price_quant_df[price_quant_df["product"] == product]
        return viz.price_quantity(filt_pricing) if len(filt_pricing) else {}

