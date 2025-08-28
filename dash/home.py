# --- Explainer Homepage for Dash app ---
# Requires: dash, dash_bootstrap_components as dbc

from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
import warnings
from navbar import get_navbar

warnings.filterwarnings("ignore")

# --- Colors & styles (tweak to taste) ---
ACCENT = {
    "padding": "24px 0",
    "color": "#DAA520",
    "margin-left": "50px",
    "margin-right": "50px",
}
MUTED = {"color": "#5f6b7a", "margin-left": "50px", "margin-right": "50px"}

section_style = {
    "padding": "14px 0",
    "margin-left": "50px",
    "margin-right": "50px",
    # "margin":"50px"
}

card_style = {
    "border": "1px solid #e9eef5",
    "borderRadius": "14px",
    "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
    "margin-left": "50px",
    "margin-right": "150px",
}


# -----------------------
# Components (Explainers)
# -----------------------


def hero_section():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1(
                                "Optima",
                                className="display-5",
                                style={"textAlign": "center"},
                            ),
                            html.P(
                                'optimal price finder for revenue optimization', 
                                style={ "textAlign": "center", **MUTED}
                            ),
                        ]
                    )
                ]
            )
        ],
        fluid=True,
        style={"padding": "58px 0 8px"},
    )


def problem_objective_section():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "(1) Problem Statement",
                                style={
                                    "padding": "10px 0",
                                    "color": "#DAA520",
                                    "margin-left": "50px",
                                    "margin-right": "30px",
                                },
                            ),
                            html.P(
                                "The objective of this product is to provide a clear, evidence-based recommended price "
                                "that maximizes expected revenue while keeping assumptions transparent.",
                                style={
                                    "color": "#5f6b7a",
                                    "margin-left": "100px",
                                    # "margin-right": "50px",
                                },
                            ),
                            dbc.ListGroup(
                                [
                                    dbc.ListGroupItem(
                                        [
                                            html.Strong("Challenge: "),
                                            "Price changes impact both units and margin in non-linear ways.",
                                        ]
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.Strong("Risk: "),
                                            "Overpricing reduces volume; underpricing leaves money on the table.",
                                        ]
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.Strong("Goal: "),
                                            "Find a defensible sweet spot that grows revenue and is explainable.",
                                        ]
                                    ),
                                ],
                                className="md-7",
                                style={
                                    "border": "1px solid #e9eef5",
                                    "borderRadius": "14px",
                                    "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                    "margin-left": "100px",
                                    "margin-right": "150px",
                                    'width':"50%",
                                    "margin-bottom":"50px"
                                },
                            ),
                        ],
                        # md=7,
                    )
                ]
            )
        ],
        fluid=True,
        style=section_style,
    )


def abstract_section():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "(2) Solution",
                                style={
                                    "padding": "20px 0",
                                    "color": "#DAA520",
                                    "margin-left": "50px",
                                    "margin-right": "30px",
                                },
                            ),
                            html.P(
                                "Provide your data in the right folder - the app will do the rest:",
                                style={
                                    "color": "#5f6b7a",
                                    "margin-left": "100px",
                                    # "margin-right": "50px",
                                },
                            ),
                            dbc.ListGroup(
                                [
                                    dbc.ListGroupItem(
                                        [
                                            html.Strong("Data Engineering: "),
                                            "raw inputs transformed into clean features",
                                        ]
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.Strong("Modeling: "),
                                            "revenue curves fitted, elasticity estimated",
                                        ]
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.Strong("Visualization: "),
                                            "clear price-revenue tradeoffs displayed",
                                        ]
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.Strong("Recommendations: "),
                                            "optimal price point at average (or expected) revenue, plus conservative and optimistic alternatives",
                                        ]
                                    ),
                                ],
                                className="md-7",
                                style={
                                    "border": "1px solid #e9eef5",
                                    "borderRadius": "14px",
                                    "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                    "margin-left": "100px",
                                    "margin-right": "150px",
                                    'width':"50%"
                                },
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Alert(
                                            [
                                                html.Strong("💡 In short: "),
                                                "this is your one-stop automated pricing insights tool from raw data to decision-ready outputs.",
                                            ],
                                            color="light",
                                        ),
                                        md=12,
                                    )
                                ],
                                className="mt-3",
                                style={
                                    "margin-left": "100px",
                                    "padding": "14px 0",
                                    "width":"55%"
                                },
                            ),
                        ],
                        # md=7,
                    )
                ]
            )
        ],
        fluid=True,
        style=section_style,
    )


def methodology_section():
    return dbc.Container(
        [
            html.Div(id="method"),
            html.H3("(3) Methodology", style=ACCENT),
            # Four step cards in one row
            dbc.Row(
                [
                    # step 1
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Step 1: You provide sales data"),
                                        html.P(
                                            "We use past prices, units, revenue, and context like seasonality and promos.",
                                            style={
                                                "color": "#5f6b7a",
                                                "margin-left": "10px",
                                                "margin-right": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                "margin-left": "100px",
                                # "margin-right": "30px"
                            },
                        ),
                        md=4,
                    ),
                    # step 2
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Step 2: Data engineering"),
                                        html.P(
                                            "The app's built-in logic transforms raw inputs into clean, model-ready features.",
                                            style={
                                                "color": "#5f6b7a",
                                                "margin-left": "10px",
                                                "margin-right": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                "margin-left": "50px",
                                "margin-right": "50px",
                            },
                        ),
                        md=4,
                    ),
                    # step 3
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Step 3: ML model trains"),
                                        html.P(
                                            "A flexible curve (GAM) captures how demand changes with price.",
                                            style={
                                                "color": "#5f6b7a",
                                                "margin-left": "10px",
                                                "margin-right": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                # "margin-left": "50px",
                                "margin-right": "120px",
                            },
                        ),
                        md=4,
                    ),
                    # step 4
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Step 4: We pick the sweet spot"),
                                        html.P(
                                            "From the learned curve we choose the price expected to maximize revenue.",
                                            style={
                                                "color": "#5f6b7a",
                                                "margin-left": "10px",
                                                "margin-right": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                "margin-left": "100px",
                                "margin-top": "30px",
                                # "margin-right": "120px",
                                # "padding": "14px 0",
                            },
                        ),
                        md=4,
                    ),
                ],
                # justify="center",
                className="g-4 mt-2",
            ),
            # Why this works row (aligned with same margins as the cards)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Alert(
                            [
                                html.Strong("💡 Why this works: "),
                                "It balances higher margins at higher prices with potentially lower unit sales, and vice-versa.",
                            ],
                            color="light",
                        ),
                        md=12,
                    )
                ],
                className="mt-3",
                style={
                    "margin-left": "100px",
                    "padding": "14px 0",
                    "width":"60%"
                },
            ),
        ],
        fluid=True,
        style=section_style,
    )


def definitions_section():
    return dbc.Container(
        [
            html.H3("(4) Key Definitions", style=ACCENT),
            dbc.Row(
                [
                    # rec price
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Recommended Price"),
                                        html.P(
                                            "The price point on the central curve where the model "
                                            "expects revenue to be highest on average.", 
                                            style={
                                                    "color": "#5f6b7a",
                                                    "margin-left": "10px",
                                                    "margin-right": "10px",
                                                    "margin-top": "20px"
                                                },
                                        ),
                                        html.Br() # placeholder
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                "margin-left": "100px",
                                # "margin-right": "30px"
                            },
                        ),
                        md=4,
                    ),
                    # p2.5, p97.5
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Alternative Scenarios"),
                                        html.P(
                                            "Conservative and optimistic cases that show how revenue "
                                            "could shift if demand reacts differently to price changes.",

                                            style={
                                                "color": "#5f6b7a",
                                                "margin-left": "10px",
                                                "margin-right": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                "margin-left": "50px",
                                "margin-right": "50px",
                            },
                        ),
                        md=4,
                    ),
                    # robustness
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Robustness"),
                                        html.P(
                                            "The recommendation is strongest when the central, "
                                            "conservative, and optimistic curves all peak around "
                                            "the same price point.",
                                            style={
                                                "color": "#5f6b7a",
                                                "margin-left": "10px",
                                                "margin-right": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                # "margin-left": "50px",
                                "margin-right": "120px",
                            },
                        ),
                        md=4,
                    ),
                    # elasticity
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Elasticity"),
                                        html.P(
                                            "How sensitive customer demand is to price changes — "
                                            "whether shoppers react strongly or only slightly "
                                            "when prices move.",
                                            style={
                                                "color": "#5f6b7a",
                                                "margin-left": "10px",
                                                "margin-right": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "border": "1px solid #e9eef5",
                                "borderRadius": "14px",
                                "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
                                "margin-left": "100px",
                                "margin-top": "30px",
                                # "margin-right": "120px",
                                # "padding": "14px 0",
                            },
                        ),
                        md=4,
                    ),
                ]
            ),
        ],
        fluid=True,
        style=section_style,
    )


def suggestions_section():
    return dbc.Container(
        [
            html.H3("(4) Suggestions & FAQs", style=ACCENT),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Suggestions for use"),
                                dbc.CardBody(
                                    html.Ul(
                                        [
                                            html.Li(
                                                "Use the recommendation as a starting point; pair with business context."
                                            ),
                                            html.Li(
                                                "If confidence is low, consider running a small A/B before rolling out broadly."
                                            ),
                                            html.Li(
                                                "Re-heck after promotions or major events; seasonality can shift the sweet spot."
                                            ),
                                        ]
                                    )
                                ),
                            ],
                            style=card_style,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("FAQs"),
                                dbc.CardBody(
                                    html.Div(
                                        [
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        "What if my category is highly seasonal?"
                                                    ),
                                                    html.P(
                                                        "Seasonality is part of the learning process; review the ‘confidence’ and refresh data after key events."
                                                    ),
                                                ]
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        "Is the recommended price always revenue‑maximizing?"
                                                    ),
                                                    html.P(
                                                        "By default yes (P50). You can switch objectives (e.g., margin‑weighted) in the dashboard if available."
                                                    ),
                                                ]
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary("Can I override it?"),
                                                    html.P(
                                                        "Absolutely. The goal is a defensible starting point, not a mandate."
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                ),
                            ],
                            style=card_style,
                        ),
                        md=6,
                    ),
                ],
                className="mt-2",
            ),
        ],
        fluid=True,
        style=section_style,
    )


def footnote():
    return dbc.Container(
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
    )


# --- Compose layout ---
def Homepage():
    """ """
    home_layout = html.Div(
        [
            hero_section(),
            html.Hr(),
            problem_objective_section(),
            abstract_section(),
            methodology_section(),
            definitions_section(),
            footnote(),
        ]
    )

    return home_layout
