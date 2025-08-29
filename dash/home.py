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
                                'optimal price finder for sustainable growth', 
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
                            html.Span(
                                "Simply place your data in the designated folder, and the application will take care of the rest: from data engineering, to modeling, and finally generating the final presentation layer. ",
                                style={
                                    "color": "#5f6b7a",
                                    "margin-left": "100px",
                                    "margin-right": "200px",
                                },
                            ),                            
                            html.Span(
                                html.I(
                                    "As the user, your role is streamlined: review the interactive graphs and outputs, and focus your attention where it matters most: on making strategic decisions. "
                                ),
                                style={
                                    "color": "#5f6b7a",
                                    "margin-left": "100px",
                                    "margin-right": "100px",
                                },
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Alert(
                                            [
                                                html.Strong("💡 In short: "),
                                                "this is your end-to-end, automated pricing insights engine - transforming raw data into clear, decision-ready outputs.",
                                            ],
                                            color="light",
                                        ),
                                        md=12,
                                    )
                                ],
                                className="mt-3",
                                style={
                                    "margin-left": "80px",
                                    "padding": "14px 0",
                                    "width":"70%"
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
    CARD_STYLE = {
        "border": "1px solid #e9eef5",
        "borderRadius": "14px",
        "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
    }
    P_STYLE = {
        "color": "#5f6b7a",
        "marginLeft": "10px",
        "marginRight": "10px",
        "marginTop": "10px",
    }

    def _step(title, desc):
        return dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(title),
                        html.P(desc, style=P_STYLE),
                    ],
                    className="d-flex flex-column",
                ),
                style=CARD_STYLE,
                className="h-100",
            ),
        )

    return dbc.Container(
        [
            html.Div(id="method"),
            html.H3("(3) Methodology", style=ACCENT),

            # Responsive grid: 1 col on xs, 2 on md, 3 on xl (so 5 steps => 3 + 2 layout)
            dbc.Row(
                [
                    _step(
                        "Step 1: 📈 You provide sales data",
                        "We use past prices, units, revenue, and context like seasonality and promos.",
                    ),
                    _step(
                        "Step 2: 🔧 Data engineering",
                        "The app's built-in logic transforms raw inputs into clean, model-ready features.",
                    ),
                    _step(
                        "Step 3: 🤖 ML model trains",
                        "A flexible curve (GAM) captures how demand changes with price.",
                    ),
                    _step(
                        "Step 4: ⭐️ Insights revealed",
                        "We highlight revenue peaks from conservative to optimistic scenarios, along with their overlaps.",
                    ),
                    _step(
                        "Step 5: 🎯 You make the final call",
                        "With clear scenarios in hand, you decide the path that maximizes value for your business.",
                    ),
                ],
                className="row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 align-items-stretch mt-2",
                style={"maxWidth": "1550px", "margin": "0 auto"},
            ),

            # Why this works — centered, consistent width
            dbc.Row(
                dbc.Col(
                    dbc.Alert(
                        [
                            html.Strong("💡 Why this works: "),
                            "It balances higher margins at higher prices with potentially lower unit sales, and vice-versa.",
                        ],
                        color="light",
                        className="mb-0",
                    ),
                    md=12,
                ),
                className="mt-3",

                style={
                    "margin-left": "80px",
                    "padding": "14px 0",
                    "width":"70%"
                },

            ),
        ],
        fluid=True,
        style=section_style,
    )


def definitions_section():
    CARD_STYLE = {
        "border": "1px solid #e9eef5",
        "borderRadius": "14px",
        "boxShadow": "0 2px 8px rgba(16,24,40,0.06)",
    }
    P_STYLE = {
        "color": "#5f6b7a",
        "marginLeft": "10px",
        "marginRight": "10px",
        "marginTop": "10px",
    }

    def _card(title, text):
        return dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [html.H5(title), html.P(text, style=P_STYLE)],
                    className="d-flex flex-column",
                ),
                style=CARD_STYLE,
                className="h-100",
            ),
        )

    return dbc.Container(
        [
            html.H3("(4) Key Definitions", style=ACCENT),

            # Same layout rhythm as methodology (row width + gutters + equal heights)
            dbc.Row(
                [
                    _card(
                        "Recommended Price",
                        "The price point on the central curve where the model expects revenue to be highest on average.",
                    ),
                    _card(
                        "Alternative Scenarios",
                        "Conservative and optimistic cases that show how revenue could shift if demand reacts differently to price changes.",
                    ),
                    _card(
                        "Robustness",
                        "The recommendation is strongest when the central, conservative, and optimistic curves all peak around the same price point.",
                    ),
                    _card(
                        "Elasticity",
                        "How sensitive customer demand is to price changes — whether shoppers react strongly or only slightly when prices move.",
                    ),
                ],
                className="row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 align-items-stretch mt-2",
                style={"maxWidth": "1550px", "margin": "0 auto"},
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
