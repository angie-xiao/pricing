# home.py — Explainer Homepage for Dash app
from dash import html
import dash_bootstrap_components as dbc

# Helpers (styles + small builders + footer)
from helpers import *

HOME_ACCENT = Style.HOME_ACCENT
HOME_MUTED = Style.HOME_MUTED
HOME_SECTION_STYLE = Style.HOME_SECTION_STYLE
HOME_CARD_STYLE = Style.HOME_CARD_STYLE 

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
                                "optimal price finder for sustainable growth",
                                style={"textAlign": "center", **HOME_MUTED},
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
                                    "marginLeft": "50px",
                                    "marginRight": "30px",
                                },
                            ),
                            html.P(
                                "The objective of this product is to provide a clear, evidence-based recommended price "
                                "that maximizes expected revenue while keeping assumptions transparent.",
                                style={"color": "#5f6b7a", "marginLeft": "100px"},
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
                                    "marginLeft": "100px",
                                    "marginRight": "150px",
                                    "width": "50%",
                                    "marginBottom": "50px",
                                },
                            ),
                        ]
                    )
                ]
            )
        ],
        fluid=True,
        style=Style.HOME_SECTION_STYLE,
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
                                    "marginLeft": "50px",
                                    "marginRight": "30px",
                                },
                            ),
                            html.Span(
                                "Simply place your data in the designated folder, and the application will take care of the rest: from data engineering, to modeling, and finally generating the final presentation layer. ",
                                style={
                                    "color": "#5f6b7a",
                                    "marginLeft": "100px",
                                    "marginRight": "200px",
                                },
                            ),
                            html.Span(
                                html.I(
                                    "As the user, your role is streamlined: review the interactive graphs and outputs, and focus your attention where it matters most: on making strategic decisions. "
                                ),
                                style={
                                    "color": "#5f6b7a",
                                    "marginLeft": "100px",
                                    "marginRight": "100px",
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
                                    "marginLeft": "80px",
                                    "padding": "14px 0",
                                    "width": "70%",
                                },
                            ),
                        ]
                    )
                ]
            )
        ],
        fluid=True,
        style=HOME_SECTION_STYLE,
    )


def methodology_section():
    return dbc.Container(
        [
            html.Div(id="method"),
            html.H3("(3) Methodology", style=HOME_ACCENT),
            # Responsive grid
            dbc.Row(
                [
                    home_step_card(
                        "Step 1: You provide sales data",
                        "We use past prices, units, revenue, and context like seasonality and promos.",
                    ),
                    home_step_card(
                        "Step 2: Data engineering",
                        "The app's built-in logic transforms raw inputs into clean, model-ready features.",
                    ),
                    home_step_card(
                        "Step 3: ML model trains",
                        "A flexible curve (GAM) captures how demand changes with price.",
                    ),
                    home_step_card(
                        "Step 4: Insights revealed",
                        "We highlight revenue peaks from conservative to optimistic scenarios, along with their overlaps.",
                    ),
                    home_step_card(
                        "Step 5: You make the final call",
                        "With clear scenarios in hand, you decide the path that maximizes value for your business.",
                    ),
                ],
                className="row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 align-items-stretch mt-2",
                style={"maxWidth": "1550px", "margin": "0 auto"},
            ),
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
                style={"marginLeft": "80px", "padding": "14px 0", "width": "70%"},
            ),
        ],
        fluid=True,
        style=HOME_SECTION_STYLE,
    )


def definitions_section():
    return dbc.Container(
        [
            html.H3("(4) Key Definitions", style=HOME_ACCENT),
            dbc.Row(
                [
                    info_card(
                        "Recommended Price",
                        "The price point on the central curve where the model expects revenue to be highest on average.",
                    ),
                    info_card(
                        "Alternative Scenarios",
                        "Conservative and optimistic cases that show how revenue could shift if demand reacts differently to price changes.",
                    ),
                    info_card(
                        "Robustness",
                        "The recommendation is strongest when the central, conservative, and optimistic curves all peak around the same price point.",
                    ),
                    info_card(
                        "Elasticity",
                        "How sensitive customer demand is to price changes — whether shoppers react strongly or only slightly when prices move.",
                    ),
                ],
                className="row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 align-items-stretch mt-2",
                style={"maxWidth": "1550px", "margin": "0 auto"},
            ),
        ],
        fluid=True,
        style=HOME_SECTION_STYLE,
    )


def footnote():
    return build_footer_two_lines(
        signature_handle="@aqxiao", link_text="github.com/angie-xiao"
    )


# --- Compose layout ---
def Homepage():
    return html.Div(
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
