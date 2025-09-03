"""
Q: how should i think of rev p50 < rev actual? so our optimization failed?
A: Short answer: not a failure by itself.

    rev_pred_0.5 is the model's expected (mean) revenue at a price. A single historical revenue_actual point can be higher than the mean and that’s perfectly normal. What matters is whether actuals mostly sit inside your prediction band and whether the chosen price makes sense given coverage and uncertainty.

    Here's how to think about it:

    When rev_actual > rev_pred_0.5 is OK

        - Mean vs realization: P50 here (with ExpectileGAM(expectile=0.5)) approximates the conditional mean, not a ceiling. Realizations can be above (and below) it.

        - Smoothing: GAM smooths noisy data. True peaks can be damped, so some points will exceed the mean curve.
"""


# faq.py
from dash import html, dash_table
import dash_bootstrap_components as dbc

ACCENT = {"color": "#DAA520"}  # if you reuse the same accent

def faq_robustness_section():
    rows = [
        {
            "Overview": (
                "Robustness (Confidence) tells you how statistically trustworthy the "
                "recommended price is, based on model agreement and evidence depth."
            ),
            "Business-friendly": (
                "If all forecast scenarios point to about the same price—and we’ve seen that "
                "price region enough times—the recommendation is more trustworthy. "
                "If scenarios disagree or we lack data around the peak, trust is lower."
            ),
            "Technical nuances": (
                "• **Spread alignment**: find peak ASPs at P2.5/P50/P97.5; compute "
                "`align_spread = max(peak) − min(peak)`. Normalize by a relative tolerance "
                "(~10% of P50), then score via a monotone penalty so tighter spreads → higher score.\n"
                "• **Evidence depth**: use the **number of distinct ASPs** observed (not just rows). "
                "Map `n_unique(asp)` to [0,1) with a saturating curve (diminishing returns) to avoid "
                "over-rewarding massive samples at one price.\n"
                "• **Blend**: final trust score = `0.6 · spread_score + 0.4 · data_score`. "
                "Labels: Strong (≥ 0.70), Medium (≥ 0.45), Weak (< 0.45).\n"
                "• **Elasticity is separate**: we show price sensitivity in its own badge so it guides "
                "business risk appetite without conflating it with statistical trust."
            ),
        }
    ]

    table = dash_table.DataTable(
        id="faq-robustness",
        columns=[
            {"name": "Overview", "id": "Overview", "presentation": "markdown"},
            {"name": "Non-technical / Business-friendly", "id": "Business-friendly", "presentation": "markdown"},
            {"name": "Technical nuances", "id": "Technical nuances", "presentation": "markdown"},
        ],
        data=rows,
        style_table={"overflowX": "auto", "border": "none"},
        style_cell={
            "whiteSpace": "pre-wrap",
            "textAlign": "left",
            "fontSize": "14px",
            "lineHeight": "1.35",
            "padding": "10px",
            "border": "none",
        },
        style_header={
            "fontWeight": 700,
            "fontSize": "13px",
            "textTransform": "uppercase",
            "backgroundColor": "#f6f6f6",
            "border": "none",
        },
        markdown_options={"html": True},
    )

    return dbc.Container(
        [

            html.H1("FAQ", className="DISPLAY-5", style={"textAlign": "center", "padding": "58px 0 8px"}),
            html.H3("Robustness (Confidence)", className="mt-3", style={"margin-left": "50px", "marginTop": "190px", "color": "#DAA520",}),                
            table,
        ],
        fluid=True,
        className="py-3",
    )


