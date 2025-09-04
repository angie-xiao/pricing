# faq.py
from dash import html, dash_table
import dash_bootstrap_components as dbc

ACCENT = {"color": "#DAA520"}
LEFT_INDENT_PX = "50px"  # same left margin for title + table


def make_faq_table(table_id, rows):
    return dash_table.DataTable(
        id=table_id,
        columns=[
            {"name": "Summary", "id": "Summary"},
            {"name": "Business-friendly", "id": "Business-friendly"},
            {"name": "Technical nuances", "id": "Technical nuances"},
        ],
        data=rows,
        style_table={
            "width": "100%",
            "overflowX": "auto",
            "border": "none",
            "tableLayout": "fixed",  # enforce column widths
        },
        style_cell={
            "padding": "12px",
            "fontSize": "14px",
            "border": "none",
            "whiteSpace": "pre-wrap",
            "textAlign": "left",
            "lineHeight": "1.35",
            "fontFamily": "Source Sans Pro, -apple-system, BlinkMacSystemFont, "
                          "Segoe UI, Roboto, Helvetica Neue, Arial, Noto Sans, sans-serif",
        },
        style_header={
            "fontWeight": 600,
            "border": "none",
            "backgroundColor": "#F5EFE7",
            "textAlign": "center",
            "fontSize": "13px",
            "textTransform": "uppercase",
            "whiteSpace": "normal",  # align header wrapping with cells
        },
        style_cell_conditional=[
            {"if": {"column_id": "Summary"}, "width": "30%"},
            {"if": {"column_id": "Business-friendly"}, "width": "30%"},
            {"if": {"column_id": "Technical nuances"}, "width": "40%"},
        ],
    )


def faq_robustness_section():
    rows = [
        {
            "Summary": (
                "Robustness tells you how statistically trustworthy the "
                "recommended price is, based on model agreement, price sensitivity, and evidence depth."
            ),
            "Business-friendly": (
                "If all forecast scenarios point to about the same price - and we've seen that "
                "price region enough times - the recommendation is more trustworthy. "
                "If scenarios disagree, or if we have high price sensitivity, or insufficient data, trust is lower."
            ),
            "Technical nuances": (
                "• Spread alignment:\n\t"
                "‣ Identify peak prices at P2.5, P50, & P97.5 predicted revenues.\n\t"
                "‣ Compute spread = max(peak) - min(peak).\n\t"
                "‣ Normalize by ~10% of the median price; \n\t\t"
                "‣ Tighter spreads → higher confidence.\n\n"

                "• Data credibility:\n\t"
                "‣ Count distinct prices tested, not just total rows.\n\t"
                "‣ Apply a saturating curve: \n\t\t"
                "• First few distinct prices increase confidence quickly\n\t\t"
                "• Then additional prices give diminishing returns\n\t\t"
                "• Preventing duplicates from inflating the score.\n\n"

                "• Elasticity (price sensitivity):\n\t"
                "‣ Lower elasticity → higher confidence.\n\t"
                "‣ Normalized across the observed range\n\t\t"
                "• So extreme values are preserved\n\t\t"
                "• Ensuring truly sensitive pricing lowers the score.\n\n"

                "• Final score blending:\n\t"
                "‣ Combine 0.6 · elasticity + 0.4 · spread to get a base score.\n\t"
                "‣ Then adjust based on data credibility:\n\t\t"
                "  • More distinct prices increase confidence quickly at first,\n\t\t"
                "  • Additional prices give diminishing returns,\n\t\t"
                "  • Repeated measurements don't inflate the score.\n\t"
                "‣ Labels: \n\t\t"
                "• Strong (≥ 0.70)\n\t\t"
                "• Medium (≥ 0.45)\n\t\t"
                "• Weak (< 0.45)."
            ),
        }
    ]

    return html.Div(
        [
            html.H3(
                "How is the robustness calculated?",
                className="mt-3",
                style={
                    "color": ACCENT["color"],
                    "marginBottom": "24px",
                },
            ),
            make_faq_table("faq-robustness", rows),
        ],
        style={
            "marginLeft": LEFT_INDENT_PX,
            "marginRight": LEFT_INDENT_PX,
        },
    )


def faq_mean_vs_actual_section():
    rows = [
        {
            "Summary": (
                "Seeing actual revenue higher than the model's expected revenue "
                "(mean estimate) is not a failure."
            ),
            "Business-friendly": (
                "The model gives you the average outcome we'd expect at a price. "
                "Some real-world results can be higher or lower - that's normal. "
                "What matters is whether most results fall inside the model's "
                "prediction band and whether the chosen price makes sense overall."
            ),
            "Technical nuances": (
                "• Mean vs realization: \n\t"
                "‣ The P50 curve is the conditional mean. \n\t"
                "‣ Individual data points can be above or below it without issue.\n\n"
                "• Smoothing: \n\t"
                "‣ GAM models smooth noisy data, which can flatten sharp peaks.\n\t"
                "‣ i.e., some observed values will naturally exceed the mean curve.\n\n"
                "• Evaluation: \n\t"
                "‣ What matters is intervals—not a single point comparison\n\t"
                "‣ But the distribution of actuals vs. The prediction."
            ),
        }
    ]

    return html.Div(
        [
            html.H3(
                "What if actual revenue is higher than the model's estimate?",
                className="mt-3",
                style={
                    "color": ACCENT["color"],
                    "marginBottom": "24px",
                },
            ),
            make_faq_table("faq-mean-vs-actual", rows),
        ],
        style={
            "marginLeft": LEFT_INDENT_PX,
            "marginRight": LEFT_INDENT_PX,
        },
    )


def faq_tuning_section():
    rows = [
        {
            "Summary": (
                "We auto-tune an ExpectileGAM via grid search after scaling data, "
                "choosing the best combination of smoothness (lam), spline density, and spline order."
            ),
            "Business-friendly": (
                "We standardize features, try a few reasonable model settings, and keep the one that fits best. "
                "If the search fails (e.g., edge cases), we fall back to a safe default so you still get a result."
            ),
            "Technical nuances": (
                "• Data scaling:\n\t"
                "‣ Standardize X with StandardScaler.\n\t"
                "‣ Standardize y to zero-mean/unit-var (guard y_std=0).\n\n"
                "• Dynamic terms:\n\t"
                "‣ Build one spline term per feature: sum(s(i) for i in range(n_features)).\n\n"
                "• Model family:\n\t"
                "‣ ExpectileGAM(expectile = user-set, default 0.5) for mean-like fit with asymmetric option.\n\n"
                "• Hyperparameter grid:\n\t"
                "‣ lam ∈ {0.01, 0.1, 1, 10, 100} (logspace 1e-2→1e2)\n\t"
                "‣ n_splines ∈ {5, 10, 20}\n\t"
                "‣ spline_order ∈ {2, 3}\n\n"
                "• Selection objective:\n\t"
                "‣ Use .gridsearch(X_scaled, y_scaled, param_grid) and keep the best model (min GCV).\n\t"
                "‣ Note: gridsearch is exhaustive (no pruning), so the grid is intentionally small.\n\n"
                "• Robustness fallback:\n\t"
                "‣ On exception, fit ExpectileGAM with s(0, n_splines=5) as a minimal, stable default.\n\n"
                "• Inference convenience:\n\t"
                "‣ Persist scalers on the model: _scaler_X, _y_mean, _y_std for later inverse-transform."
            ),
        }
    ]

    return html.Div(
        [
            html.H3(
                "How was the model tuned?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-tuning", rows),
        ],
        style={"marginLeft": LEFT_INDENT_PX, "marginRight": LEFT_INDENT_PX},
    )


def faq_section():
    return html.Div(
        [
            dbc.Container(
                [
                    html.H1(
                        "FAQ",
                        className="display-5",
                        style={"textAlign": "center", "padding": "58px 0 58px"},
                    ),
                    faq_mean_vs_actual_section(),
                    html.Div(style={"height": "32px"}),
                    faq_robustness_section(),
                    html.Div(style={"height": "32px"}),
                    faq_tuning_section(),  # ← add this
                ],
                fluid=True,
                className="px-3",
                style={"maxWidth": "1500px", "margin": "0 auto"},
            ),
            html.Div(
                [
                    html.Span("made with ♥️ | "),
                    html.Span(html.I("@aqxiao")),
                    html.P("github.com/angie-xiao"),
                ],
                className="text-center py-3",
                style={
                    "fontSize": "0.8em",
                    "color": "#ac274f",
                    "textAlign": "center",
                    "backgroundColor": "#f3f3f3",
                    "marginTop": "40px",
                    "borderRadius": "0",
                    "width": "100%",
                },
            ),
        ]
    )
