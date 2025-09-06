# faq.py
from dash import html, dash_table
import dash_bootstrap_components as dbc

ACCENT = {"color": "#DAA520"}
LEFT_INDENT_PX = "50px"  # same left margin for title + table
INDENT = "\u00A0\u00A0\u00A0\u00A0"  # 4 non-breaking spaces


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
            "tableLayout": "fixed",
        },
        style_cell={
            "padding": "12px",
            "fontSize": "14px",
            "border": "none",
            "whiteSpace": "pre-wrap",
            "height": "auto",
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
            "whiteSpace": "normal",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Summary"}, "width": "30%"},
            {"if": {"column_id": "Business-friendly"}, "width": "30%"},
            {"if": {"column_id": "Technical nuances"}, "width": "40%"},
        ],
    )


def faq_gam_section():
    rows = [
        {
            "Summary": (
                "ExpectileGAM - a special case of GAM (Generalized Additive Model). \n\n"
                "It fits smooth, nonlinear demand curves at different expectiles (2.5%, 50%, 97.5%) "
                "so we can see conservative, expected, and optimistic scenarios."
            ),
            "Business-friendly": (
                "A GAM is a way to learn the true shape of the demand curve instead of forcing a straight line. \n\n"
                "ExpectileGAM goes further by drawing not just one curve, but three: "
                "a cautious version, a middle version, and an optimistic version. "
                "This shows us the range of outcomes at different prices, not just the average."
            ),
            "Technical nuances": (
                "• GAM basics:\n"
                "    ‣ y = β0 + f(price) + ε, with f(price) as a smooth spline\n\n"
                "• Why ExpectileGAM:\n"
                "    ‣ Minimizes weighted squared error (asymmetric MSE) \n\t"
                "    • Squared loss is smooth\n\t"
                "    • Better behaved gradients & more reliable convergence\n"
                "    ‣ Great for smaller sample sizes or noisy data points (our case)\n\n"
                "• Setup:\n"
                "    ‣ Predictor (X): ASP (average selling price)\n"
                "    ‣ Response (y): shipped units per product-week\n"
                "    ‣ Fit at q = 0.025, 0.5, 0.975 \n\t"
                "    → conservative, median, & optimistic curves\n\n"
                "• Outputs:\n"
                "    ‣ Predicted units at each ASP\n"
                "    ‣ Convert to revenue = ASP * predicted units\n\n"
                "• Evaluation:\n"
                "    ‣ Compare predicted vs. actual revenue\n"
                "    ‣ RMSE used to monitor fit quality"
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "What model did we use?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-gam", rows),
        ],
        style={"marginLeft": LEFT_INDENT_PX, "marginRight": LEFT_INDENT_PX},
    )


def faq_tuning_section():
    rows = [
        {
            "Summary": (
                "We tune the ExpectileGAM with a compact grid search, testing smoothness and spline settings, "
                "and pick the model with lowest error. If search fails, we fall back to a safe default."
            ),
            "Business-friendly": (
                "We try a few reasonable model settings and keep whichever explains the data best. "
                "If anything goes wrong, the system defaults to a stable backup so predictions are always available."
            ),
            "Technical nuances": (
                "• Preprocessing:\n"
                "    ‣ Standardize X (price) - so price stays scale-neutral\n"
                "    ‣ Normalize y (units) - so big sellers don't dominate\n\n"
                "• Model terms:\n"
                "    ‣ One spline term for price only: s(0).\n\n"
                "• Hyperparameter grid:\n"
                "    ‣ λ ∈ {0.01, 0.1, 1, 10, 100}\n\t"
                "    → log-spaced smoothness penalties; wiggly to almost linear.\n"
                "    ‣ n_splines ∈ {5, 10, 20}\n\t"
                "    → curve resolution, from coarse to flexible.\n"
                "    ‣ spline_order ∈ {2, 3}\n\t"
                "    → quadratic (gentle) vs cubic (sharper bends).\n\n"
                "• Selection:\n"
                "    ‣ Use .gridsearch() to minimize GCV.\n"
                "    ‣ Grid is small to stay fast and interpretable.\n\n"
                # "• Fallback:\n"
                # "    ‣ On error, fit ExpectileGAM(s(0, n_splines=5)) as a stable default.\n\n"
                # "• Post-fit:\n"
                # "    ‣ Persist scalers (_scaler_X, _y_mean, _y_std) for consistent inverse-transform."
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



def faq_optimal_asp_section():
    rows = [
        {
            "Summary": (
                "The optimal ASP is chosen by simulating revenue across the demand curves "
                "and selecting the price that maximizes expected revenue."
            ),
            "Business-friendly": (
                "For each product, we map the entire revenue curve (price × predicted demand). "
                "The recommended price is where revenue peaks. "
                "We also show conservative and optimistic alternatives so stakeholders can see the upside and downside."
            ),
            "Technical nuances": (
                "• Revenue curves:\n"
                "    ‣ Predict revenues at each ASP from ExpectileGAMs at q=0.025, 0.5, 0.975.\n"
                "• Aggregation:\n"
                "    ‣ Take mean of the three → revenue_pred_avg.\n\n"
                "• Optimizer:\n"
                "    ‣ Recommended ASP = argmax(revenue_pred_avg).\n"
                "    ‣ Guardrails: best_025 (conservative), best_50 (median), best_975 (optimistic).\n\n"
                "• Dashboard:\n"
                "    ‣ Show best_avg vs. current price to quantify upside.\n"
                "    ‣ Conservative/optimistic points included for sensitivity context."
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "How is the optimal ASP recommended?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-optimal-asp", rows),
        ],
        style={"marginLeft": LEFT_INDENT_PX, "marginRight": LEFT_INDENT_PX},
    )


def faq_robustness_section():
    rows = [
        {
            "Summary": (
                "Robustness tells you how statistically trustworthy the recommended price is, "
                "based on model agreement, price sensitivity, and evidence depth."
            ),
            "Business-friendly": (
                "If all forecast scenarios point to about the same price—and we've seen that price region enough times—"
                "the recommendation is more trustworthy. "
                "If scenarios disagree, or if we have high price sensitivity, or insufficient data, trust is lower."
            ),
            "Technical nuances": (
                "• Spread alignment:\n"
                "    ‣ Identify peak prices at P2.5, P50, P97.5 predicted revenues.\n"
                "    ‣ Spread = max(peak) - min(peak).\n"
                "    ‣ Normalize by ~10% of median price; tighter spreads → higher confidence.\n\n"
                "• Data credibility:\n"
                "    ‣ Count distinct ASPs tested, not just total rows.\n"
                "    ‣ Apply saturating curve: first few prices add confidence quickly, then diminishing returns.\n"
                "    ‣ Prevent duplicates from inflating score.\n\n"
                "• Elasticity (price sensitivity):\n"
                "    ‣ Lower elasticity → higher confidence.\n"
                "    ‣ Normalized across observed range so extreme values are preserved.\n\n"
                "• Final score blending:\n"
                "    ‣ Combine 0.6 * elasticity + 0.4 * spread.\n"
                "    ‣ Adjust by data credibility curve.\n"
                "    ‣ Labels: Strong (≥ 0.70), Medium (≥ 0.45), Weak (< 0.45)."
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "How is the robustness calculated?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-robustness", rows),
        ],
        style={"marginLeft": LEFT_INDENT_PX, "marginRight": LEFT_INDENT_PX},
    )


def faq_mean_vs_actual_section():
    rows = [
        {
            "Summary": (
                "Seeing actual revenue higher than the model's expected revenue (mean estimate) is not a failure."
            ),
            "Business-friendly": (
                "The model gives you the average outcome we'd expect at a price. "
                "Some real-world results can be higher or lower—that's normal. "
                "What matters is whether most results fall inside the model's prediction band "
                "and whether the chosen price makes sense overall."
            ),
            "Technical nuances": (
                "• Mean vs. realization:\n"
                "    ‣ The P50 curve is the conditional mean.\n"
                "    ‣ Individual data points may sit above or below it without issue.\n\n"
                "• Smoothing:\n"
                "    ‣ GAM models smooth noisy data, which can flatten sharp peaks.\n"
                "    ‣ Some observed values will exceed the mean curve naturally.\n\n"
                "• Evaluation:\n"
                "    ‣ Focus on prediction intervals, not single-point deviations.\n"
                "    ‣ Distribution of actuals vs. prediction band is what matters."
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "What if actual revenue is higher than the model's estimate?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-mean-vs-actual", rows),
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

                    faq_gam_section(),
                    html.Div(style={"height": "32px"}),
                    faq_tuning_section(),
                    html.Div(style={"height": "32px"}),
                    faq_optimal_asp_section(),
                    html.Div(style={"height": "32px"}),
                    faq_mean_vs_actual_section(),
                    html.Div(style={"height": "32px"}),
                    faq_robustness_section(),
                    html.Div(style={"height": "32px"}),
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
