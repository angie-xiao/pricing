# faq.py
from dash import html, dash_table
import dash_bootstrap_components as dbc

ACCENT = {"color": "#DAA520"}
LEFT_INDENT_PX = "50px"  # same left margin for title + table
INDENT = "\u00a0\u00a0\u00a0\u00a0"  # 4 non-breaking spaces


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


# 1) End-to-end pipeline (matches DataEngineer + PricingPipeline)
def faq_pipeline_section():
    rows = [
        {
            "Summary": (
                "We clean and join pricing + product data, aggregate to daily product-ASP level, "
                "keep Top-N products by revenue, then build price curves and recommended prices. "
                "We also compute elasticity and annualized opportunity deltas."
            ),
            "Business-friendly": (
                "Think of it as a funnel:\n"
                f"{INDENT}1) Clean & match your sales with product tags/weights\n"
                f"{INDENT}2) Roll up by day and price (ASP)\n"
                f"{INDENT}3) Focus on Top-N highest-revenue products so the dashboard stays sharp\n"
                f"{INDENT}4) Learn demand/revenue curves and pick the best price\n"
                f"{INDENT}5) Show upside vs current price, including annualized impact"
            ),
            "Technical nuances": (
                "• Engineering:\n"
                f"{INDENT}‣ Daily aggregation per (asin, product, order_date) with shipped_units & revenue_share_amt\n"
                f"{INDENT}‣ ASP = revenue_share_amt / shipped_units (rounded to 0.1); keep ASP>0 & units>0\n"
                f"{INDENT}‣ Collapse to (asin, product, asp): totals + days_sold = #daily rows at that ASP,\n"
                f"{INDENT}  daily_units, daily_rev = totals / #rows\n\n"
                "• Top-N filter:\n"
                f"{INDENT}‣ Keep only Top-N products by total revenue_share_amt to avoid over-display\n\n"
                "• Frames exposed to UI:\n"
                f"{INDENT}‣ all_gam_results, best_avg/best50/best975/best25, elasticity_df, curr_price_df,\n"
                f"{INDENT}‣ opps_summary (delta_units/_revenue incl. annualized), meta (data_start/end, days_covered, annual_factor)"
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "What does the pipeline do end-to-end?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-pipeline", rows),
        ],
        style={"marginLeft": LEFT_INDENT_PX, "marginRight": LEFT_INDENT_PX},
    )


# 2) Model choice (matches GAMModeler/GAMTuner)
def faq_gam_section():
    rows = [
        {
            "Summary": (
                "We use an ExpectileGAM (a smooth, nonlinear model) at three expectiles "
                "(2.5%, 50%, 97.5%) to learn revenue vs price. "
                "Features are [ASP, days_sold]; the target is daily revenue."
            ),
            "Business-friendly": (
                "Instead of forcing a straight line, we let the data trace a flexible curve. "
                "And we don’t rely on a single curve — we fit cautious, typical, and optimistic versions "
                "to visualize downside/upside around the expected outcome."
            ),
            "Technical nuances": (
                "• Targets & features:\n"
                f"{INDENT}‣ y = daily_rev\n"
                f"{INDENT}‣ X = [asp, days_sold] (both standardized)\n\n"
                "• Terms:\n"
                f"{INDENT}‣ ExpectileGAM with s(0, spline_order=3) + s(1, spline_order=2)\n\n"
                "• Expectiles:\n"
                f"{INDENT}‣ q ∈ { {0.025, 0.5, 0.975} } ⇒ conservative / median / optimistic revenue curves\n\n"
                "• Sample weights (per point):\n"
                f"{INDENT}‣ sqrt(daily_rev) × exp(-γ_time · days_sold/median_days) × tail_boost\n"
                f"{INDENT}‣ tail_boost increases weight for price points farther from the median ASP (capped)\n\n"
                "• Outputs stored:\n"
                f"{INDENT}‣ revenue_pred_{'{q}'} for q∈{{0.025,0.5,0.975}}, plus revenue_pred_avg (mean across q)\n"
                f"{INDENT}‣ revenue_actual = daily_rev (for comparison)"
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


# 3) Tuning (matches GAMTuner grid)
def faq_tuning_section():
    rows = [
        {
            "Summary": (
                "We grid-search smoothness (λ) and spline count (n_splines) with standardized features and "
                "sample weights. If grid-search errors, we fall back to a safe default fit."
            ),
            "Business-friendly": (
                "We try a small, sensible set of settings and keep what fits best. "
                "If anything goes wrong, we still return stable predictions."
            ),
            "Technical nuances": (
                "• Standardization: X is scaled via StandardScaler.\n\n"
                "• Hyperparameter grid:\n"
                f"{INDENT}‣ λ ∈ logspace(10^-4, 10^3) with 8 points\n"
                f"{INDENT}‣ n_splines ∈ {{5, 10, 20, 30}}\n\n"
                "• Terms fixed: s(asp, order=3) + s(days_sold, order=2)\n\n"
                "• Selection: ExpectileGAM.gridsearch(Xs, y, lam, n_splines, weights)\n\n"
                "• Fallback: on exception, fit ExpectileGAM with the same terms (no search)"
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


# 4) Optimal price (matches Optimizer + PricingPipeline.best50/best_avg)
def faq_optimal_asp_section():
    rows = [
        {
            "Summary": (
                "We compute revenue across the learned curves and recommend the ASP that maximizes "
                "the average predicted revenue (mean of 2.5/50/97.5 expectiles)."
            ),
            "Business-friendly": (
                "For each product, we trace the full revenue curve (price × predicted demand). "
                "The recommended price is where expected revenue peaks. "
                "We also surface conservative/median/optimistic peak points for sensitivity."
            ),
            "Technical nuances": (
                "• Curves: revenue_pred_q from ExpectileGAM at q ∈ {0.025, 0.5, 0.975}\n"
                "• Aggregation: revenue_pred_avg = mean(revenue_pred_0.025, 0.5, 0.975)\n"
                "• Optimizer: best_avg = argmax(revenue_pred_avg) per product\n"
                "• Guardrails: best25 (q=0.025), best50 (q=0.5), best975 (q=0.975)\n"
                "• Current vs best: we find the nearest modeled row at the current price to compare expected units/revenue"
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


# 5) Elasticity (matches ElasticityAnalyzer)
def faq_elasticity_section():
    rows = [
        {
            "Summary": (
                "Elasticity here is a quick, UI-only indicator of price sensitivity using "
                "log-percentage changes between the highest/lowest ASP and units."
            ),
            "Business-friendly": (
                "It answers: 'When price moves across its observed range, how much do units move?' "
                "Higher ratios imply stronger demand response to price."
            ),
            "Technical nuances": (
                "• For each product:\n"
                f"{INDENT}‣ pct_change_price = 100·[ln(asp_max) − ln(asp_min)]\n"
                f"{INDENT}‣ pct_change_qty   = 100·[ln(units_max) − ln(units_min)]\n"
                f"{INDENT}‣ ratio = pct_change_qty / pct_change_price (nan-safe)\n"
                "• We also provide a percentile rank (pct) of ratio across products for relative comparison.\n"
                "• Note: this does not come from the GAM; it’s a simple spread-based proxy for the UI."
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "What does 'Elasticity' mean here?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-elasticity", rows),
        ],
        style={"marginLeft": LEFT_INDENT_PX, "marginRight": LEFT_INDENT_PX},
    )


# 6) Annualized opportunity math (matches opps_summary + meta)
def faq_annualization_section():
    rows = [
        {
            "Summary": (
                "Upside units/revenue are computed vs current price, then scaled to a yearly view "
                "based on the actual number of days in your dataset."
            ),
            "Business-friendly": (
                "We compare the model’s best price to your current price to estimate added units and revenue. "
                "Then we annualize by accounting for the time window of your data so short samples don’t overstate impact."
            ),
            "Technical nuances": (
                "• From best50 (q=0.5) we take best_price, units_pred_best, revenue_pred_best\n"
                "• At current_price we find the nearest modeled row ⇒ units_pred_curr, revenue_pred_curr\n"
                "• Deltas:\n"
                f"{INDENT}‣ delta_units = units_pred_best − units_pred_curr\n"
                f"{INDENT}‣ delta_revenue = revenue_pred_best − revenue_pred_curr\n"
                "• Annualization factor:\n"
                f"{INDENT}‣ days_covered = (data_end − data_start) + 1\n"
                f"{INDENT}‣ annual_factor = 365 / max(1, days_covered)\n"
                "• Annualized metrics:\n"
                f"{INDENT}‣ delta_units_annual = delta_units × annual_factor\n"
                f"{INDENT}‣ delta_revenue_annual = delta_revenue × annual_factor\n"
                f"{INDENT}‣ revenue_best_annual = revenue_pred_best × annual_factor"
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "How do you compute the annualized opportunities?",
                className="mt-3",
                style={"color": ACCENT["color"], "marginBottom": "24px"},
            ),
            make_faq_table("faq-annualization", rows),
        ],
        style={"marginLeft": LEFT_INDENT_PX, "marginRight": LEFT_INDENT_PX},
    )


# 7) Mean vs actual (kept, but clarified for daily_rev target)
def faq_mean_vs_actual_section():
    rows = [
        {
            "Summary": (
                "Seeing actual revenue higher (or lower) than the model’s expected revenue at a price is normal. "
                "We care about the band of plausible outcomes, not any single point."
            ),
            "Business-friendly": (
                "The 50% curve is the typical expectation. Real days will bounce above or below it. "
                "As long as most points sit within the conservative–optimistic band, the model is behaving."
            ),
            "Technical nuances": (
                "• Expectile q=0.5 acts like a mean under symmetric squared loss; we also fit q=0.025 and q=0.975\n"
                "• GAMs smooth noisy data, so sharp spikes can exceed the mean curve\n"
                "• Evaluate coverage of actuals within the (2.5%–97.5%) band rather than single deviations"
            ),
        }
    ]
    return html.Div(
        [
            html.H3(
                "What if actual revenue beats the model’s estimate?",
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
                    faq_pipeline_section(),
                    html.Div(style={"height": "32px"}),
                    faq_gam_section(),
                    html.Div(style={"height": "32px"}),
                    faq_tuning_section(),
                    html.Div(style={"height": "32px"}),
                    faq_optimal_asp_section(),
                    html.Div(style={"height": "32px"}),
                    faq_elasticity_section(),
                    html.Div(style={"height": "32px"}),
                    faq_annualization_section(),
                    html.Div(style={"height": "32px"}),
                    faq_mean_vs_actual_section(),
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
