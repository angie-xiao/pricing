# overview.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# pull shared helpers
from helpers import (
    OVERVIEW_ACCENT,
    kpi_card,
    model_fit_units,
    update_elasticity_kpi_by_product,
    scenario_table,
    annualized_kpis_signed,
    coverage_note,
)


def layout(products_lookup: pd.DataFrame):
    """Pass a Top-N-only DataFrame with columns ['asin','product']."""
    return dbc.Container(
        [
            html.H1(
                "Overview",
                className="display-5",
                style={"textAlign": "center", "padding": "58px 0 8px"},
            ),
            # Product selector (Top-N only)
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            "Select a Product:",
                            style={
                                "fontWeight": 600,
                                "textAlign": "right",
                                "marginRight": "10px",
                            },
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="product_dropdown_snap",
                            options=[
                                {"label": r["product"], "value": r["asin"]}
                                for _, r in products_lookup.iterrows()
                            ],
                            value=(
                                products_lookup["asin"].iloc[0]
                                if len(products_lookup)
                                else None
                            ),
                            style={"width": "300px"},
                            clearable=False,
                        ),
                        width="auto",
                    ),
                ],
                justify="center",
                align="center",
                style={"padding": "10px 0 20px"},
            ),
            # KPI row (exact order requested)
            dbc.Row(
                [
                    kpi_card(
                        "card_title_date_range",
                        "Number of Days",
                        "date_range_value",
                        bg="#eef2fa",
                        id_subtext="date-range-subtext",
                    ),
                    kpi_card(
                        "card_title_curr_price_snap",
                        "Current Price",
                        "curr_price_snap",
                        bg="#eef2fa",
                    ),
                    kpi_card(
                        "card_title_snap",
                        "Recommended Price",
                        "card_asp_snap",
                        bg="#F5E8D8",
                    ),
                    kpi_card(
                        "card_title_elasticity_snap",
                        "Elasticity",
                        "elasticity_ratio_snap",
                        bg="#eef2fa",
                        id_subtext="elasticity-subtext",
                    ),
                    kpi_card(
                        "card_title_units_opp_ann",
                        "Annualized Units Sold Opportunity",
                        "units_opp_ann_value",
                        bg="#eef8f0",
                    ),
                    kpi_card(
                        "card_title_rev_best_ann",
                        "Annualized Revenue Opportunity",
                        "rev_best_ann_value",
                        bg="#eef8f0",
                    ),
                    kpi_card(
                        "card_title_fit_snap",
                        "Model Fit (Daily Revenue)",
                        "fit_value_snap",
                        bg="#eef2fa",
                        id_subtext="fit-subtext",
                    ),
                ],
                className="g-4 align-items-stretch",
                justify="center",
                align="center",
                style={"padding": "10px 0 10px"},
            ),
            html.Br(),
            html.Hr(className="my-4", style={"padding": "20px"}),
            # Predictive graph + scenario table + explainer
            dbc.Row(
                [
                    # title
                    html.H3(
                        "Predictive Graph",
                        className="mt-3",
                        style={
                            "marginLeft": "50px",
                            "marginTop": "190px",
                            "color": "#DAA520",
                        },
                    ),
                    # graph
                    dbc.Col(
                        [
                            dcc.Loading(
                                type="circle",
                                children=dcc.Graph(
                                    id="gam_results_pred",
                                    style={"height": "560px"},
                                    config={"displaylogo": False},
                                ),
                            )
                        ],
                        md=8,
                        xs=12,
                        className="mb-0",
                    ),
                    # scenario table and explainer
                    dbc.Col(
                        [
                            html.H6(
                                "Scenario Summary",
                                className="mb-2",
                                style={"textAlign": "center", "marginTop": "40px"},
                            ),
                            dash_table.DataTable(
                                id="scenario_table",
                                columns=[
                                    {"name": "Case", "id": "case"},
                                    {"name": "Price", "id": "price"},
                                    {"name": "Revenue", "id": "revenue"},
                                ],
                                data=[],
                                style_table={"border": "none", "marginBottom": "12px"},
                                style_cell={
                                    "textAlign": "center",
                                    "border": "none",
                                    "fontSize": "14px",
                                    "padding": "6px",
                                },
                                style_header={
                                    "fontWeight": 600,
                                    "border": "none",
                                    "backgroundColor": "#f6f6f6",
                                },
                            ),
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
                                                        "Pick the price where the central curve's expected revenue is highest.",
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.B("Range: "),
                                                        "Conservative and optimistic curves show how outcomes may vary.",
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.B("Confidence: "),
                                                        "Stronger when all curves peak around a similar price and there's lots of nearby data.",
                                                    ]
                                                ),
                                            ],
                                            className="predictive-explainer-list",
                                        ),
                                    ]
                                ),
                                className="shadow-sm",
                                style={"marginTop": "70px"},
                            ),
                        ],
                        md=4,
                        xs=12,
                    ),
                ],
                className="g-3 mb-2",
            ),
            # Coverage note
            html.Div(
                html.Span(html.I(id="coverage_note")),
                style={
                    "textAlign": "center",
                    "color": "#5f6b7a",
                    "fontSize": "0.9em",
                    "marginBottom": "50px",
                },
            ),
            # Footer
            html.Div(
                [
                    html.Div(
                        [html.Span("made with ♥️ | "), html.Span(html.I("@aqxiao"))],
                        style={"marginBottom": "4px"},
                    ),
                    html.A(
                        "github.com/angie-xiao",
                        href="https://github.com/angie-xiao",
                        style={"textDecoration": "none", "color": "#ac274f"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "height": "80px",
                    "fontSize": "0.8em",
                    "color": "#ac274f",
                    "backgroundColor": "#f3f3f3",
                    "borderRadius": "0",
                    "width": "100%",
                    "marginTop": "40px",
                },
            ),
        ],
        fluid=True,
    )


# ---------- callbacks ----------
def register_callbacks(
    app,
    price_quant_df,
    best50_optimal_pricing_df,
    curr_price_df,
    elasticity_df,
    all_gam_results,
    products_lookup,
    meta,
    viz_cls,
):
    viz = viz_cls()

    @app.callback(
        Output("card_title_date_range", "children"),
        Output("date_range_value", "children"),
        Output("date-range-subtext", "children"),
        Output("card_title_curr_price_snap", "children"),
        Output("curr_price_snap", "children"),
        Output("card_title_snap", "children"),
        Output("card_asp_snap", "children"),
        Output("card_title_elasticity_snap", "children"),
        Output("elasticity_ratio_snap", "children"),
        Output("elasticity-subtext", "children"),
        Output("card_title_units_opp_ann", "children"),
        Output("units_opp_ann_value", "children"),
        Output("card_title_rev_best_ann", "children"),
        Output("rev_best_ann_value", "children"),
        Output("card_title_fit_snap", "children"),
        Output("fit_value_snap", "children"),
        Output("fit-subtext", "children"),
        Output("gam_results_pred", "figure"),
        Output("scenario_table", "data"),
        Output("coverage_note", "children"),
        Input("product_dropdown_snap", "value"),
    )
    def overview(asin):
        if not asin:
            empty_fig = viz.empty_fig("Select a product")
            return (
                "",
                "—",
                "",
                "",
                "—",
                "",
                "—",
                "",
                "—",
                "",
                "",
                "—",
                "",
                "—",
                "",
                "—",
                "",
                empty_fig,
                [],
                "",
            )

        try:
            display_name = products_lookup.loc[
                products_lookup["asin"] == asin, "product"
            ].iloc[0]
        except Exception:
            display_name = ""

        cp = curr_price_df.loc[curr_price_df["asin"] == asin, "current_price"]
        curr_price_val = f"${float(cp.iloc[0]):,.2f}" if len(cp) else "—"

        best_row = best50_optimal_pricing_df.loc[
            best50_optimal_pricing_df["asin"] == asin
        ]
        asp_val = f"${float(best_row['asp'].iloc[0]):,.2f}" if len(best_row) else "—"

        elast_val, elast_subtext = update_elasticity_kpi_by_product(
            display_name, elasticity_df
        )

        filt = all_gam_results[all_gam_results["asin"] == asin]
        pred_graph = (
            viz.gam_results(filt) if len(filt) else viz.empty_fig("No model data")
        )

        scenario_df = (
            scenario_table(filt)
            if len(filt)
            else pd.DataFrame([{"case": "—", "price": "—", "revenue": "—"}])
        )
        scenario_data = scenario_df.to_dict("records")

        du_ann, rev_best_ann = annualized_kpis_signed(
            asin,
            best50_optimal_pricing_df,
            curr_price_df,
            all_gam_results,
            meta.get("annual_factor", 1.0),
        )

        fit_val, fit_sub = model_fit_units(filt)
        cov = coverage_note(filt)

        start_date = pd.to_datetime(meta.get("data_start"))
        end_date = pd.to_datetime(meta.get("data_end"))
        if pd.notna(start_date) and pd.notna(end_date):
            num_days = (end_date - start_date).days + 1
            date_range_val = f"{num_days:,}"
            date_range_subtext = (
                f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
        else:
            date_range_val = "—"
            date_range_subtext = "—"

        return (
            "",
            date_range_val,
            date_range_subtext,
            "",
            curr_price_val,
            "",
            asp_val,
            "",
            elast_val,
            elast_subtext,
            "",
            du_ann,
            "",
            rev_best_ann,
            "",
            fit_val,
            fit_sub,
            pred_graph,
            scenario_data,
            cov,
        )
