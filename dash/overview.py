# overview.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
from helpers import (
    Style,
    OverviewUI as UI,
    Metrics,
    Scenario,
    Notes,
    OverviewHelpers as OH,
)

# constants previously imported directly
OVERVIEW_ACCENT = Style.OVERVIEW_ACCENT


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
        Output("gam_results", "figure"),
        Output("scenario_table", "data"),
        Output("coverage_note", "children"),
        Input("product_dropdown_snap", "value"),
        # prevent_initial_call=True  # Add this to prevent initial callback
    )
    def overview(asin):
        if not asin:
            return OH.empty_overview_payload(viz)

        try:
            # KPIs: date range
            date_range_val, date_range_subtext = OH.date_kpis(meta)

            # Names & prices
            pname = OH.display_name(asin, products_lookup)
            curr_price_val = OH.current_price_text(asin, curr_price_df)
            asp_val = OH.recommended_price_text(asin, best50_optimal_pricing_df)

            # Elasticity KPI
            elast_val, elast_sub = OH.elasticity_texts(pname, elasticity_df)

            # Filtered data & dual graph
            filt = OH.filter_product_rows(asin, all_gam_results)
            fig_dual = OH.pred_graph(viz, filt)

            # Scenario table (combined revenue + units)
            scenario_data = OH.scenario_records(
                filt,
                include_weighted=True,  # Add parameter to include weighted predictions
            )

            # Annualized KPI deltas (use pipeline's annual_factor)
            du_ann, rev_best_ann = OH.annualized_kpis(
                asin, best50_optimal_pricing_df, curr_price_df, all_gam_results
            )

            # Fit & coverage
            # fit_val, fit_sub, cov = OH.fit_and_coverage(filt)
            fit_val, fit_sub = Metrics.model_fit_units(filt)
            cov = Notes.coverage(filt)

            return (
                "",  # card_title_date_range (title is in layout)
                date_range_val,  # date_range_value
                date_range_subtext,  # date-range-subtext
                "",  # card_title_curr_price_snap
                curr_price_val,  # curr_price_snap
                "",  # card_title_snap
                asp_val,  # card_asp_snap
                "",  # card_title_elasticity_snap
                elast_val,  # elasticity_ratio_snap
                elast_sub,  # elasticity-subtext
                "",  # card_title_units_opp_ann
                du_ann,  # units_opp_ann_value
                "",  # card_title_rev_best_ann
                rev_best_ann,  # rev_best_ann_value
                "",  # card_title_fit_snap
                fit_val,  # fit_value_snap
                fit_sub,  # fit-subtext
                fig_dual,  # gam_results
                scenario_data,  # scenario_table (combined)
                cov,  # coverage_note
            )
        except Exception as e:
            print(f"Callback error: {str(e)}")
            return OH.empty_overview_payload(viz)


def layout(products_lookup):
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
            # KPI row
            dbc.Row(
                [
                    UI.kpi_card(
                        "card_title_date_range",
                        "Number of Days",
                        "date_range_value",
                        bg="#eef2fa",
                        id_subtext="date-range-subtext",
                    ),
                    UI.kpi_card(
                        "card_title_curr_price_snap",
                        "Current Price",
                        "curr_price_snap",
                        bg="#eef2fa",
                    ),
                    UI.kpi_card(
                        "card_title_snap",
                        "Recommended Price",
                        "card_asp_snap",
                        bg="#F5E8D8",
                    ),
                    UI.kpi_card(
                        "card_title_elasticity_snap",
                        "Elasticity",
                        "elasticity_ratio_snap",
                        bg="#eef2fa",
                        id_subtext="elasticity-subtext",
                    ),
                    UI.kpi_card(
                        "card_title_units_opp_ann",
                        "Annualized Units Sold Opportunity",
                        "units_opp_ann_value",
                        bg="#eef8f0",
                    ),
                    UI.kpi_card(
                        "card_title_rev_best_ann",
                        "Annualized Revenue Opportunity",
                        "rev_best_ann_value",
                        bg="#eef8f0",
                    ),
                    UI.kpi_card(
                        "card_title_fit_snap",
                        "Model Fit on Daily Revenue (RMSE)",
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
            # Graph + scenario section
            dbc.Row(
                [
                    html.H3(
                        "Price vs Expected Revenue",
                        className="mt-3",
                        style={
                            "marginLeft": "50px",
                            "marginTop": "190px",
                            "color": OVERVIEW_ACCENT,
                        },
                    ),
                    dbc.Col(
                        [
                            dcc.Loading(
                                type="circle",
                                children=dcc.Graph(
                                    id="gam_results",
                                    style={"height": "560px"},
                                    config={"displaylogo": False},
                                ),
                            )
                        ],
                        md=8,
                        xs=12,
                        className="mb-0",
                    ),
                    # Scenario table and explainer
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
