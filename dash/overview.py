# overview.py
from dash import Input, Output
from helpers import (
    Style,                        # colors / constants
    OverviewUI as UI,             # kpi_card
    Metrics,                      # model_fit_units, update_elasticity_kpi_by_product
    Scenario,                     # scenario_table
    Notes,                        # coverage_note
    OverviewHelpers as OH,        # callback micro-helpers you already wired
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
        Output("gam_results_dual", "figure"),
        Output("scenario_table", "data"),
        Output("coverage_note", "children"),
        Input("product_dropdown_snap", "value"),
    )
    def overview(asin):
        if not asin:
            return OH.empty_overview_payload(viz)

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
        fig_dual = OH.dual_graph(viz, filt)

        # Scenario table (combined revenue + units)
        scenario_data = OH.scenario_records(filt)

        # Annualized KPI deltas (use pipeline’s annual_factor)
        du_ann, rev_best_ann = OH.annualized_kpis(
            asin, best50_optimal_pricing_df, curr_price_df, all_gam_results, meta.get("annual_factor", 1.0)
        )

        # Fit & coverage
        fit_val, fit_sub, cov = OH.fit_and_coverage(filt)

        return (
            "",                         # card_title_date_range (title is in layout)
            date_range_val,             # date_range_value
            date_range_subtext,         # date-range-subtext
            "",                         # card_title_curr_price_snap
            curr_price_val,             # curr_price_snap
            "",                         # card_title_snap
            asp_val,                    # card_asp_snap
            "",                         # card_title_elasticity_snap
            elast_val,                  # elasticity_ratio_snap
            elast_sub,                  # elasticity-subtext
            "",                         # card_title_units_opp_ann
            du_ann,                     # units_opp_ann_value
            "",                         # card_title_rev_best_ann
            rev_best_ann,               # rev_best_ann_value
            "",                         # card_title_fit_snap
            fit_val,                    # fit_value_snap
            fit_sub,                    # fit-subtext
            fig_dual,                   # gam_results_dual
            scenario_data,              # scenario_table (combined)
            cov,                        # coverage_note
        )