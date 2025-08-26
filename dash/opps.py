# opps.py
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc

from dash import html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd

def layout(opp_table_df):
    # Build columns robustly whether DF or list-of-dicts
    if isinstance(opp_table_df, pd.DataFrame):
        cols = [{"name": c.replace("_", " ").title(), "id": c} for c in opp_table_df.columns]
        data = opp_table_df.to_dict("records")
    elif isinstance(opp_table_df, list) and opp_table_df and isinstance(opp_table_df[0], dict):
        keys = list(opp_table_df[0].keys())
        cols = [{"name": k.replace("_", " ").title(), "id": k} for k in keys]
        data = opp_table_df
    else:
        # empty or unexpected input: show empty table
        cols, data = [], []

    return dbc.Container(
        [
            html.H2(
                "Top Pricing Optimization Opportunities",
                className="mt-3",
                style={"textAlign": "center", "padding": "30px 0"},
            ),
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id="opp_table",
                        columns=cols,
                        data=data,
                        sort_action="native",
                        page_size=12,
                        style_table={"width": "100%", "overflowX": "auto", "border": "none"},
                        style_cell={"padding": "12px", "fontSize": "14px", "border": "none", "textAlign": "center"},
                        style_header={"fontWeight": 600, "border": "none", "backgroundColor": "#f6f6f6", "textAlign": "center"},
                    ),
                    width=12, lg=10, className="mx-auto",
                ),
            ),
        ],
        fluid=True,
        className="px-3",
        style={"maxWidth": "1500px", "margin": "0 auto"},
    )



def register_callbacks(app, opp_table_df):
    @app.callback(
        Output("opp_table", "data"),
        Input("opp_table", "id"),   # dummy input so callback runs once
        prevent_initial_call=False
    )
    def _fill_table(_):
        # convert df to records for DataTable
        return opp_table_df.to_dict("records")