import os
import warnings
from dash import Dash, html, dash_table, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from gam_pricing_model import output_key_dfs, viz
import random
from datetime import datetime
import dash_bootstrap_components as dbc


warnings.filterwarnings("ignore")


class get_dfs:

    def __init__(
        self, data_folder="data", pricing_file="730d.csv", product_file="products.csv"
    ):
        """
        must have the 2 files to start with
        """
        # read
        dash_folder = os.path.dirname(os.path.realpath(__file__))
        pricing_folder = os.path.dirname(dash_folder)
        data_folder = os.path.join(pricing_folder, data_folder)
        pricing_path, product_path = os.path.join(
            data_folder, pricing_file
        ), os.path.join(data_folder, product_file)
        # pd dfs
        self.pricing_df, self.product_df = pd.read_csv(pricing_path), pd.read_csv(
            product_path
        )

    def return_dct(self):
        """
        return
            a dictionary of dfs needed to get Dash running
        """
        dct = output_key_dfs(self.pricing_df, self.product_df, 10).initial_dfs()
        price_quant_df, best50, all_gam_results = (
            dct["price_quant_df"],
            dct["best50"],
            dct["all_gam_results"],
        )
        best50_optimal_pricing_df = best50[["product", "asp"]]
        asp_product_topsellers = output_key_dfs(
            self.pricing_df, self.product_df, 10
        ).data_engineer()

        tmp_curr_price = self.product_df.copy()
        tmp_curr_price["product"] = (
            self.product_df["tag"] + " " + self.product_df["weight"].astype(str)
        )
        curr_price_df = tmp_curr_price[["product", "current_price"]]

        elasticity_df = (
            output_key_dfs(self.pricing_df, self.product_df, 10)
            .elasticity()
            .rename(columns={"ratio": "ratio"})[["product", "ratio"]]
            .sort_values(by=["ratio"], ascending=False)
        )
        elasticity_df["ratio"] = round(elasticity_df["ratio"], 2)

        dct_output = {
            "pricing_df": self.pricing_df,
            "product_df": self.product_df,
            "price_quant_df": price_quant_df,
            "best50": best50,
            "all_gam_results": all_gam_results,
            "best50_optimal_pricing_df": best50_optimal_pricing_df,
            "asp_product_topsellers": asp_product_topsellers,
            "elasticity_df": elasticity_df,
            "curr_price_df": curr_price_df,
        }

        return dct_output

#################################### GET DFS ####################################
    
d = get_dfs(
    data_folder="data", pricing_file="730d.csv", product_file="products.csv"
).return_dct()
pricing_df = d["pricing_df"]
product_df = d["product_df"]
price_quant_df = d["price_quant_df"]
best50 = d["best50"]
all_gam_results = d["all_gam_results"]
best50_optimal_pricing_df = d["best50_optimal_pricing_df"]
asp_product_topsellers = d["asp_product_topsellers"]
elasticity_df = d["elasticity_df"]
curr_price_df = d["curr_price_df"]

################################# INITIATE DASH ##################################
app = Dash()

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.LUX]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = html.Div(
    children=[
        html.Br(),
        ################################# HEADER #################################
        html.H1(
            children="Optimal Pricing Modeling",
            style={
                "textAlign": "center",
                "display": "flex",  # flex layout
                "justifyContent": "center",  # Align children with space between (left and right)
                "alignItems": "center",  # Center the content vertically (top and bottom)
                "font-size": "50px",
                # "border": f"1px solid white",
                # "background-color": "#383838"
            },
        ),
        html.Br(),
        ############################ DROP DOWN & CARD ############################
        dbc.Row(
            [
                # Col 1: Dropdown
                dbc.Col(
                    [
                        html.Label(
                            "Select a Product:",
                            style={
                                "textAlign": "center",
                                "margin-left": "70px",
                            },
                        ),
                        dcc.Dropdown(
                            options=price_quant_df["product"].unique(),
                            value=price_quant_df["product"].unique()[0],
                            id="product_dropdown",
                            style={
                                "width": "80%",
                                "margin-left": "20px",
                                # 'border': '1px solid #F5E8D8'
                            },
                        ),
                    ],
                    # Adjust width as needed; leaving room for the hero card on the right
                    width=3,
                ),
                # Col 2: curr price
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    id="card_title_curr_price",
                                    className="kpi-title",
                                    style={
                                        "color": "#121212",
                                        "textAlign": "center",
                                        "margin-bottom": "10px",
                                        "margin-top": "10px",
                                    },
                                ),
                                html.H2(
                                    "Current Price",
                                    className="kpi-eyebrow",
                                    style={
                                        "color": "#121212",
                                        "textAlign": "center",
                                    },
                                ),
                                html.H1(
                                    id="curr_price",
                                    className="kpi-value",
                                    style={
                                        "color": "#DAA520",
                                        "textAlign": "center",
                                    },
                                ),
                            ]
                        ),
                        style={"backgroundColor": "#f3f0f0"},
                    ),
                    width=3,
                    className="kpi-card",
                ),
                # Col 3: reco price
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    id="card_title",
                                    className="kpi-title",
                                    style={
                                        "color": "#121212",
                                        "textAlign": "center",
                                        "margin-bottom": "10px",
                                        "margin-top": "10px",
                                    },
                                ),
                                html.H2(
                                    "Rec. Price",
                                    className="kpi-eyebrow",
                                    style={
                                        "color": "#121212",
                                        "textAlign": "center",
                                        # "font-size":28
                                    },
                                ),
                                html.H1(
                                    id="card_asp",
                                    className="kpi-value",
                                    style={
                                        "color": "#DAA520",
                                        "textAlign": "center",
                                    },
                                ),
                            ]
                        ),
                        style={"backgroundColor": "#F5E8D8"},
                    ),
                    width=3,
                    className="kpi-card",
                ),
                # Col 4: elasticity
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    id="card_title_elasticity",
                                    className="kpi-title",
                                    style={
                                        "color": "#121212",
                                        "textAlign": "center",
                                        "margin-bottom": "10px",
                                        "margin-top": "10px",
                                    },
                                ),
                                html.H2(
                                    "Elasticity",
                                    className="kpi-eyebrow",
                                    style={
                                        "color": "#121212",
                                        "textAlign": "center",
                                    },
                                ),
                                html.H1(
                                    id="elasticity_ratio",
                                    className="kpi-value",
                                    style={
                                        "color": "#DAA520",
                                        "textAlign": "center",
                                    },
                                ),
                            ]
                        ),
                        style={"backgroundColor": "#f3f0f0"},
                    ),
                    width=3,
                    className="kpi-card",
                ),
            ],
            style={"margin-left": "20px", "margin-right": "80px", "margin-top": "20px"},
            align="center",
            justify="center",
        ),
        html.Br(),
        ########################## GRAPH - OPTIMAL PRICE #########################
        # DESCRIPTIONS
        html.H1(
            children="Predictive - Recommended Price at Expected Revenue",
            style={
                "font-size": "30px",
                "textAlign": "center",
                "display": "flex",
                "margin-left": "50px",
                "margin-top": "50px",
            },
        ),
        html.H1(
            children="with Conservative-Optimistic Bands",
            style={
                "font-size": "20px",
                "textAlign": "center",
                "display": "flex",
                "margin-left": "50px",
                # "margin-top": "20px",
            },
        ),
        html.Div(
            children=[
                html.B("◦ Goal: "),
                "Pick the price where we expect to make the most money on average",
                html.I(" (i.e., expected or mean revenue)." ),
            ],
            style={"margin-left": "50px", "margin-top": "2px"},
        ),
        html.Div(
            children=[
                html.B("◦ How: "),
                "Model how sales change with price based on historical data ",
                html.I("(i.e., 'x' markers in graph)." ),
            ],
            style={"margin-left": "50px", "margin-top": "2px"},
        ),
        html.Div(
            children=[
                html.B("◦ Range of scenarios: "),
                "From conservative to optimistic, or from low to higher sales, represented by ",
                html.I("(i.e., the gray band).")
            ],
            style={"margin-left": "50px", "margin-top": "2px"},
        ),
        html.Div(
            children=[
                html.B("◦ Recommended price: "),
                "The peak of the expected/mean revenue curve ",
                html.I("(i.e., the red dot)."),
                "to maximize the expected/mean revenue.",
            ],
            style={"margin-left": "50px", "margin-top": "2px"},
        ),
        html.Div(
            children=[
                html.B("◦ Robusticity: "),
                "Recommended price is the most robust if expected ",
                html.I("(i.e., red dot), "),
                "conservative, and optimistic points ", 
                html.I("(i.e., yellow dots) "),
                "are near the same peak."
            ],
            style={"margin-left": "50px", "margin-top": "2px"},
        ),
        dbc.Row(
            [
                # GRAPH - OPTIMAL PRICING
                dbc.Col(
                    html.Div(
                        dcc.Graph(
                            id="gam_results",
                            figure=viz().gam_results(all_gam_results),
                            style={"margin-left": "50px", "margin-top": "20px"},
                        ),
                    ),
                    # width=6
                ),
            ]
        ),
        ################################ DESCRIPTIVE EDA ###############################
        html.H1(
            children="Descriptive - Product Sales by Price",
            style={
                "font-size": "30px",
                "textAlign": "center",
                "display": "flex",
                "margin-left": "50px",
                "margin-top": "20px",
            },
        ),
        dcc.Graph(
            id="price-quantity",
            figure=viz().price_quantity(price_quant_df),
            style={"margin-left": "50px", "margin-top": "20px"},
        ),
        html.Br(),
        ################################ FOOTNOTE #################################
        html.Div(
            [
                html.Span("made with ♥️ |"),
                html.Span(html.I(" @aqxiao")),
                html.P("github.com/angie-xiao"),
            ],
            style={
                "font-size": "0.8em",
                "color": "#ac274f",
                "textAlign": "center",
                "background-color": "#f3f3f3",
            },
        ),
    ]
)

print(
    "\n",
    "-" * 10,
    datetime.now().strftime("%H:%M:%S"),
    " Page Updated " + "-" * 10,
    "\n",
)


@callback(
    Output("card_title_curr_price", "children"),
    Output("curr_price", "children"),
    Output("card_title", "children"),
    Output("card_asp", "children"),
    Output("card_title_elasticity", "children"),
    Output("elasticity_ratio", "children"),
    Output("gam_results", "figure"),
    Output("price-quantity", "figure"),
    # Output("elasticity_table", "data"),
    # Output("elasticity_graph", "figure"),
    Input("product_dropdown", "value"),
)
def update_figure(val):
    """ """
    filtered_pricing = price_quant_df[price_quant_df["product"] == val]
    filtered_gam = all_gam_results[all_gam_results["product"] == val]

    filtered_table = best50_optimal_pricing_df[
        best50_optimal_pricing_df["product"] == val
    ]

    price_quant_fig = viz().price_quantity(filtered_pricing)
    gam_res_fig = viz().gam_results(filtered_gam)

    card_title = filtered_table["product"]
    card_title_elasticity = card_title.copy()
    card_asp = "$" + (filtered_table["asp"]).astype(str)
    card_title_curr_price = card_title.copy()
    curr_price = "$" + (
        curr_price_df[curr_price_df["product"] == val]["current_price"]
    ).astype(str)

    elasticity_ratio = elasticity_df[elasticity_df["product"] == val]["ratio"]

    return (
        card_title_curr_price,
        curr_price,
        card_title,
        card_asp,
        card_title_elasticity,
        elasticity_ratio,
        gam_res_fig,
        price_quant_fig,
    )


if __name__ == "__main__":
    app.run(debug=True)


"""
ref:
https://dash.plotly.com/tutorial
https://medium.com/@wolfganghuang/advanced-dashboards-with-plotly-dash-things-to-consider-before-you-start-9754ac91fd10

# activate pyvenv
pyenv shell pricing-venv

# in terminal:
cd dash


# call the app
py app.py # winddows
python3 app.py  # mac


host:
http://127.0.0.1:8050/
"""
