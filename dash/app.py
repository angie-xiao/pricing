import os
import warnings
from dash import Dash, html, dash_table, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from  gam_pricing_model import output_key_dfs, viz
import random
from datetime import datetime
import dash_bootstrap_components as dbc


warnings.filterwarnings('ignore')

    
#################################### READ DFS ####################################
dash_folder = os.path.dirname(os.path.realpath(__file__))
pricing_folder = os.path.dirname(dash_folder)
data_folder = os.path.join(pricing_folder, 'data')
pricing_path, product_path = os.path.join(data_folder, '730d.csv'), os.path.join(data_folder, 'products.csv')
pricing_df,product_df = pd.read_csv(pricing_path),pd.read_csv(product_path)

dct = output_key_dfs(pricing_df,product_df,10).initial_dfs()
price_quant_df, best50, all_gam_results = dct['price_quant_df'], dct['best50'], dct['all_gam_results']
best50_optimal_pricing_df = best50[['product', 'asp']]

################################# INITIATE DASH ##################################
app = Dash()

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.LUX]
app = Dash(__name__, external_stylesheets=external_stylesheets)
# app = Dash(external_stylesheets=[dbc.themes.LUX])

# dropdown menu items
menu_opt = []
for item in price_quant_df['product'].unique():
    menu_opt.append(dbc.DropdownMenuItem(item, id=item))

# input_lst = []
# for item in price_quant_df['product'].unique():
#     input_lst.append(Input(item, "n_clicks"))

# App layout
app.layout = html.Div(children=[

    html.Br(),

    ############################## HEADER ##############################
    html.H1(
        children='Optimal Pricing Modeling',
        style={
            'textAlign': 'center',
            "display": "flex",                      # flex layout
            "justifyContent": "center",             # Align children with space between (left and right)
            "alignItems": "center",                 # Center the content vertically (top and bottom)
            # "border": f"1px solid white",
            # "background-color": "#383838"
        }
    ),
    html.Br(),


    ############################### DROP DOWN & CARD ###############################
    dbc.Row(
        [
            # Left column: Label + Dropdown stacked
            dbc.Col(
                [
                    html.Label(
                        'Select a Product:',
                        style={'textAlign': 'center', 
                            'margin-left': "20px",
                        }
                    ),
                    dcc.Dropdown(
                        options=price_quant_df['product'].unique(),
                        value=price_quant_df['product'].unique()[0],
                        id='product_dropdown',
                        style={
                            'width': '70%',
                            # 'margin-left': "20px",
                            # 'border': '1px solid #ccc'
                        }
                    ),
                ],
                # Adjust width as needed; leaving room for the hero card on the right
                width=3,
            ),

            # Right column: KPI Card (unchanged content/styles)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                "Optimal Price", className="kpi-eyebrow",
                                style={'color': '#121212', 'textAlign': 'center',}
                            ),
                            html.H2(
                                id="card_title", className="kpi-title",
                                style={'color': '#121212', 'textAlign': 'center',
                                       'margin-bottom': "10px", "margin-top": "10px"}
                            ),
                            html.H1(
                                id="card_asp", className="kpi-value",
                                style={'color': '#DAA520', 'textAlign': 'center',}
                            ),
                        ]
                    ),
                    style={"backgroundColor": "#F5E8D8"},
                ),
                width=3,
                className="kpi-card",
            ),
        ],
        style={'margin-left': "20px", "margin-top": "20px"},
        align="center",
        justify="center",
    ),

    ################################ EDA SCATTER ###############################
    html.H1(
        children='Product Sales by Price',
        style={
            "font-size":"30px",
            'textAlign': 'center',
            "display": "flex",
            'margin-left':"50px", "margin-top":"20px"
        }
    ),
    dcc.Graph(
        id='price-quantity',
        figure=viz().price_quantity(price_quant_df),
        style = {
            'margin-left':"50px", "margin-top":"20px"
        }
    ),
    html.Br(),

    ############################## GRAPH & TABLE - OPTIMAL PRICE ##############################

    # DESCRIPTIONS
    html.H1(
        children='Optimal Pricing Point for Top Products',
        style={
            "font-size":"30px",
            'textAlign': 'center',
            "display": "flex",
            'margin-left':"50px", "margin-top":"20px"
        }
    ),
    # sub descriptions
    html.Div(
        children="- Gray band: 95% confidence interval for the predicted revenue.",
        style={ 'margin-left':"50px", "margin-top":"10px"}
    ),
    html.Div(
        children="- Blue points: maximum revenues at 2.5% percentile and 97.5% percentile.",
        style={ 'margin-left':"50px", }
    ),
    html.Div(
        children="- Red spot: optimized price - this is where the median predicted revenue is at its maximum value.",
        style={ 'margin-left':"50px", "margin-bottom": "10px"}
    ),
    
    dbc.Row([
            # GRAPH - OPTIMAL PRICING
            dbc.Col(
                html.Div(
                    dcc.Graph(
                        id='gam_results',
                        figure=viz().gam_results(all_gam_results),
                        style = {
                            'margin-left':"50px", "margin-top":"20px"
                        }
                    ),
                ),
                # width=6
            ), 

    ]),

])

print('\n', '-'*10 , datetime.now().strftime("%H:%M:%S"), ' Page Updated ' + '-'*10, '\n')


@callback(
    Output('price-quantity','figure'),
    Output('gam_results', 'figure'),
    Output("card_title", "children"),
    Output("card_asp", "children"),
    Input("product_dropdown", "value") 
)

def update_figure(val):
    ''' '''
    filtered_pricing = price_quant_df[price_quant_df['product'] == val]
    filtered_gam = all_gam_results[all_gam_results['product']==val]

    filtered_table = best50_optimal_pricing_df[best50_optimal_pricing_df['product']==val]

    fig1 = viz().price_quantity(filtered_pricing)
    fig2 = viz().gam_results(filtered_gam)

    new_product = filtered_table['product']
    new_asp = "$" + (filtered_table['asp']).astype(str)

    # filterd_table = filtered_table.to_dict('records')

    return fig1,fig2, new_product, new_asp



if __name__ == '__main__':
    app.run(debug=True)



'''
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
'''
