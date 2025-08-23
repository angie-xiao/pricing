import os
import warnings
from dash import Dash, html, dash_table, dcc
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

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')


################################# INITIATE DASH ##################################
app = Dash()

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = html.Div(children=[
    
    # Scatter/line: price & quantity
    html.H1(children='Exploratory Data Analysis'),
    dcc.Graph(
        id='price_quantity',
        figure=viz('cyborg').price_quantity(price_quant_df)
    ),

    html.H1(children='Optimal Pricing Point by Product'),
    dbc.Row([
            dbc.Col(
                dash_table.DataTable( 
                    fill_width=True,
                    id = 'best50_optimal_pricing',
                    data = best50_optimal_pricing_df.to_dict('records'),
                    columns = [{'id': c, 'name':c} for c in best50_optimal_pricing_df.columns],
                    # style
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Date', 'Region']
                    ],
                    style_as_list_view=True,
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto', 'lineHeight': '15px'
                    },
                ),    
                width=3, 
                # lg={'size': 6,  "offset": 0, 'order': 'first'})
            ),
            dbc.Col(
                html.Div(
                    dcc.Graph(
                        id='gam_results',
                        figure=viz('cyborg').gam_results(all_gam_results)
                    ),
                    # style={'width':'30%'}
                ),
                width=6
            )
    ]),

    # # GAM opt results
    # html.H1(children='GAM Optimization Result for Top 10 Best Sellers'),

    # html.Div(children="""
    #     - Gray band: 95% confidence interval for the predicted revenue.\n
    # """),
    # html.Div(children="""
    #     - Blue points: maximum revenues at 2.5% percentile and 97.5% percentile.\n
    # """),
    # html.Div(children="""
    #     - Red spot: optimized price - this is where the median predicted revenue is at its maximum value.
    # """),
    # dcc.Graph(
    #     id='gam_results',
    #     figure=viz().gam_results(all_gam_results)
    # ),
    
        
    # # p50 max rev with optimal pricing table
    # html.H1(children='Optimal Prices by Product Table'),
    # html.Div(children='''
    #     Median Predicted Revenue at Maximum Value.
    # '''),
    # dash_table.DataTable(
    #     data=best50.to_dict('records'), 
    #     page_size=10,
    #     style_cell_conditional=[
    #         {
    #             'if': {'column_id': c},
    #             'textAlign': 'left'
    #         } for c in ['Date', 'Region']
    #     ],
    #     style_as_list_view=True,
    # )
])

print('\n', '-'*10 , datetime.now().strftime("%H:%M:%S"), ' Page Updated ' + '-'*10, '\n')

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
