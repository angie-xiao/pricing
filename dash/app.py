
import os
import warnings
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
from  gam_pricing_model import output_key_dfs, viz
import random
from datetime import datetime
 

warnings.filterwarnings('ignore')

    
#################################### READ DFS ####################################

dash_folder = os.path.dirname(os.path.realpath(__file__))
pricing_folder = os.path.dirname(dash_folder)
data_folder = os.path.join(pricing_folder, 'data')
pricing_path, tags_path = os.path.join(data_folder, '730d.csv'), os.path.join(data_folder, 'products.csv')
pricing_df,tags_df = pd.read_csv(pricing_path),pd.read_csv(tags_path)

price_quant_df = output_key_dfs(pricing_df,tags_df,10).price_quant()    # key df #1
d = output_key_dfs(pricing_df, tags_df,10).optimization()               # modeling + optimization
best50 = d['best50']                                                    # key df #2
all_gam_results = d['all_gam_results']                                  # key df #3

################################# INITIATE DASH ##################################
app = Dash()

# App layout
app.layout = html.Div(children=[
    
    # Scatter/line: price & quantity
    html.H1(children='Exploratory Data Analysis'),
    dcc.Graph(
        id='price_quantity',
        figure=viz().price_quantity(price_quant_df)
    ),


    # GAM opt results
    html.H1(children='GAM Optimization Result for Top 10 Best Sellers'),

    html.Div(children="""
        - Gray band: 95% confidence interval for the predicted revenue.\n
    """),
    html.Div(children="""
        - Blue points: maximum revenues at 2.5% percentile and 97.5% percentile.\n
    """),
    html.Div(children="""
        - Red spot: optimized price - this is where the median predicted revenue is at its maximum value.
    """),
    dcc.Graph(
        id='gam_results',
        figure=viz().gam_results(all_gam_results)
    ),
    
        
    # p50 max rev with optimal pricing table
    html.H1(children='Optimal Prices by Product Table'),
    html.Div(children='''
        Median Predicted Revenue at Maximum Value.
    '''),
    dash_table.DataTable(data=best50.to_dict('records'), page_size=10)
    
])

print('\n', '-'*10 , datetime.now().strftime("%H:%M:%S"), ' Page Updated ' + '-'*10, '\n')

if __name__ == '__main__':
    app.run(debug=True)



'''
ref:
https://dash.plotly.com/tutorial

# in terminal:
cd dash

# windows
py app.py

# mac
python3 app.py

host:
http://127.0.0.1:8050/
'''
