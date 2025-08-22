import os
import warnings
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
from  gam_pricing_model import *
import random
from datetime import datetime
 

warnings.filterwarnings('ignore')


def read_dfs(data_folder_path, product_file, pricing_file):        

    df_path = f"../{data_folder_path}/{pricing_file}"
    df = pd.read_csv(df_path)
    
    tag_path = f"../{data_folder_path}/{product_file}"
    tags = pd.read_csv(tag_path)

    d = {
        'pricing': df,
        'product_tags': tags
    }
    
    # print('\n', '-' * 10, ' finished reading input files ', '-' * 10, '\n')
    
    return d


def price_quantity_graph(folder, filename):
    '''
    Draw price & quantity scatterplot graph
    
    params
        folder: (sub) folder where data file is
        filename: data file to graph
    '''

    data_path = f"../{folder}/{filename}"
    asp_product_topsellers = pd.read_csv(data_path)
    
    # aggregate
    price_quantity = asp_product_topsellers[['asp','shipped_units','product']].groupby(
        ['asp','product']
    ).sum().reset_index()

    # plot
    fig = px.scatter(
            price_quantity,
            x='asp',
            y='shipped_units',
            log_y=True,
            color='product',
            # opacity=.5,
            width=1200,
            height=600,
            trendline='lowess', # used when the relationship is curved
            trendline_color_override='blue',
            title='Product Sales: Price vs Shipped Units'
        ).update_traces(
            marker=dict(size=7)
        ).update_layout(
            legend_title_text='Product',
            yaxis_range=[0, None]
        ).update_xaxes(
            title_text='Price'
        ).update_yaxes(
            title_text='Shipped Units'
        )

    return fig

    
################################# READ DFS #################################
df_dict = read_dfs(
    data_folder_path='data', 
    pricing_file='730d.csv',
    product_file='products.csv',
)
df = df_dict['pricing']
tags = df_dict['product_tags']

# modeling + optimization
d = optimization(df,tags)
best50 = d['best50']
# best_975 = d['best975']
# best_025 = d['best25']
all_gam_results = d['all_gam_results']
    

################################# initiate Dash #################################
app = Dash()

# App layout
app.layout = html.Div(children=[
    
    # Scatter/line: price & quantity
    html.H1(children='Exploratory Data Analysis'),
    dcc.Graph(
        id='price_quantity',
        figure=price_quantity_graph(folder='data', filename='price_quantity.csv')         # input by VM
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
        figure=viz_gam_results(all_gam_results)
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
py app.py

host:
http://127.0.0.1:8050/
'''