import os
import warnings
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
from  gam_pricing_model import *

warnings.filterwarnings('ignore')



# def path_finder(foldername, filename):
#     '''
#     return file path based on user OS path
#     '''
#     current_directory = os.getcwd()
#     base_folder = os.path.dirname(current_directory)
#     data_folder = os.path.join(current_directory, foldername)
#     file_path = os.path.join(data_folder, filename)

#     # file_path = os.path.join(os.path.dirname(__file__), foldername, filename)

#     return file_path


def read_dfs(data_folder_path, product_file, pricing_file):        
    # df_path = path_finder(data_folder_path, pricing_file)
    # tags_path = path_finder(data_folder_path, product_file)

    # df = pd.read_csv(df_path)
    # tags = pd.read_csv(tags_path)

    df_path = f"../{data_folder_path}/{pricing_file}"
    df = pd.read_csv(df_path)
    
    tag_path = f"../{data_folder_path}/{product_file}"
    tags = pd.read_csv(tag_path)

    d = {
        'pricing': df,
        'product_tags': tags
    }
    
    print('-' * 10, ' finished reading input files ', '-' * 10)
    
    return d


def update_graph(folder, filename):
    '''
    Draw graph
    
    params
        folder: (sub) folder where data file is
        filename: data file to graph
    '''
    # data_path = path_finder(folder, filename)
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


# read dfs
df_dict = read_dfs(
    data_folder_path='data', 
    pricing_file='730d.csv',
    product_file='products.csv',
)
df = df_dict['pricing']
tags = df_dict['product_tags']


d = optimization(df,tags)
best50 = d['best50']       # get data


# initiate Dash
app = Dash()


# App layout
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=update_graph(folder='data', filename='price_quantity.csv')         # input by VM
    )
    
])


# app.layout = [
#     # header
#     html.Div(className='row', 
#              children='Exploratory Data Analysis',
#              style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),

#     # interactive scatter/line graph (price & quantity)
#     html.Div(className='row', children=[
#         dcc.Graph(
#             id='example-graph',
#             figure=update_graph(folder='data', filename='price_quantity.csv')         # input by VM
#         )
#     ]),

#     # optimal pricing table
#     html.Div(className='row', children=[
#         html.Div(className='Optimal Pricing', children=[
#             dash_table.DataTable(
#                 data=best50.to_dict('records'), page_size=11, 
#                 style_table={'overflowX': 'auto'}
#             )
#         ])
#     ])
# ]



if __name__ == '__main__':
    app.run(debug=True)



'''
https://dash.plotly.com/tutorial

# in terminal:
cd dash
py app.py
'''