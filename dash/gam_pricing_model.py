# """
# # in terminal:

# .venv\Scripts\activate.bat

# pip3 install scikit-learn
# pip3 install plotnine
# """

import os
import random

# Data Wrangling
import pandas as pd
import numpy as np

# Modeling
from sklearn.preprocessing import LabelEncoder
from pygam import GAM, ExpectileGAM, s, l, f
import statsmodels.api as sm

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
from plotnine import ggplot, aes, geom_ribbon, geom_line,facet_wrap,labs,theme,geom_point
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly as ggplotly



def data_engineer(df,tags,top_n):
    """
    """

    # formatting
    df.columns = [x.lower() for x in df.columns]
    tags.columns = [x.lower() for x in tags.columns]

    # merge
    df_tags = df.merge(
        tags, how='left', on='asin'
    )
    
    # add a product tag, + weight
    df_tags['product'] = df_tags['tag'] + ' ' + df_tags['weight'].astype(str)


    # date
    df_tags['order_date'] = pd.to_datetime(df_tags['order_date'])
    df_tags['week_num'] = df_tags['order_date'].dt.isocalendar().week
    df_tags['year'] = df_tags['order_date'].dt.year

    # aggregate on weekly basis
    weekly_df_tags = df_tags[['week_num', 'year', 'asin', 'item_name', 'tag', 'event_name', 'product', 'shipped_units', 'revenue_share_amt']].groupby(
        ['week_num', 'year', 'asin', 'item_name', 'tag', 'product','event_name']
    ).sum().reset_index()

    # calculate aps
    weekly_df_tags['asp'] = weekly_df_tags['revenue_share_amt'] / weekly_df_tags['shipped_units']
    weekly_df_tags = weekly_df_tags[weekly_df_tags['asp']!=0]

    # 1 decimal
    asp_product = weekly_df_tags.copy()
    asp_product['asp'] = round(asp_product['asp'],1)
    asp_product_df = asp_product[['product', 'asp', 'tag', 'event_name', 'shipped_units', 'revenue_share_amt']]\
                                .groupby(['tag','product', 'asp', 'event_name']).sum().reset_index()

    # data points where shipped units < 10 (too little evidence)
    asp_product_df = asp_product_df[asp_product_df['shipped_units']>=10]
    
    # only focus on top sellers
    top10_product_lst = asp_product_df[['product', 'revenue_share_amt']].groupby('product').sum().reset_index().sort_values(by=['revenue_share_amt'],ascending=False)['product'][:top_n].tolist()

    # filter for topsellers
    asp_product_topsellers = asp_product_df[
        asp_product_df['product'].isin(top10_product_lst)
    ]

    # print('-' * 10, ' finished data engineering', '-'*10, '\n')
    
    return asp_product_topsellers


def model(df,tags):
    
    df_tags = data_engineer(df, tags, top_n=10)

    data_filtered = df_tags[df_tags['event_name']=='NO DEAL']

    unique_prod = data_filtered['product'].unique()

    all_gam_results = pd.DataFrame()


    # Loop through products
    for product in unique_prod:

        if product and str(product) == product: # skip nan/none
            
            product_data = data_filtered[data_filtered['product']==product] # Filter for current product

            # Predictors & target split
            X = product_data[['asp']]
            y = product_data['shipped_units']

            # List of quantiles for modeling
            quantiles = [0.025, 0.5, 0.975]
            gam_results = {}

            # Fit the GAM model
            for q in quantiles:
                gam = ExpectileGAM(s(0), expectile=q) # initiate the model
                gam.fit(X,y) #fit
                gam_results[f'pred_{q}'] = gam.predict(X) # predict for that quantile
                # print(q, "|", product, "|", gam.deviance_residuals(X,y).mean())
            # print("-----------\n")

            # Store the results in a DF
            predictions_gam = pd.DataFrame(gam_results).set_index(X.index)
            predictions_gam_df = pd.concat([product_data[['asp', 'product','shipped_units']], predictions_gam], axis=1)
            all_gam_results = pd.concat([all_gam_results, predictions_gam_df], axis=0)
        
    # print("-"*10, ' finished modeling ', "-"*10, "\n")

    return all_gam_results


def optimization(df,tags):
    '''
    return a dictionary of best 2.5%, 50%, 97.5% dfs
    '''
    all_gam_results = model(df,tags)

    # Calculate Revenue for each predicted price band
    for col in all_gam_results.columns:
        if col.startswith('pred'):
            all_gam_results['revenue_' + col] = all_gam_results['asp'] * all_gam_results[col]

    # Actual revenue
    all_gam_results['revenue_actual'] = all_gam_results['asp'] * all_gam_results['shipped_units']

    # View
    # all_gam_results.sample(2)


    # Calculating where the predicted median revenue is the max
    best_50 = (
        all_gam_results
        .groupby('product')
        .apply(lambda x: x[x['revenue_pred_0.5'] == x['revenue_pred_0.5'].max()].head(1))
        .reset_index(level=0, drop=True)
    )

    # Calculating where the predicted 97.5% percentile revenue is the max
    best_975 = (
        all_gam_results
        .groupby('product')
        .apply(lambda x: x[x['revenue_pred_0.975'] == x['revenue_pred_0.975'].max()].head(1))
        .reset_index(level=0, drop=True)
    )

    # Calculating where the predicted 2.5% percentile revenue is the max
    best_025 = (
        all_gam_results
        .groupby('product')
        .apply(lambda x: x[x['revenue_pred_0.025'] == x['revenue_pred_0.025'].max()].head(1))
        .reset_index(level=0, drop=True)
    )
    
    d = {
        'best50':best_50,
        'best975': best_975,
        'best25':best_025,
        'all_gam_results':all_gam_results
    }
    
    # print("-"*10, ' finished pricing optimization ', "-"*10, "\n")
    
    return d


def viz_gam_results(all_gam_results):
    """
    params
        df: pricing df
        tags: product df

    return
        ggplot obj
    """

    # df_dct = optimization(df, tags)
    # best_50 = df_dct['best50']
    # best_975 = df_dct['best975']
    # best_025 = df_dct['best25']
    # all_gam_results = df_dct['all_gam_results']

    # map color to product
    product_lst = all_gam_results['product'].unique()
    pltly_qual = (px.colors.qualitative.Dark24) # color palette
    pltly_qual.extend(px.colors.qualitative.Light24)
    colors = random.sample(pltly_qual, len(product_lst))
    
    color_dct = dict()

    i = 0
    while i < len(product_lst):
        color_dct[product_lst[i]] = colors[i]
        i+=1

    # plot
    fig = go.Figure()

    for group_name, group_df in all_gam_results.groupby('product'):
        
        # scatter plot (actual)
        fig.add_trace(go.Scatter(
            x=group_df['asp'],
            y=group_df['revenue_pred_0.5'],
            mode='markers',
            name=group_name,  # Name for the legend
            marker=dict(color=color_dct[group_name], size=10),
            legendgroup=group_name,
            )
        )
        
        # error band
        fig.add_trace(go.Scatter(
            name=f"group {group_name} error",
            x=group_df['asp'].tolist() + group_df['asp'].tolist()[::-1], # x, then x reversed
            y=group_df['revenue_pred_0.975'].tolist() + group_df['revenue_pred_0.025'].tolist()[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            legendgroup=group_name,
            showlegend=False    
            )
        )

    fig.update_layout(
        yaxis_type="log",
        title='BAU GAM Results',
        width=1200,
        height=650,
    )

    return fig

