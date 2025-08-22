# """
# # in terminal:

# .venv\Scripts\activate.bat

# pip3 install scikit-learn
# pip3 install plotnine
# """

import os

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



def data_engineer(df,tags):
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
    
    print('finished data engineering')
    
    return df_tags


def model(df,tags):
    
    df_tags = data_engineer(df, tags)

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
                print(q, "|", product, "|", gam.deviance_residuals(X,y).mean())
            print("-----------\n")

            # Store the results in a DF
            predictions_gam = pd.DataFrame(gam_results).set_index(X.index)
            predictions_gam_df = pd.concat([product_data[['asp', 'product','shipped_units']], predictions_gam], axis=1)
            all_gam_results = pd.concat([all_gam_results, predictions_gam_df], axis=0)
        
    print("-"*10, ' finished modeling ', "*"*10, "\n")

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
    }
    
    print("-"*10, ' finished pricing optimization ', "-"*10, "\n")
    
    return d

##################################### output #####################################

# # Visualize the GAM Optimization Result
# (ggplot(
#     # Data
#     data = all_gam_results,
#     # Axes
#     mapping = aes(x='our_price', y='revenue_pred_0.5', color='product', group='product') ) + 
# # Adding the Band
# geom_ribbon(aes(ymax= 'revenue_pred_0.975', ymin= 'revenue_pred_0.025'), 
#                 fill='#d3d3d3', color= '#FF000000', alpha=0.7, show_legend=False) +
# # Adding the points
# geom_point(aes(y='revenue_actual'), alpha=0.15, color="#2C3E50") +
# # Adding 50th percentile line
# geom_line(aes(y='revenue_pred_0.5'), alpha=0.5, color='darkred') +
# # Addimg the 50th pct points
# geom_point(data=best_50, color='red') + 
# # Addimg the 97th pct points
# geom_point(data=best_975, mapping= aes(y='revenue_pred_0.975'), color='blue') + 
# # Addimg the 2.5th pct points
# geom_point(data=best_025, mapping= aes(y='revenue_pred_0.025'), color='blue') + 
# # Wraps by product
# facet_wrap('product', scales='free') + 
# # Labels
# labs(
#     title='Price Optimization',
#     subtitle='Maximum median revenue (red point) vs 95% Maximum Confidence Interval',
#     x= 'our_price',
#     y= 'Predicted Revenue'
#     ) +
# theme(figure_size=(12,7))

# )


if __name__ == '__main__':
    optimization(
        data_folder_path='data', 
        product_file='730d.csv', pricing_file='products.csv')
#
