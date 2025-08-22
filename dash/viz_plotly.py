# %% 

'''
in VSCode:
create a venv


in terminal:
activate: .\.venv\Scripts\activate

pip3 install 
pip3 install pygam
pip3 install ipykernel
pip3 install --upgrade nbformat

# pip3 install chart_studio
# pip3 install plotly --upgrade
# pip3 install --upgrade pandas numpy matplotlib

'''

# %%
import pandas as pd
import chart_studio
import chart_studio.plotly as py
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.io as pio


def return_fig():
    # read df
    asp_product_topsellers = pd.read_csv('../data/price_quantity.csv')

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

# read creds
# plty_creds = list(py.get_credentials().values())
# chart_studio.tools.set_credentials_file(
#     username=plty_creds[0], api_key=plty_creds[1]
# )
# chart_studio.tools.set_config_file(
#     world_readable=False, sharing='private'
# )

# export html
# pio.write_html(fig, file='ProductvsQuantity.html', auto_open=True)

