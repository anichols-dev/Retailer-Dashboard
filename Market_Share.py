#%%    SETUP FOR NEIGHBORHOOD ADJACENCY 

#                 INPUTS
#       pois.pkl:                                              Dictionary of POI data from Populate_Adjacents.py
#       adjacency_dictionary.pkl:                              Dictionary of Adjacency data from Populate_Adjacents.py
#       recommendation_1.csv,r...n_2, r...n_3:                 CSV for each Neighborhood that holds neighborhood share information (share brand sales, share adjacent sales, etc) from Populate_Adjacents.py

#                 OUTPUT
#      n_adj_plots ('neighborhood_adjacents_dictionary.pkl):    Dictionary of Neighborhood Adjacent Calculations  

#%%
import pandas as pd
from scipy.cluster.hierarchy import ward, fcluster, linkage, dendrogram
from scipy.spatial.distance import pdist, euclidean, cosine, correlation
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import pickle
import datetime
import plotly.express as px
from collections import defaultdict



global adjacency_dictionary
global pois
global by_date_data
global t_stats
global number_of_neighborhoods

test_date = datetime.date(2020, 5, 28)
number_of_neighborhoods = 2

#load data
with open('assets/pois.pkl', 'rb') as file:
    pois = pickle.load(file)
with open('assets/adjacency_dictionary.pkl', 'rb') as file:
    adjacency_dictionary = pickle.load(file)
with open('assets/clustered_by_date.pkl', 'rb') as file:
    clustered_by_date = pickle.load(file)

# Load the Neighborhood breakdown data to select top 80
clustered_recommendations = []
for clstr in range(len(pois)):
    t_stats = [] #list of each recommendation table, index is neighborhood
    if clstr == len(pois)-1:
        for nbr in range(number_of_neighborhoods): 
            df = pd.read_csv(f'assets/recommendation_{nbr}.csv')
            t_stats.append(df)
    else:
        for nbr in range(number_of_neighborhoods): 
            df = pd.read_csv(f'assets/Cluster_{clstr}_recommendation_{nbr}.csv')
            t_stats.append(df)
    clustered_recommendations.append(t_stats)

clustered_neighborhood_products = []
for i in range(len(pois)):
    neighborhood_products = []
    for y in clustered_recommendations[i]:
        neighborhood_products.append(list(y['Unnamed: 0']))
    clustered_neighborhood_products.append(neighborhood_products)

#%% COMPUTE DATA FOR NIEGHBORHOODS COMBINED

clustered_monthly_sales = []
clustered_universal_sales = []

for i in range(len(pois)):
    # Initialize a dictionary to store aggregated data
    monthly_sales = None
    monthly_sales = defaultdict(lambda: defaultdict(float))
    # Iterate through each date in the dictionary
    for date, products_dict in clustered_by_date[i].items():
        month = pd.to_datetime(date).strftime('%Y-%m')
        
        # Iterate through each neighborhood and its products
        for nbr, nlist in enumerate(clustered_neighborhood_products[i]):
            # Sum the dollar sales for the neighborhood in the current month
            total_sales = 0
            for product in nlist:
                if product in products_dict:
                    total_sales += products_dict[product]['dollar_sales'].sum()
            
            # Add the sales to the monthly_sales dictionary
            monthly_sales[month][nbr] += total_sales

    # Convert the monthly_sales dictionary to a DataFrame
    universe_sales_df = None
    universe_sales_df = pd.DataFrame(monthly_sales).T.fillna(0)
    universe_sales_df.index.name = 'Month'
    universe_sales_df.reset_index(inplace=True)

    # Calculate the total sales per month
    universe_sales_df['Total_Sales'] = universe_sales_df.select_dtypes(include='number').sum(axis=1)

    # Calculate the percentage share for each neighborhood
    for count, nbr in enumerate(clustered_neighborhood_products[i]):
        universe_sales_df[f'{count}_Share'] = (universe_sales_df[count] / universe_sales_df['Total_Sales']) * 100
    clustered_universal_sales.append(universe_sales_df)


#%% NOW COMPUTE FOR EACH NEIGHBORHOOD

clustered_by_nbr_sales = []

for i in range(len(pois)):
    by_nbr_sales = {}
    for nbr, neighborhood_product_list in enumerate(clustered_neighborhood_products[i]):
        # Initialize a dictionary to store aggregated data
        nbr_sales = defaultdict(lambda: defaultdict(float))

        # Iterate through each date in the dictionary
        for date, products_dict in clustered_by_date[i].items():
            month = pd.to_datetime(date).strftime('%Y-%m')
            
            # Iterate through each neighborhood and its products
            for prod in neighborhood_product_list:
                # Sum the dollar sales for the neighborhood in the current month
                total_sales = 0
                if prod in products_dict:
                    total_sales += products_dict[prod]['dollar_sales'].sum()
                # Add the sales to the monthly_sales dictionary
                    nbr_sales[month][nbr] += total_sales
                    nbr_sales[month][prod] += total_sales

        # Convert the monthly_sales dictionary to a DataFrame
        nbr_sales = pd.DataFrame(nbr_sales).T.fillna(0)
        nbr_sales.index.name = 'Month'
        nbr_sales.reset_index(inplace=True)

        # Calculate the total sales per month
        nbr_sales['Total_Sales'] = nbr_sales.select_dtypes(include='number').sum(axis=1)
        by_nbr_sales[nbr] = nbr_sales

    # Calculate the percentage share for each neighborhood
    for count in by_nbr_sales:
        for nbr in clustered_neighborhood_products[i][count]:
            if nbr in by_nbr_sales[count]:   
                by_nbr_sales[count][f'{nbr}_Share'] = (by_nbr_sales[count][nbr] / by_nbr_sales[count][count]) * 100
    clustered_by_nbr_sales.append(by_nbr_sales)




#%% PLOT IT
def plot_share (data, x, y, color, colormap, title, text=None):
    fig = px.bar(data, x=x, y=y, color=color, color_discrete_map=colormap, text = text)

    # Edit the layout
    fig.update_layout(title=f"<b>{title}</b>",
                    xaxis_title='<b>Date</b>',
                    yaxis_title='<b>Market Share</b>',
                    #legend=dict(traceorder='reversed', font_size=11),
                    margin=dict(l=80, r=20, t=50, b=50),
                    autosize=True)

    fig.update_layout(
        title_x=0.3,
        font_family="Arial",
        font_size = 12,
        xaxis=dict(
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
        ),
        yaxis=dict(
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
        ),
        showlegend = True, 
        legend = dict(font = dict(size = 10))
    )
    # st.plotly_chart(fig,  use_container_width=True)
    return fig

#%%
print('Melting Time Series Data....')
clustered_universal_melts = []
for i in range(len(pois)):
    u_melted_df = clustered_universal_sales[i].melt(id_vars=['Month'], 
                                    value_vars=[i for i in range(len(clustered_neighborhood_products[i]))],
                                    var_name='Neighborhood', 
                                    value_name='Market_Sales')
    clustered_universal_melts.append(u_melted_df)
    colors = {
        0: 'blue',
        1: 'red'
    }
    # Create the bar chart
    fig = px.bar(u_melted_df, 
                x='Month', 
                y='Market_Sales', 
                color='Neighborhood', 
                title= f'Cluster {i} Neighborhood Share of Champagne',
                labels={'Market_Sales': 'Market Sales ($)'},
                barmode='stack', color_discrete_map=colors)

    fig2 = plot_share(u_melted_df, x='Month', y='Market_Sales', color='Neighborhood', colormap=colors, title='Neighborhood Share of Champagne')
# Show the plot
#%%
clustered_nbr_melts = []

for i in range(len(pois)):
    nbr_melts = []
    for nbr, data in clustered_by_nbr_sales[i].items():
        melted_df = data.melt(id_vars=['Month'], 
                                    value_vars=[y for y in clustered_neighborhood_products[i][nbr]],
                                    var_name='Product', 
                                    value_name='Market_Sales')
        nbr_melts.append(melted_df)
        # Create the bar chart
        fig = px.bar(melted_df, 
                    x='Month', 
                    y='Market_Sales', 
                    color='Product', 
                    title= f'Cluster {i} Neighborhood {nbr} Share of Champagne',
                    labels={'Market_Sales': 'Market Sales ($)'},
                    barmode='stack', color_discrete_sequence=px.colors.qualitative.G10)

    clustered_nbr_melts.append(nbr_melts)
# %%
print('Finishing Time Series')
for i in range(len(pois)):
    clustered_universal_melts[i].to_csv(f'assets/Cluster_{i}_Market_timeseries.csv')
    for nbr, data in enumerate(clustered_nbr_melts[i]):
        data.to_csv(f'assets/Cluster_{i}_Neighborhood_{nbr}_timeseries.csv')
# %%
