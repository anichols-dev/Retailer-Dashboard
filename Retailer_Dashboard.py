
#%%         SETUP FOR RETAILER DASHBOARD

#                 INPUTS
#    neighborhood_adjacency_dictionary.pkl:         Dictionary of Normalized Adjacency Calculations from Neighborhood_Adjacents.py
#    recommendation_1.csv,r...n_2, r...n_3:         CSV for each Neighborhood that holds neighborhood share information (share brand sales, share adjacent sales, etc) from Normalized_Adjacency.py
#    tops.pkl:                                      Top _______



import pandas as pd
from scipy.cluster.hierarchy import ward, fcluster, linkage, dendrogram
from scipy.spatial.distance import pdist, euclidean, cosine, correlation
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json
import streamlit as st
import plotly.express as px
import scipy.cluster.hierarchy as hc
import plotly.figure_factory as ff
import scipy.spatial as sp


st.set_page_config(page_title="[RETAILER] Dashboard", layout="wide")
st.title("Retailer Dashboard!")
#st.markdown("<h3 style="text-align: left; font-style: italic;">_Powered by Amygda_")
tab1, tab2 = st.tabs(["Category Management", "Anomaly Detection"])

#set up neighborhoods
global nhoods
global num_of_neighborhoods
global t_stats
global plotlies

aggfile = "assets/requirements/aggregations/Aggregated-Master.csv"


cmasta = pd.read_csv(aggfile)
    
#%% MAKE RECOMMENDATION TABLES
def create_h_and_d(txn_df):

    num_of_nhoods = 2
    neighborhood_list = []
    prods_of_focus = list(txn_df[txn_df['product_of_interest_flag']==1]['product_aggregation'].unique())

    focus_txn_dicts = []
    for poi in prods_of_focus:
        poi_dict = {}
        poi_dict['prod_agg'] = poi
        poi_mask = (txn_df['product_aggregation'] == poi) & (txn_df['product_of_interest_flag'] == 1)
        poi_dict['txn_lst'] = list(txn_df[poi_mask]['transaction_id'].unique())
        focus_txn_dicts.append(poi_dict)

    basket_corr_df = pd.DataFrame()
    for d in focus_txn_dicts:
        prod_agg = d['prod_agg']
        txn_lst = d['txn_lst']
        mask_1 = txn_df['transaction_id'].isin(txn_lst)
        temp_df = txn_df[mask_1]
        mask_2 = (temp_df['product_aggregation'] == prod_agg) & (temp_df['product_of_interest_flag']=='1')
        new_temp = temp_df[~mask_2]
        iso_adj = pd.DataFrame(new_temp[['subcategory', 'unit_sales']].groupby(['subcategory'])['unit_sales'].sum())
        iso_adj_reidx = iso_adj.reset_index()
        iso_adj_reidx['primary_prod_name'] = ''
        iso_adj_reidx['primary_prod_name'] = prod_agg
        new_iso = iso_adj_reidx[['primary_prod_name', 'subcategory', 'unit_sales']]
        new_iso.columns = ['primary_product', 'adjacent_subcategory', 'unit_sales']
        basket_corr_df = pd.concat([basket_corr_df, new_iso], axis=0)

    adj_matrix = basket_corr_df.pivot(index='primary_product', columns='adjacent_subcategory', values='unit_sales').fillna(0)
    for idx, row in adj_matrix.iterrows():
        adj_matrix.loc[idx] = row / row.sum()

    am_corr = adj_matrix.T.corr()
    am_dism = 1-am_corr

    link_ward = hc.linkage(sp.distance.squareform(am_dism), method='ward')

    champagnes = adj_matrix.index.tolist()
    cleaned_champagnes = [c.replace(' CHAMPAGNE', '').replace('CHAMPAGNE ', '').replace('CHAMPAGNE', '') for c in champagnes] #?

    #CREATE INITIAL CHAMPAGNE UNIVERSE DENDROGRAM IN PLOTLY
    universe_dn_initial = ff.create_dendrogram(
        X=am_dism,
        #labels=cleaned_champagnes, 
        orientation='bottom',
        linkagefun=lambda x: link_ward)

    # Customize layout
    universe_dn_initial.update_layout(
        title='Champagne Universe Dendrogram',
        font=dict(size=10),
        width=630, height=600,
        xaxis=dict(tickangle=-90)
    )

    # Extract the heights of the two main clusters
    heights = np.array([row[2] for row in link_ward])
    N1_height = heights[11]  # Last merge height for the orange cluster
    N2_height = heights[12]  # Last merge height for the green cluster

    # Print the heights
    #print(f"Height of Green Cluster: {N1_height}")
    #print(f"Height of Red Cluster: {N2_height}")

    # Create df showing components of each cluster

    # Initialize cluster composition dictionary
    cluster_composition = {}
    sub_cluster_composition = {}

    # Populate the dictionary with initial leaves
    n_leaves = link_ward.shape[0] + 1
    for i in range(n_leaves):
        cluster_composition[i] = [i]
        sub_cluster_composition[i] = [i]

    # Update the dictionary with merged clusters
    for i, row in enumerate(link_ward):
        cluster_id = n_leaves + i
        left = int(row[0])
        right = int(row[1])
        cluster_composition[cluster_id] = cluster_composition[left] + cluster_composition[right]
        sub_cluster_composition[cluster_id] = [left, right]

    # Create a DataFrame to display the cluster contents
    clusters = list(range(n_leaves, n_leaves + link_ward.shape[0]))
    leaves = [cluster_composition[c] for c in clusters]
    sub_clusters = [
        [f"Leaf {x}" if x < len(champagnes) else f"Cluster {x}" for x in sub_cluster_composition[c]]
        if sub_cluster_composition[c] != cluster_composition[c] else '-'
        for c in clusters
    ]
    df = pd.DataFrame({'Sub-Components': sub_clusters, 'Total Leaves': leaves}, index=clusters)
    df.index.name = 'Cluster'

    # Create a Styler object to adjust the font size
    styled_df = df.style.set_table_styles([{
        'selector': 'td',
        'props': [('font-size', '10pt')]
    }])


    #PREPROCESSING STUFF

    indices_alex = list(universe_dn_initial['layout']['xaxis']['ticktext'])
    ordered_indices = [int(x) for x in indices_alex]
    # Extract flat clusters
    clusters = fcluster(link_ward, 2, criterion='maxclust') # N=2

    colors = {}
    for count, data in enumerate(clusters):
        colors[count] = data

    color_list = []
    for num in indices_alex:
        color_list.append('N' + str(colors[int(num)]))

    # Mapping colors to leaf indices
    leaf_to_color = {int(leaf): color for leaf, color in zip(indices_alex, color_list)}

    # Color coding: 'N1' -> green, 'N2' -> red

    # Create a formatted output for link_ward with color coding
    formatted_link_ward = {}
    cluster_brain = styled_df.data.reset_index()
    for count, row in enumerate(link_ward):
        idx1 = int(row[0])
        #print(idx1)
        if count == len(link_ward)-1:
            color1= 'reset'
        else:
            color1 = leaf_to_color.get(idx1, 'reset')
            if color1 == 'reset':
                temp = cluster_brain[cluster_brain['Cluster'] == idx1]
                color1 = leaf_to_color.get(list(temp['Total Leaves'])[0][0],'reset')
        # Choose the color based on the first cluster
        if color1 not in formatted_link_ward:
            formatted_link_ward[color1] = []
        formatted_link_ward[color1].append(list(row))
        



    #GENERATE LINKAGE MATRIX FOR EACH NEIGHBORHOOD

    n1_linkage_matrix = np.array(formatted_link_ward['N1'])

    # Extract all unique indices from the first two columns
    unique_indices_n1 = np.unique(n1_linkage_matrix[:, :2].astype(int).flatten())

    # Create a mapping from old indices to new consecutive indices
    old_to_new_n1 = {old: new for new, old in enumerate(unique_indices_n1)}

    # Apply the mapping to the linkage matrix
    n1_linkage_matrix_transformed = n1_linkage_matrix.copy()
    n1_linkage_matrix_transformed[:, 0] = [old_to_new_n1[int(i)] for i in n1_linkage_matrix_transformed[:, 0]]
    n1_linkage_matrix_transformed[:, 1] = [old_to_new_n1[int(i)] for i in n1_linkage_matrix_transformed[:, 1]]

    n2_linkage_matrix = np.array(formatted_link_ward['N2'])

    # Extract all unique indices from the first two columns
    unique_indices_n2 = np.unique(n2_linkage_matrix[:, :2].astype(int).flatten())

    # Create a mapping from old indices to new consecutive indices
    old_to_new_n2 = {old: new for new, old in enumerate(unique_indices_n2)}

    # Apply the mapping to the linkage matrix
    n2_linkage_matrix_transformed = n2_linkage_matrix.copy()
    n2_linkage_matrix_transformed[:, 0] = [old_to_new_n2[int(i)] for i in n2_linkage_matrix_transformed[:, 0]]
    n2_linkage_matrix_transformed[:, 1] = [old_to_new_n2[int(i)] for i in n2_linkage_matrix_transformed[:, 1]]


    #CREATE UNIVERSE DN

    universe_dn = ff.create_dendrogram(
        X=am_dism,
        #labels=cleaned_champagnes, 
        orientation='left',
        linkagefun=lambda x: link_ward)

    # Customize layout
    universe_dn.update_layout(
        title='Champagne Universe Dendrogram w/ Labels',
        font=dict(size=10),
        width=630, height=600,
        yaxis=dict(
            tickangle=0,
            ticktext=[f"<b>{cleaned_champagnes[i]}</b>" for i in ordered_indices],
        ),

    )

    #universe_dn.update_xaxes(tickcolor='white')
    #universe_dn.update_yaxes(tickcolor='white')

    #universe_dn.show()



    #CREATE N1 DENDROGRAM IN PLOTLY

    n1_dn_plotly = ff.create_dendrogram(
        X=am_dism,
        orientation='left',
        linkagefun=lambda x: n1_linkage_matrix_transformed,
        color_threshold=np.inf
    )

    # Change the color of all dendrogram traces to green
    for trace in n1_dn_plotly['data']:
        trace.update(marker=dict(color='green'))

    # Define the light orange color
    light_green = 'rgba(240, 255, 240, 1.0)'

    # Update layout to set the background color to light green
    n1_dn_plotly.update_layout(
        title='N1 Dendrogram',
        font=dict(size=10),
        width=630,
        height=600,
        yaxis=dict(
            tickangle=0,
            ticktext=[f"<b>{cleaned_champagnes[i]}</b>" for i in ordered_indices[0:6]],
        ),
        plot_bgcolor=light_green
    )

    # Show the dendrogram
    #n1_dn_plotly.show()



    #CREATE N2 DENDROGRAM IN PLOTLY

    n2_dn_plotly = ff.create_dendrogram(
        X=am_dism,
        orientation='left',
        linkagefun=lambda x: n2_linkage_matrix_transformed,
        color_threshold=np.inf
    )

    # Change the color of all dendrogram traces to red
    for trace in n2_dn_plotly['data']:
        trace.update(marker=dict(color='red'))

    # Define a light red color
    light_red = 'rgba(255, 224, 224, 1.0)'

    # Update layout to set the background color to light red
    n2_dn_plotly.update_layout(
        title='N2 Dendrogram',
        font=dict(size=10),
        width=600,
        height=600,
        yaxis=dict(
            tickangle=0,
            ticktext=[f"<b>{cleaned_champagnes[i]}</b>" for i in ordered_indices[6:16]]
        ),
        plot_bgcolor=light_red
    )

    # Show the dendrogram
    #n2_dn_plotly.show()


    #PREPROCESSING BEFORE CREATING HEATMAP W/ COLORBARS

    X = am_corr.values
    ordered_indices = [int(x) for x in indices_alex]
    # print("Ordered Indices:", ordered_indices)

    # Ensure ordered_indices are within valid range
    # Filter out indices that exceed array size
    ordered_indices = [idx for idx in ordered_indices if idx < X.shape[1]]

    # Reorder X correctly
    X_ordered = X[ordered_indices, :][:, ordered_indices]

    # Define color mappings
    color_map_plotly = {'N1': 'green', 'N2': 'red'}

    # Extract leaves_color_list and map colors
    leaves_color_list_plotly = color_list
    colors_plotly = [color_map_plotly[color] for color in leaves_color_list_plotly]

    # Normalize the values to fit the colorscale
    normalized_values = np.linspace(0, 1, len(colors_plotly))

    # Construct the colorscale using the normalized values and colors
    colorscale_test = []
    for i, color in enumerate(colors_plotly):
        colorscale_test.append([normalized_values[i], color])

    #HEATMAP W/ COLORBARS

    # Create a subplot figure with a grid layout
    fig = make_subplots(
        rows=2, cols=2,  # Define the number of rows and columns
        shared_xaxes='columns',  # Share the x-axis between subplots
        shared_yaxes='rows',  # Share the y-axis between subplots
        column_widths=[0.95, 0.05],
        row_heights=[0.05, 0.95],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    # Values that will come in handy...

    # Generate values from 1 to 15 in a column format
    # This will be helpful for Plot 4
    z_values = np.arange(1, (len(cleaned_champagnes)+1)).reshape(-1, 1)

    # Generate values in a row format
    # This will be helpful for Plot 1
    z_values2 = np.array([[1] * (colors_plotly.count('green')) + [2] * (colors_plotly.count('red'))])

    # Plot 1 (row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=z_values2,
        colorscale=colorscale_test,
        showscale=False  # Remove the color scale legend
    ), row=1, col=1)

    # Plot 2 (row=1, col=2)
    # BLANK

    # Plot 3 (row=2, col=1)
    heatmap_trace = go.Heatmap(
        z=X_ordered,
        colorscale='Blues',  # Choose a suitable colorscale
        zmax=np.max(X_ordered),  # Set max value for color scale
        zmin=np.min(X_ordered),  # Set min value for color scale
        text=np.round(X_ordered, 3),  # Display values rounded to 3 decimal places
        texttemplate="%{text}",  # Show text values as is
        textfont={"size": 6},  # Adjust font size for text values
        showscale=False  # Remove color bar
    )
    fig.add_trace(heatmap_trace, row=2, col=1)

    # Plot 4 (row=2, col=2)
    fig.add_trace(go.Heatmap(
        z=z_values,
        colorscale=colorscale_test,
        showscale=False  # Remove the color scale legend
    ), row=2, col=2)

    # Update layout for Plot 1 (row=1, col=1)
    fig.update_xaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=1, col=1)

    # Update layout for Plot 3 (row=2, col=1)
    fig.update_xaxes(
        showticklabels=True, 
        tickvals=np.arange(0, X_ordered.shape[1]),
        ticktext=[cleaned_champagnes[i] for i in ordered_indices],
        tickangle=-90,
        showgrid=False, 
        zeroline=False,
        tickfont=dict(size=7),
        row=2, col=1)
    fig.update_yaxes(
        showticklabels=True, 
        tickvals=np.arange(0, X_ordered.shape[1]),
        ticktext=[cleaned_champagnes[i] for i in ordered_indices],
        tickangle=0,
        showgrid=False, 
        zeroline=False,
        tickfont=dict(size=7),
        row=2, col=1)

    # Update layout for Plot 4 (row=2, col=2)
    fig.update_xaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=2, col=2)

    # Update layout
    fig.update_layout(height=600, width=600, title= { 
        'text': "Champagne Heatmap", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top' })

    # Show plot
    #fig.show()


    # SEPARATE UNIVERSE INTO EACH NEIGHBORHOOD

    # Extract the 6x6 square (Canard Duchene to Ruinart)
    n1 = X_ordered[:colors_plotly.count('green'), :colors_plotly.count('green')]

    # Extract the 9x9 square (Louis Roederer to Piper Heidsieck)
    n2 = X_ordered[colors_plotly.count('green'):, colors_plotly.count('green'):]

    # Create n1 heatmap
    fig_n1 = make_subplots(
        rows=2, cols=2,
        shared_xaxes='columns',
        shared_yaxes='rows',
        column_widths=[0.95, 0.05],
        row_heights=[0.05, 0.95],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    # Green row/column colorbars
    z_values_n1_row = np.array([[1] * colors_plotly.count('green')])
    z_values_n1_col = np.arange(1, (colors_plotly.count('green')+1)).reshape(-1, 1)
    colorscale_green = [[0, 'green'], [1, 'green']]

    # Add green colorbars around n1 heatmap
    fig_n1.add_trace(go.Heatmap(z=z_values_n1_row, colorscale=colorscale_green, showscale=False), row=1, col=1)
    fig_n1.add_trace(go.Heatmap(z=z_values_n1_col, colorscale=colorscale_green, showscale=False), row=2, col=2)

    # Add main n1 heatmap
    fig_n1.add_trace(go.Heatmap(
        z=n1,
        colorscale='Blues',
        zmax=np.max(X_ordered),
        zmin=np.min(X_ordered),
        text=np.round(n1, 3),
        texttemplate="%{text}",
        textfont={"size": 6},
        showscale=False
    ), row=2, col=1)

    # Update layout for n1 heatmap
    fig_n1.update_xaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=1, col=1)
    fig_n1.update_yaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=1, col=1)
    fig_n1.update_xaxes(
        tickvals=np.arange(0, colors_plotly.count('green')),
        ticktext=[cleaned_champagnes[i] for i in ordered_indices[:colors_plotly.count('green')]],
        tickangle=-90,
        tickfont=dict(size=7),
        row=2, col=1)
    fig_n1.update_yaxes(
        tickvals=np.arange(0, colors_plotly.count('green')),
        ticktext=[cleaned_champagnes[i] for i in ordered_indices[:colors_plotly.count('green')]],
        tickfont=dict(size=7),
        row=2, col=1)
    fig_n1.update_xaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=2, col=2)
    fig_n1.update_yaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=2, col=2)
    fig_n1.update_layout(height=600, width=600, title_text="Neighborhood 1 Heatmap")

    # Create n2 heatmap
    fig_n2 = make_subplots(
        rows=2, cols=2,
        shared_xaxes='columns',
        shared_yaxes='rows',
        column_widths=[0.95, 0.05],
        row_heights=[0.05, 0.95],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    # Red row/column colorbars
    z_values_n2_row = np.array([[1] * colors_plotly.count('red')])
    z_values_n2_col = np.arange(1, (colors_plotly.count('red')+1)).reshape(-1, 1)
    colorscale_red = [[0, 'red'], [1, 'red']]
    neighborhood_list.append(fig_n1)

    # Add red colorbars around n2 heatmap
    fig_n2.add_trace(go.Heatmap(z=z_values_n2_row, colorscale=colorscale_red, showscale=False), row=1, col=1)
    fig_n2.add_trace(go.Heatmap(z=z_values_n2_col, colorscale=colorscale_red, showscale=False), row=2, col=2)

    # Add main n2 heatmap
    fig_n2.add_trace(go.Heatmap(
        z=n2,
        colorscale='Blues',
        zmax=np.max(X_ordered),
        zmin=np.min(X_ordered),
        text=np.round(n2, 3),
        texttemplate="%{text}",
        textfont={"size": 6},
        showscale=False
    ), row=2, col=1)

    # Update layout for n2 heatmap
    fig_n2.update_xaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=1, col=1)
    fig_n2.update_yaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=1, col=1)
    fig_n2.update_xaxes(
        tickvals=np.arange(0, colors_plotly.count('red')),
        ticktext=[cleaned_champagnes[i] for i in ordered_indices[colors_plotly.count('green'):]],
        tickangle=-90,
        tickfont=dict(size=7),
        row=2, col=1)
    fig_n2.update_yaxes(
        tickvals=np.arange(0, colors_plotly.count('red')),
        ticktext=[cleaned_champagnes[i] for i in ordered_indices[colors_plotly.count('green'):]],
        tickfont=dict(size=7),
        row=2, col=1)
    fig_n2.update_xaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=2, col=2)
    fig_n2.update_yaxes(showticklabels=False, tickcolor='white', showgrid=False, zeroline=False, row=2, col=2)
    fig_n2.update_layout(height=600, width=600, title_text="Neighborhood 2 Heatmap")
    neighborhood_list.append(fig_n2)

    # Show plots
    #fig_n1.show()
    #fig_n2.show()
    n_dn_plotly = []
    n_dn_plotly.append(n1_dn_plotly)
    n_dn_plotly.append(n2_dn_plotly)

    return fig, neighborhood_list, n_dn_plotly, universe_dn

#%%

def recommendation_table(table_stats):

    def get_colors(value, min_val, max_val):
        # Normalize the value
        norm_value = (value - min_val) / (max_val - min_val)
        # Define the color ranges for green to yellow to orange to red
        if norm_value < 0.33:
            # Interpolate between green (0, 255, 0) and yellow (255, 255, 0)
            r = int(765 * norm_value)
            g = 255
            b = 0
        elif norm_value < 0.66:
            # Interpolate between yellow (255, 255, 0) and orange (255, 165, 0)
            r = 255
            g = int(255 - 270 * (norm_value - 0.33))
            b = 0
        else:
            # Interpolate between orange (255, 165, 0) and red (255, 0, 0)
            r = 255
            g = int(165 - 165 * (norm_value - 0.66) / 0.34)
            b = 0
        return f'rgba({r}, {g}, {b}, .35)'

    def get_rec_colors(rec):
        if rec == 'Maintain':
            return f'rgba(0, 255, 0, .5)'
        elif rec == 'Reduce':
            return f'rgba(255, 255, 0, 0.5)'
        else:
            return f'rgba(255, 0, 0, .5)'
        

    def unisort(list1, list2):
        sorted_indices = sorted(range(len(list1)), key=lambda i: list1[i], reverse=True)
        sorted1 = [list1[i] for i in sorted_indices]
        # Step 2: Reorder list2 based on the new order of indexes from temp1
        sorted2 = [list2[i] for i in sorted_indices]
        return sorted2

    def colored2(colors, unsorted):
        proper_colors = []
        thesort = sorted(list(unsorted), reverse=True)
        dict = {}
        for i, val in enumerate(thesort):
            dict[val] = colors[i]
        for val in unsorted:
            for i in dict:
                if i == val:
                    proper_colors.append(dict[i])
        return proper_colors

    import numpy as np

    #  NORMALIZE BASKET DRIVE COEFFICIENTS
    def fit_log_function(X1, X2):
        # Given Y values
        Y2 = 1
        Y1 = 0.5 # PLAY WITH THIS VALUE TO OPTIMIZE BASKET DRIVE SC0RE
        log_X1 = np.log(X1)
        log_X2 = np.log(X2)

        # Set up the system of linear equations:
        # Y1 = A * log_X1 + C
        # Y2 = A * log_X2 + C
        A_matrix = np.array([[log_X1, 1], [log_X2, 1]])
        B_matrix = np.array([Y1, Y2])

        # Solve for A and C
        A, C = np.linalg.solve(A_matrix, B_matrix)

        return A, C

    def find_basket_drive_coefficients(adjacent_share, brand_share, min, max):
        A, C = fit_log_function(min, max)
        coefficient = (A*np.log(brand_share))+C
        print(f"brand share: {brand_share}")
        print(f"adjacent share: {adjacent_share}")
        print(f"A: {A},  C:  {C} \n")
        return adjacent_share*coefficient
    

    def apply_recommendation(nhood):
        lower2 = True
        i=0
        for index, row in nhood.iterrows():
            if nhood['SHARE_BRAND_SALES_NBRH'].iloc[i] == max(nhood['SHARE_BRAND_SALES_NBRH'].values):
                nhood.loc[index, 'recommendation'] = 'Maintain'
            elif nhood['CUMULATIVE_SHARE_BRAND_SALES'].iloc[i-1] <= 80:
                nhood.loc[index, 'recommendation'] = 'Maintain'
            else: #lower 20% of pois (by brand share)
                if lower2:
                    lower20mean = nhood['BASKET_DRIVE'].iloc[i:].mean()
                    lower2 = False
                if nhood['BASKET_DRIVE'].iloc[i] >= lower20mean:
                    nhood.loc[index, 'recommendation'] = 'Reduce'
                else:
                    nhood.loc[index, 'recommendation'] = 'Eliminate'
            i+=1
        return nhood


    def make_table(table_stats):
        fig_list = []
        for data in table_stats:
            fig1 = go.Figure()
           
            
            #normalize basket_drive
            min_val = min(data['SHARE_BRAND_SALES_NBRH'])
            max_val = max(data['SHARE_BRAND_SALES_NBRH'])
            temps = data.apply(lambda row: find_basket_drive_coefficients(row['SHARE_ADJACENT_SALES_NBRH'], row['SHARE_BRAND_SALES_NBRH'], min_val, max_val), axis =1)
            data['BASKET_DRIVE'] = (temps/data['SHARE_BRAND_SALES_NBRH'])-1

            #reset recommendation
            data = apply_recommendation(data)

            # Create a list of colors for the 'Recommendation' column based on the condition
            rec_colors = [f'rgba(0, 255, 0, .7)' if val == 'Maintain' else f'rgba(255, 165, 0, 0.7)' if val =='Reduce' else f'rgba(255, 0, 0, .7)' for val in data['recommendation']]
            brand_colors = [get_colors(i, 0, len(data['SHARE_BRAND_SALES_NBRH'])-1) for i, val in enumerate(data['SHARE_BRAND_SALES_NBRH'])]
            adjacent_colors = [get_colors(i, 0, len(data['SHARE_ADJACENT_SALES_NBRH'])-1) for i, val in enumerate(data['SHARE_ADJACENT_SALES_NBRH'])]
            basket_colors = [get_colors(i, 0, len(data['BASKET_DRIVE'])-1) for i, val in enumerate(data['BASKET_DRIVE'])]

            #sort and unsort
            temp1 = data['SHARE_ADJACENT_SALES_NBRH']
            temp2 = data['BASKET_DRIVE']
            sorted_adj_colors = colored2(adjacent_colors, temp1)
            data['recommendation'] = data['recommendation'].str.upper()
            rec_text_colors = ['darkgreen' if recs == 'MAINTAIN' else 'brown' if recs == 'REDUCE' else 'darkred' for recs in data['recommendation']]
            sorted_basket_colors = colored2(basket_colors, temp2)
            

            #figures
            fig1.add_trace(go.Table(
                header=dict(
                    values=['<b>PRODUCT</b>', '<b>SHARE BRAND SALES</b>', '<b>SHARE ADJACENT SALES</b>', '<b>BASKET DRIVE SCORE</b>', '<b>RECOMMENDATION</b>'],
                    fill_color='paleturquoise',
                    align='center',
                    line=dict(color=['white', 'white', 'white', 'white', 'black'],
                              width=[2, 2, 2, 2, 2]),
                    font=dict(family="Arial", size=12, color="black")
                ),
                cells=dict(
                    values=[
                        [f'<b>{val}</b>' for val in data['Unnamed: 0']],
                        [f'{val:.2f}%' for val in data['SHARE_BRAND_SALES_NBRH']],
                        [f'{val:.2f}%' for val in data['SHARE_ADJACENT_SALES_NBRH']],
                        [f'{val:.2f}%' for val in data['BASKET_DRIVE']],
                        [f'<b>{rec}</b>' for rec in data['recommendation']]
                    ],
                    fill=dict(color=[
                        ['white'] * len(data),
                        brand_colors, sorted_adj_colors, sorted_basket_colors, rec_colors, rec_colors
                    ]),
                    align=[
                        'center', 'center', 'center', 'center', 'center'
                    ],
                    line=dict(color=['white', 'white', 'white', 'white', 'black'],
                              width=[2, 2, 2, 2, 2]),
                    font=dict(family="Arial", size=12, color=[
                        ['black'] * len(data),
                        ['black'] * len(data),
                        ['black'] * len(data),
                        ['black'] * len(data),
                        rec_text_colors
                    ])
                ),
                domain=dict(x=[0, 1], y=[0, 1])
            ))

            # Update layout
            fig1.update_layout(title={
                                    'text': " KEY:  <span style='color:green'>Green = Category Driver</span>,   <span style='color:orange'>Orange = Basket Driver</span>,   <span style='color:red'>Red = Under-Performer</span>",
                                    'x': 0,  # Center the title
                                    'xanchor': 'left'
                                },
                            autosize = True, margin=dict(l=0, r=0, t= 30, b=0), height = 350)
            fig1.update_layout(
                uniformtext_minsize=8,  # Minimum font size
                uniformtext_mode='hide',  # Hide text that doesn't fit
            )
            fig_list.append(fig1)
            # Show the table
        return fig_list
    
    return make_table(table_stats)
    
#%% Output Neighborhood Adjacent Tables

def neighborhood_adjacents(n_plots, selected_nbrhd):
    columns = 0
    annotations2 = []
    annotations1 = []
    for cat in n_plots[selected_nbrhd]['fig']:
        for trace in n_plots[selected_nbrhd]['fig'][cat].data:
            if trace.cells.values[0]: 
                columns +=1
    print(f'Cols: {columns}')
    if columns == 4:
        cols = st.columns(2)
        #output the tables
        rightFig = make_subplots(
                rows=3, cols=1,
                vertical_spacing=0.1, 
                specs=[[{"type": "table"}], [{"type": "table"}], [{"type": "table"}]]
            )
        leftFig = make_subplots(
                rows=1, cols=1,
                vertical_spacing=0.1, 
                specs=[[{"type": "table"}]]
            )
        row = 1
        for count, y in enumerate(n_plots[selected_nbrhd]['fig']):
            for trace in n_plots[selected_nbrhd]['fig'][y].data:
                if trace.cells.values[0]:
                    if y == 'all':
                        leftFig.add_trace(trace, row=1, col=1)
                        leftFig.update_layout(height=800, showlegend=False)
                        annotations1.append(dict(
                                            x=0.5,
                                            y=1,
                                            xref='paper',
                                            yref='paper',
                                            xanchor='center',
                                            yanchor='bottom',
                                            text=f'<b>{y.upper()}</b>',
                                            showarrow=False,
                                            font=dict(size=20)
                                        ))
                        leftFig.update_layout(annotations=annotations1)
                        cols[0].plotly_chart(leftFig, use_container_width=True)
                    else:
                        print(f'This is trace for row {row}, {trace}')
                        rightFig.add_trace(trace, row=row, col=1)
                        rightFig.update_xaxes(title_text=f'Title {y}', row=row, col=1)
                        annotations2.append(dict(
                                            x=0.5,
                                            y=1.0 - (row - 1)*.365,
                                            xref='paper',
                                            yref='paper',
                                            xanchor='center',
                                            yanchor='bottom',
                                            text=f'<b>{y.upper()}</b>',
                                            showarrow=False,
                                            font=dict(size=20)
                                        ))
                        row+=1
    elif columns > 2:
        cols = st.columns(2)
        col = 0
        #output the tables
        for i in n_plots[selected_nbrhd]['fig']:
            for trace in n_plots[selected_nbrhd]['fig'][i].data:
                if trace.cells.values[0]:
                    cols[col].plotly_chart(n_plots[selected_nbrhd]['fig'][i], use_container_width=True)
                    col +=1
    elif columns > 0: 
        cols = st.columns(columns-1)
        col = 0
        #output the tables
        for i in n_plots[selected_nbrhd]['fig']:
            for trace in n_plots[selected_nbrhd]['fig'][i].data:
                if trace.cells.values[0] and i != 'all':
                    cols[col].plotly_chart(n_plots[selected_nbrhd]['fig'][i], use_container_width=True)
                    col +=1

    rightFig.update_layout(height=700, width=800, annotations=annotations2)
    cols[1].plotly_chart(rightFig, use_container_width=True)


def basket_stats(basket, n_stats, year='all', quarter='all'):
    output = {}
    basket_data = {}
    if year == 'all':
        basket_data = basket['all']
    elif quarter == 'all':
        basket_data = basket['year'][year]
    else:
        basket_data = basket['quarter'][year][quarter]
    for nbr in n_stats:
        output[nbr] = {
                'basket_value': 0,
                'items_per_basket': 0,
                'adjacent_value': 0,
                'adjacent_items_per_basket': 0,
                'total_txns': 0        
                }
        for poi in n_stats[nbr]['top_80']:      
            if basket_data[poi]['basket_value'] > 1:
                output[nbr]['basket_value'] += basket_data[poi]['total_basket_val']
                output[nbr]['items_per_basket'] += basket_data[poi]['total_items']
                output[nbr]['adjacent_value'] += basket_data[poi]['total_adjacent_basket_val']
                output[nbr]['adjacent_items_per_basket'] += basket_data[poi]['total_adjacent_items']
                output[nbr]['total_txns'] += basket_data[poi]['total_txns']

        #Average every total
        output[nbr]['basket_value'] = output[nbr]['basket_value']/output[nbr]['total_txns']
        output[nbr]['items_per_basket'] = output[nbr]['items_per_basket']/output[nbr]['total_txns']
        output[nbr]['adjacent_value'] = output[nbr]['adjacent_value']/output[nbr]['total_txns']
        output[nbr]['adjacent_items_per_basket'] = output[nbr]['adjacent_items_per_basket']/output[nbr]['total_txns']
    
    return output

def find_delta(current, basket, n_stats, year='all', quarter='all', delt=True):
    #select proper data
    if year =='all' or delt == False:
        delt = False
    elif quarter == 'all':
        previous = basket_stats(basket, n_stats, year-1)
    else:
        if quarter == 1:
            previous = basket_stats(basket, n_stats, year-1, 4)
        else:
            previous = basket_stats(basket, n_stats, year-1, quarter)

    #compute calculations
    delta = {}
    for nbr in current:
        delta[nbr] = {}
        for var in current[nbr]:
            if delt:
                delta[nbr][var] = round((current[nbr][var]-previous[nbr][var])/previous[nbr][var] * 100, 2)
            else:
                delta[nbr][var] = None
    return delta

def plot_ridge(data, select_products, nbr):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    filtered_data = data[data['product_aggregation'].isin(select_products)]
    filtered_data['product_aggregation'] = filtered_data['product_aggregation'].str.replace('CHAMPAGNE', '', regex=False).str.strip()

    # Create the ridge plot
    sns.violinplot(x="hour", y="product_aggregation", data=filtered_data, bw=.15, scale='width', split=True, native_scale=True)
    plt.title(f'Density of Purchase Times for Neighborhood {nbr}')
    plt.xlabel('Time of Day')
    plt.ylabel('')

    # Format the x-axis to show hours
    ax = plt.gca()
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x // 60):02}:{int(x % 60):02}'))
    # Set x-axis limits to 0-1440 (0-24 hours)
    ax.set_xlim(9, 22)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
    st.pyplot(plt)

def plot_nbr_share(melt, nbr = 'All' , cluster_sel=None, max = None):
    # Create the bar chart
    if cluster_sel == max-1 and nbr == 'All':
        melt['Neighborhood'] = melt['Neighborhood'].astype(str)
        fig2 = px.bar(melt, 
                x='Month', 
                y='Market_Sales', 
                color='Neighborhood', 
                title= f'Market Share of Neighborhoods for All Retail Clusters',
                labels={'Market_Sales': 'Market Sales ($)'},
                barmode='stack', color_discrete_sequence=px.colors.qualitative.Set1)
    else:
        if nbr == 'All':
            fig2 = px.bar(melt, 
                    x='Month', 
                    y='Market_Sales', 
                    color='Neighborhood', 
                    title= f'Neighborhood Share for Retail Cluster {cluster_sel+1}',
                    labels={'Market_Sales': 'Market Sales ($)'},
                    barmode='stack', color_discrete_sequence=px.colors.qualitative.G10)
        else:
            if cluster_sel == max-1:
                fig2 = px.bar(melt, 
                    x='Month', 
                    y='Market_Sales', 
                    color='Product', 
                    title= f'Product Share of Neighborhood {nbr+1} for All Retail Clusters',
                    labels={'Market_Sales': 'Market Sales ($)'},
                    barmode='stack', color_discrete_sequence=px.colors.qualitative.G10)
            else:
                fig2 = px.bar(melt, 
                        x='Month', 
                        y='Market_Sales', 
                        color='Product', 
                        title= f'Product Share of Neighborhood {nbr+1} for Retail Cluster {cluster_sel+1}',
                        labels={'Market_Sales': 'Market Sales ($)'},
                        barmode='stack', color_discrete_sequence=px.colors.qualitative.G10)
    return fig2

def harris_projections(df_in, nbr):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from prophet import Prophet
    from statsmodels.tsa.seasonal import seasonal_decompose
    import streamlit as st
    df_in['transaction_datetime'] = pd.to_datetime(df_in['transaction_datetime'])
    df1 = df_in[df_in['product_of_interest_flag'] == 1]
    forecast_steps = 92

    # Define the product groups
    n1 = [
        'CHAMPAGNE CANARD DUCHENE CHAMPAGNE',
        'BILLECART-SALMON CHAMPAGNE',
        'DOM PERIGNON CHAMPAGNE',
        'POMMERY CHAMPAGNE',
        'LAURENT-PERRIER CHAMPAGNE',
        'RUINART CHAMPAGNE'
    ]
    n2 = [
        'LOUIS ROEDERER CHAMPAGNE',
        'TATTINGER CHAMPAGNE',
        'LANSON CHAMPAGNE',
        'MOET & CHANDON CHAMPAGNE',
        'VEUVE CLICQUOT CHAMPAGNE',
        'G. H. Mumm CHAMPAGNE',
        'PERRIER JOUET CHAMPAGNE',
        'NICOLAS CHAMPAGNE',
        'PIPER HEIDSIECK CHAMPAGNE'
    ]

    # Filter data for each product group
    df_n1 = df1[df1['prod_agg'].isin(n1)]
    df_n2 = df1[df1['prod_agg'].isin(n2)]

    # Aggregate sales data
    sales_data_n1 = df_n1.groupby(df_n1['transaction_datetime'].dt.date)['dollar_sales'].sum()
    sales_data_n2 = df_n2.groupby(df_n2['transaction_datetime'].dt.date)['dollar_sales'].sum()

    # Convert the index to datetime
    sales_data_n1.index = pd.to_datetime(sales_data_n1.index)
    sales_data_n2.index = pd.to_datetime(sales_data_n2.index)

    # Function to perform seasonal decomposition and forecasting
    def forecast_sales(sales_data, title):
        decomposition = seasonal_decompose(sales_data, model='additive', period=365)
        trend = decomposition.trend.dropna()  # Remove NaN values

        df_prophet = pd.DataFrame({
            'ds': trend.index,
            'y': trend.values
        })

        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_steps, freq='D')
        forecast = model.predict(future)

        plt.figure(figsize=(14, 6))
        plt.plot(df_prophet['ds'], df_prophet['y'], label='Original Trend')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Trend', linestyle='--')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='k', alpha=0.1)
        plt.legend(loc='upper left')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.grid(True)
        
        # Use st.pyplot() to display the plot in Streamlit
        st.pyplot(plt)

    # Generate plots for N1 and N2
    if nbr == 1:
        forecast_sales(sales_data_n1, 'N1 Trend Component with Prophet Forecast')
    elif nbr == 2:
        forecast_sales(sales_data_n2, 'N2 Trend Component with Prophet Forecast')
    else:
        st.write("Invalid 'nbr' value. Please use 1 for N1 or 2 for N2.")

#%% #############################  LOAD IN DATA  ###############################################


import pickle
########## cluster #######


# The Big Three
with open('assets/clustered_adjacent_plots.pkl', 'rb') as file:
    clustered_n_plots = pickle.load(file)

with open('assets/clustered_basket_stats.pkl', 'rb') as file:
    clustered_basket_stats = pickle.load(file)

with open('assets/clustered_neighborhood_adjacent_stats.pkl', 'rb') as file:
    clustered_neighborhood_adj_stats = pickle.load(file)

# Purchase Times
with open('assets/clustered_purchase_times.pkl', 'rb') as file:
    clustered_time_df = pickle.load(file)

cluster = len(clustered_basket_stats)


num_of_neighborhoods = 2
nhoods = []
for n in range(num_of_neighborhoods):
    nhoods.append('Neighborhood ' + str(n+1))
# Load the Neighborhood breakdown data to select top 80
clustered_recommendations = []

for clstr in range(cluster):
    t_stats = [] #list of each recommendation table, index is neighborhood
    if clstr == cluster-1:
        for nbr in range(num_of_neighborhoods): 
            df = pd.read_csv(f'assets/recommendation_{nbr}.csv')
            t_stats.append(df)
    else:
        for nbr in range(num_of_neighborhoods): 
            df = pd.read_csv(f'assets/Cluster_{clstr}_recommendation_{nbr}.csv')
            t_stats.append(df)
    clustered_recommendations.append(t_stats)

clustered_neighborhood_products = []
for i in range(cluster):
    neighborhood_products = []
    for y in clustered_recommendations[i]:
        neighborhood_products.append(list(y['Unnamed: 0']))
    clustered_neighborhood_products.append(neighborhood_products)

# Recommentation Tables
clustered_plotlies = []
for i in range(cluster):
    clustered_plotlies.append(recommendation_table(clustered_recommendations[i]))

# Time Series
clustered_timeseries_plots = []
for i in range(cluster):
    timeseries = {} 
    for nbr in range(num_of_neighborhoods):
        filename = pd.read_csv(f'assets/Cluster_{i}_Neighborhood_{nbr}_timeseries.csv')
        timeseries[nbr] = plot_nbr_share(filename, nbr = nbr, cluster_sel=i, max = cluster)
    timeseries['market'] = plot_nbr_share(pd.read_csv(f'assets/Cluster_{i}_Market_timeseries.csv'), cluster_sel = i, max = cluster)
    clustered_timeseries_plots.append(timeseries)




#############
# The Big Three
# with open('neighborhood_top_adjacents_dictionary.pkl', 'rb') as file:
#     n_plots = pickle.load(file)

# with open('poi_basket_stats.pkl', 'rb') as file:
#     poi_basket_stats = pickle.load(file)

# with open('neighborhood_stats.pkl', 'rb') as file:
#     neighborhood_stats = pickle.load(file)

# # Purchase Times
# time_df = pd.read_csv('time_df.csv')

# Heatmaps
heatmap, hm_nbr_list, n_dendro, universe_dendro = create_h_and_d(cmasta)

#%% ###########################  STREAMLIT OUTPUT ###########################################


#Category management tab
with tab1:
    
    # SELECT BOXES
    #st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line divider
    colS1, colS2 = st.columns(2)
    with colS1:
        nbr_option = st.selectbox(
            'Select a Neighborhood:',
            nhoods)
        neighborhood = int(nbr_option.split(" ")[1])-1
    with colS2:
        cluster_list = ['All Retailers', 'Retail Cluster 1', 'Retail Cluster 2', 'Retail Cluster 3', 'Retail Cluster 4']
        c_option = st.selectbox(
            'Isolate a Retail Cluster:',
            cluster_list)
        if c_option == 'All Retailers':
            cluster_select = cluster-1
        else:
            cluster_select = int(c_option.split(" ")[2])-1
    st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line divider

    # PRODUCT NEIGHBORHOODS
    st.subheader(f"Consumers purchase champagne in {num_of_neighborhoods} category **Neighborhoods**.")
  # Add a horizontal line divider

    # PROJECTIONS
    if 'projections' not in st.session_state:
        st.session_state.projections = False
    def toggle_projection():
        st.session_state.projections = not st.session_state.projections
    button_label = "Hide Projection" if st.session_state.projections else "Project Sales"
    if st.button(button_label, on_click=toggle_projection):
        pass
    if st.session_state.projections:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(hm_nbr_list[neighborhood])
        with col2:
            #st.image(heatmap_screenshot)
            st.plotly_chart(heatmap)

        colD1, colD2= st.columns(2)
        with colD1:
            st.plotly_chart(n_dendro[neighborhood])
        with colD2:
            st.plotly_chart(universe_dendro)
        st.markdown("<hr>", unsafe_allow_html=True)
        df5 = pd.read_csv(aggfile)
        harris_projections(df5, neighborhood+1)

    # RETAIL CLUSTERS
    # colR1, colR2 = st.columns(2)
    # with colR1:
    #     st.subheader(f"And your locations fall into {cluster-1} Retail Clusters.")
    #     st.write('')
    #     st.write('')
    #     st.write('')
    #     st.markdown(":yellow[**Cluster 1:** BIG.]")
    #     st.markdown(":blue[**Cluster 2:** Small.]")
    #     st.markdown(":purple[**Cluster 3:** Just right!.]")
    # with colR2:
    #     st.image(heatmap_screenshot)
    #     #st.plotly_chart(heatmap)

    st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line divider

    # TIME SERIES MARKET SHARE PLOTS
    st.subheader("Market Share")
    st.plotly_chart(clustered_timeseries_plots[cluster_select]['market'])
    st.subheader("Product Share by Neighborhood")
    st.plotly_chart(clustered_timeseries_plots[cluster_select][neighborhood])

    # CATEGORY MANAGEMENT TABLES 
    st.subheader("Category Management")
    st.plotly_chart(clustered_plotlies[cluster_select][neighborhood])
    st.markdown('###### Basket Averages for Category Drivers')
    st.markdown(""" <style> .stSelectbox > div:first-child { width: 100px;  /* Adjust the width as needed */ }</style> """, unsafe_allow_html=True)

    selects, cola, colb, colc, cold = st.columns(5)
    delt = True
    with selects:
        s1, s2 = st.columns(2)
        with s1:
            years = ['all']
            years = years + list(clustered_basket_stats[cluster_select]['year'].keys())
            selected_year = st.selectbox("Select Year", years)
            selected_quarter = 'all'
            if selected_year !='all':
                with s2: 
                    quarters = ['all']
                    quarters = quarters + list(clustered_basket_stats[cluster_select]['quarter'][years[1]].keys())
                    selected_quarter = st.selectbox("Select Quarter", quarters)
                    if selected_year == years[1]:
                        delt=False
    display_baskets = basket_stats(clustered_basket_stats[cluster_select], clustered_neighborhood_adj_stats[cluster_select], selected_year, selected_quarter)
    deltas = find_delta(display_baskets, clustered_basket_stats[cluster_select], clustered_neighborhood_adj_stats[cluster_select], selected_year, selected_quarter, delt)
    with cola:
        st.metric(label = 'Value per Basket', value = '$' + str(round(display_baskets[neighborhood]['basket_value'], 2)), delta=(str(deltas[neighborhood]['basket_value']) + '%') if deltas[neighborhood]['basket_value'] != None else None if selected_year != 'all' else None)
    with colb:
        st.metric(label = 'Items per Basket', value = str(round(display_baskets[neighborhood]['items_per_basket'], 1)), delta = (str(deltas[neighborhood]['items_per_basket']) + '%') if deltas[neighborhood]['items_per_basket'] != None else None if selected_year != 'all' else None)
    with colc:
        st.metric(label = 'Adjacent Value per Basket', value = '$' + str(round(display_baskets[neighborhood]['adjacent_value'], 2)), delta = (str(deltas[neighborhood]['adjacent_value']) + '%') if deltas[neighborhood]['adjacent_value'] != None else None if selected_year != 'all' else None)
    with cold:
        st.metric(label = 'Adjacent Items per Basket', value = str(round(display_baskets[neighborhood]['adjacent_items_per_basket'], 1)), delta = (str(deltas[neighborhood]['adjacent_items_per_basket']) + '%') if deltas[neighborhood]['adjacent_items_per_basket'] != None else None if selected_year != 'all' else None)
    st.write(' ')
    st.write(' ')
    st.write(' ')


    # ADVANCED INSIGHTS
    if 'advanced_insights' not in st.session_state:
        st.session_state.advanced_insights = False
    def toggle_tables():
        st.session_state.advanced_insights = not st.session_state.advanced_insights
    button_label = "Hide Advanced Insights" if st.session_state.advanced_insights else "Show Advanced Insights"
    if st.button(button_label, on_click=toggle_tables):
        pass
    if st.session_state.advanced_insights:
            product_aggregations = list(clustered_recommendations[cluster_select][neighborhood]['Unnamed: 0'])
            plot_ridge(clustered_time_df[cluster_select], product_aggregations, neighborhood+1)
    # CO-PURCHASED PRODUCTS TABLES
    st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line divider
    st.subheader("Top Co-Purchased Products for Category Drivers")
    neighborhood_adjacents(clustered_n_plots[cluster_select], neighborhood)





                
# %% 
