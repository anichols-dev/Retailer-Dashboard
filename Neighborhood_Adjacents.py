#%%    SETUP FOR NEIGHBORHOOD ADJACENCY 

#                 INPUTS
#       pois.pkl:                                              Dictionary of POI data from Populate_Adjacents.py
#       adjacency_dictionary.pkl:                              Dictionary of Adjacency data from Populate_Adjacents.py
#       recommendation_1.csv,r...n_2, r...n_3:                 CSV for each Neighborhood that holds neighborhood share information (share brand sales, share adjacent sales, etc) from Populate_Adjacents.py

#                 OUTPUT
#      n_adj_plots ('neighborhood_adjacents_dictionary.pkl):    Dictionary of Neighborhood Adjacent Calculations  


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

global adjacency_dictionary
global pois
global t_stats
global number_of_neighborhoods

number_of_neighborhoods = 2


# Load the dictionary from the file

with open('assets/pois.pkl', 'rb') as file:
    pois = pickle.load(file)
with open('assets/adjacency_dictionary.pkl', 'rb') as file:
    adjacency_dictionary = pickle.load(file)

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


##########################################   FUNCTIONS   ##############################################
#%% assemble transactions with top 80% of neighborhood and the adjacent trasnactions for that group

#assemble neighborhoods, select the top 80, and find the adjacent transactions for the group as a whole
def assemble_top_80(t_stats, pois, adj_dictionary):
    neighbor_adjacents = {}
    #make a list of the top 80% of poi's in each neighborhood
    neighborhoods = [[] for i in range(number_of_neighborhoods)]
    print(neighborhoods)
    for number, ndf in enumerate(t_stats):
        cumulative = 0
        neighbor_adjacents[number] = {'top_80': [], 
            'poi_txns': {},
            'all_txns': [],
            'adjacents': {},
            }
        for index, row in ndf.iterrows():
            if cumulative <= 80:
                cumulative += row['SHARE_BRAND_SALES_NBRH'] 
                neighborhoods[number].append(row['Unnamed: 0'])

    test = 1
    print(neighborhoods)
    for nnum, nlist in enumerate(neighborhoods):
        test+=1
        for poi in pois:
            if poi in neighborhoods[nnum]: #for the top 80% of products
                #poi transactions
                txns = pois[poi]['u_tx']
                neighbor_adjacents[nnum]['top_80'].append(poi)
                neighbor_adjacents[nnum]['poi_txns'][poi] = txns
                neighbor_adjacents[nnum]['all_txns'] = list(set(neighbor_adjacents[nnum]['all_txns']).union(txns))
                #adjacent transactions
                for adjacent in pois[poi]['adjacent']:
                    old_adjacent = pois[poi]['adjacent'][adjacent]
                    if adjacent not in neighbor_adjacents[nnum]['adjacents']:
                        if old_adjacent['txn_list']:
                            new_adjacent = {'normalized_adjacency': 0, 'pct_basket_all': 0, 'compiled_txns': [], 'history': {}, 'txn_count': 0, 'category': '', 'subcategory': ''
                                }
                            new_adjacent['category'] = old_adjacent['category']
                            new_adjacent['subcategory'] = old_adjacent['subcategory']
                            new_adjacent['compiled_txns'] = old_adjacent['txn_list']
                            new_adjacent['history'][poi] = old_adjacent['txn_list']
                            new_adjacent['pct_basket_all'] = adj_dictionary[adjacent]['AA']
                            neighbor_adjacents[nnum]['adjacents'][adjacent] = new_adjacent
                    else:
                        new_adjacent = neighbor_adjacents[nnum]['adjacents'][adjacent]
                        new_adjacent['compiled_txns'] = list(set(new_adjacent['compiled_txns']).union(old_adjacent['txn_list']))
                        new_adjacent['txn_count'] = len(new_adjacent['compiled_txns'])
                        if old_adjacent['txn_list']:
                            new_adjacent['history'][poi] = old_adjacent['txn_list']
    return neighbor_adjacents

#filter out any outliers
def clean_adjacents(neighbor_adjacents, num_neighborhoods):
    for nbr in range(num_neighborhoods):
        top80 = neighbor_adjacents[nbr]['top_80']
        for poi in top80:
            
            words = poi.split()
            if len(words) == 2:
                inverse_poi = f"{words[1]} {words[0]}"
            elif len(words) == 3:
                inverse_poi = f"{words[2]} {words[0]} {words[1]}"
            elif len(words) == 4:
                inverse_poi = f"{words[3]} {words[0]} {words[1]} {words[2]}"

            if inverse_poi in neighbor_adjacents[nbr]['adjacents'].keys():
                print(f'Deleted: ' + str(inverse_poi) + ' with txn_count: ' + str(neighbor_adjacents[nbr]['adjacents'][inverse_poi]['txn_count']))
                del neighbor_adjacents[nbr]['adjacents'][inverse_poi]
    return neighbor_adjacents

#%%
# Computation to Find Normalize Adjacency
def compute_adjacencies(adj_data, printBool):
    for nbhd in adj_data:
        dicts = adj_data[nbhd]
        poi_txn_count = len(dicts['all_txns'])
        for adj in dicts['adjacents']:
            AA = dicts['adjacents'][adj]['pct_basket_all']*100
            A = dicts['adjacents'][adj]['txn_count']
            M = A / poi_txn_count *100
            adjacency_calc = ((M-AA)/AA)
            if printBool:
                print('\n=======================================')
                print(adj)
                print(f'AA:  {AA}')
                print(f'Adjacent transaction count:  {A}')
                print(f'Neighborhood transaction count:  {poi_txn_count}')
                print(f'M:  {M}')
                print(f'adjacency calc: {adjacency_calc}')
            dicts['adjacents'][adj]['AA'] = AA
            dicts['adjacents'][adj]['M'] = M
            dicts['adjacents'][adj]['scaled_adjacency'] = adjacency_calc*100*np.log(AA+1)
            dicts['adjacents'][adj]['normalized_adjacency'] = adjacency_calc*100
    return adj_data


#%% 
#Select top adjacencies for each neighborhood
def select_top_nbr(neighbor, top, lim):
    top_adjacents = {
            0: {
                'all': [],
                'spirits':  [],
                'beer': [],
                'wine': [] },
            1:{ 
                'all': [],
                'spirits':  [],
                'beer': [],
                'wine': [] },
            2:{
                'all': [],
                'spirits':  [],
                'beer': [],
                'wine': [] }
        }
    for nbr in neighbor:
        # Extract the adjacent products and their normalized adjacency values
        adjacents = neighbor[nbr]['adjacents']
        global sorted_adjacents
        # Sort the adjacents by normalized adjacency in descending order
        sorted_adjacents = list(sorted(adjacents.items(), key=lambda x: x[1].get('scaled_adjacency', float('-inf')), reverse=True))
        # Select the top 5 adjacents
        rank_l = [1,1,1,1]
        #select top 5
        for y in range(len(sorted_adjacents)):
            if not sorted_adjacents[y][0] in neighbor[nbr]['top_80']:
                if sorted_adjacents[y][1]['txn_count'] >= lim[nbr] or rank_l[0]+rank_l[1]+rank_l[2] <= top*3:
                    if sorted_adjacents[y][1]['category'] == 'WINE' and rank_l[0] <= top:
                        sorted_adjacents[y][1]['rank']= rank_l[0]
                        rank_l[0]+=1
                        top_adjacents[nbr]['wine'].append(sorted_adjacents[y])
                        #all
                        sorted_adjacents[y][1]['rank']= rank_l[3]
                        rank_l[3]+=1
                        top_adjacents[nbr]['all'].append(sorted_adjacents[y])
                    elif sorted_adjacents[y][1]['category'] == 'BEER' and rank_l[1] <= top:
                        sorted_adjacents[y][1]['rank']= rank_l[1]
                        rank_l[1]+=1
                        top_adjacents[nbr]['beer'].append(sorted_adjacents[y])
                        #all
                        sorted_adjacents[y][1]['rank']= rank_l[3]
                        rank_l[3]+=1
                        top_adjacents[nbr]['all'].append(sorted_adjacents[y])
                    elif sorted_adjacents[y][1]['category'] == 'SPIRITS' and rank_l[2] <= top:
                        sorted_adjacents[y][1]['rank']= rank_l[2]
                        rank_l[2]+=1
                        top_adjacents[nbr]['spirits'].append(sorted_adjacents[y])
                        #all
                        sorted_adjacents[y][1]['rank']= rank_l[3]
                        rank_l[3]+=1
                        top_adjacents[nbr]['all'].append(sorted_adjacents[y])
    return top_adjacents          


#%% POPULTE PLOTLY PLOTS

def populate_plots(top_adj, number_of_nbrhds):
    #setup dictionary to hold table information
    n_plots = {}
    for i in range(number_of_nbrhds):
        temp_dict = {'plotly': {}, 'fig': {'all':None,  'wine':None, 'beer':None, 'spirits':None}}
        n_plots[i] = temp_dict

    for i in n_plots:
        plotly = {
                'all': {
                    'adj': [],
                    'rank': [],
                    'category':[],
                    'subcategory':[],
                    'txn_counts': [],
                    'normalized_adjacency': [],
                    'scaled_adjacency': []
                    }, 
                'spirits': {
                    'adj': [],
                    'rank': [],
                    'category':[],
                    'subcategory':[],
                    'txn_counts': [],
                    'normalized_adjacency': [],
                    'scaled_adjacency': []
                    },
                'beer': {
                    'adj': [],
                    'rank': [],
                    'category':[],
                    'subcategory':[],
                    'txn_counts': [],
                    'normalized_adjacency': [],
                    'scaled_adjacency': []
                    },
                'wine': {
                    'adj': [],
                    'rank': [],
                    'category':[],
                    'subcategory':[],
                    'txn_counts': [],
                    'normalized_adjacency': [],
                    'scaled_adjacency': []
                    },
                }
        for category, data in top_adj[i].items():
            for adjacent, d2 in data:
                plotly[category]['adj'].append(adjacent)
                plotly[category]['rank'].append(d2['rank'])
                plotly[category]['category'].append(d2['category'])
                plotly[category]['subcategory'].append(d2['subcategory'])
                plotly[category]['txn_counts'].append(d2['txn_count'])
                plotly[category]['normalized_adjacency'].append(d2['normalized_adjacency'])
                plotly[category]['scaled_adjacency'].append(d2['scaled_adjacency'])
        n_plots[i]['plotly'] = plotly

    # FULL SIZED TABLE
    # for nbr in n_plots:
    #     for cat in n_plots[nbr]['plotly']:
    #         if cat == 'all':
    #             fig = go.Figure(data=[go.Table(
    #                 header= dict(values =['<b>ADJACENT</b>', '<b>CATEGORY</b>','<b>TRANSACTION COUNT</b>', '<b>SCALED ADJACENCY</b>', '<b>NORMALIZED ADJACENCY</b>'],
    #                     fill_color='paleturquoise',
    #                     align='center'),
    #                 cells=dict(values=[n_plots[nbr]['plotly'][cat]['adj'], n_plots[nbr]['plotly'][cat]['category'], n_plots[nbr]['plotly'][cat]['txn_counts'], [f'{val:.2f}%' for val in n_plots[nbr]['plotly'][cat]['scaled_adjacency']], [f'{val:.2f}%' for val in n_plots[nbr]['plotly'][cat]['normalized_adjacency']]],
    #                                     fill_color='lavender')
    #                 )
    #             ])
    #             fig.update_layout(title='Neighborhood ' + str(nbr+1) + ': ' + str(cat).upper(),
    #                                 autosize = True,  margin=dict(l=0, r=0, t=30, b=0))
    #         else:
    #             fig = go.Figure(data=[go.Table(
    #                 header= dict(values =['<b>ADJACENT</b>', '<b>SUBCATEGORY</b>','<b>TRANSACTION COUNT</b>', '<b>SCALED ADJACENCY</b>', '<b>NORMALIZED ADJACENCY</b>'],
    #                     fill_color='paleturquoise',
    #                     align='center'),
    #                 cells=dict(values=[n_plots[nbr]['plotly'][cat]['adj'], n_plots[nbr]['plotly'][cat]['subcategory'], n_plots[nbr]['plotly'][cat]['txn_counts'], [f'{val:.2f}%' for val in n_plots[nbr]['plotly'][cat]['scaled_adjacency']], [f'{val:.2f}%' for val in n_plots[nbr]['plotly'][cat]['normalized_adjacency']]],
    #                                     fill_color='lavender')
    #                 )
    #             ])
    #             fig.update_layout(title='Neighborhood ' + str(nbr+1) + ': ' + str(cat).upper(),
    #                                 autosize = True,  margin=dict(l=0, r=0, t=30, b=0))
    #         n_plots[nbr]['fig'][cat] = fig

    #CLEANED TABLE
    for nbr in n_plots:
        for cat in n_plots[nbr]['plotly']:
            if cat == 'all':
                fig = go.Figure(data=[go.Table(
                    header= dict(values =['<b>ADJACENT</b>', '<b>CATEGORY</b>', '<b>SCALED ADJACENCY</b>'],
                        align='center', fill_color='lavender'),
                    cells=dict(values=[n_plots[nbr]['plotly'][cat]['adj'], n_plots[nbr]['plotly'][cat]['category'], [f'{val:.2f}%' for val in n_plots[nbr]['plotly'][cat]['scaled_adjacency']]])
                    )
                ])
                fig.update_layout(title='Neighborhood ' + str(nbr+1) + ': ' + str(cat).upper(),
                                    autosize = True,  margin=dict(l=0, r=0, t=30, b=0))
            else:
                fig = go.Figure(data=[go.Table(
                    header= dict(values =['<b>ADJACENT</b>', '<b>SUBCATEGORY</b>', '<b>SCALED ADJACENCY</b>'],align='center', fill_color='lavender'),
                    cells=dict(values=[n_plots[nbr]['plotly'][cat]['adj'], n_plots[nbr]['plotly'][cat]['subcategory'], [f'{val:.2f}%' for val in n_plots[nbr]['plotly'][cat]['scaled_adjacency']]])
                    )
                ])
                fig.update_layout(title='Neighborhood ' + str(nbr+1) + ': ' + str(cat).upper(),
                                    autosize = True,  margin=dict(l=0, r=0, t=30, b=0))
            n_plots[nbr]['fig'][cat] = fig
    return n_plots




##########################################   EXECUTION   ##############################################
#%% CALL THE FUNCTIONS

clustered_neighbor_adjacent_data = []
clustered_select_top = []
clustered_adjacent_plots = []
clustered_25th_limit = []

for i in range(len(pois)):
    print(f'Cluster {i}')
    neighbor_adjacent_data = None
    neighbor_adjacent_data = assemble_top_80(clustered_recommendations[i], pois[i], adjacency_dictionary[i]) 
    neighbor_adjacent_data = compute_adjacencies(neighbor_adjacent_data, False) #(data on adjacents, should I print out info?)
    neighbor_adjacent_data = clean_adjacents(neighbor_adjacent_data, number_of_neighborhoods)
    clustered_neighbor_adjacent_data.append(neighbor_adjacent_data)

    # create a limit list per neighborhood
    twofive = []
    for nbr in range(number_of_neighborhoods):
        if neighbor_adjacent_data[nbr]:
            masta = pd.DataFrame(neighbor_adjacent_data[nbr]['adjacents'])
            masta = masta.transpose()
            masta = masta[['normalized_adjacency', 'scaled_adjacency', 'txn_count', 'AA', 'M']]
            masta = masta[masta['txn_count']>5] #remove wildly small txn_counts
            masta = masta.astype(float)
            desc = masta.describe()
            twofive.append(desc.loc['25%','txn_count'])
    clustered_25th_limit.append(twofive)

    testa_top = None
    testa_top = select_top_nbr(neighbor_adjacent_data, 5, twofive)  #(data on adjacents, select the top ___ adjacents, transaction count minimum)
    n_adj_plots = None
    n_adj_plots = populate_plots(testa_top, number_of_neighborhoods)  #(top adjacents, number of neighborhoods)

    clustered_select_top.append(testa_top)
    clustered_adjacent_plots.append(n_adj_plots)

#%% OUTPUT THE DATA

with open('assets/clustered_adjacent_plots.pkl', 'wb') as file:
    pickle.dump(clustered_adjacent_plots, file)

with open('assets/clustered_neighborhood_adjacent_stats.pkl', 'wb') as file:
    pickle.dump(clustered_neighbor_adjacent_data, file)


#%% SHOW FIGURES 
#show them all!
nta = clustered_adjacent_plots[1]
for i in nta:
    for cat, figure in nta[i]['fig'].items():
        if cat == 'all':
            #figure.show()
            pass


#%%   Dataframe of top for viewing purposes

# masta = pd.DataFrame(neighbor_adjacent_data[1]['adjacents'])
# masta = masta.transpose()
# masta = masta[['normalized_adjacency', 'scaled_adjacency', 'txn_count', 'AA', 'M']]
# masta = masta.sort_values(by='scaled_adjacency', ascending = False)
# pd.set_option('display.max_rows', None)
# masta = masta[masta['txn_count']>5]
# masta = masta.astype(float)


