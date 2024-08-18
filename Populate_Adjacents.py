#%%        SETUP FOR NEIGHBORHOOD ADJACENCY 

#                 INPUTS
#       cdf:                                                   CSV of raw, each line is a product sold Dataset
#       quickStep:                                             Boolean to pull directly from a POI-Aggregated Dataset
#              aggregated_df                                   If quickStep, aggregated_df pulls in the CSV of a POI-Aggregated Dataset
      
#                 OUTPUT
#       pois.pkl:                                              List of Dictionaries of POI data from Normalized_Adjacency.py
#       adjacency_dictionary.pkl:                              List of Dictionaries of Adjacency data from Normalized_Adjacency.py
#       tops.pkl:                                              List of Dictionaries of Top Adjacencies Prepared to be input and formated into a Table
#       recommendation_1.csv,r...n_2, r...n_3:                 CSVs for each Neighborhood that holds neighborhood share information (share brand sales, share adjacent sales, etc) from Populate_Adjacents.py
#       Cluster_{i}_recommendation_1.csv,r...n_2, r...n_3:                 CSVs for each Neighborhood that holds neighborhood share information (share brand sales, share adjacent sales, etc) from Populate_Adjacents.py


from scipy.cluster.hierarchy import ward, fcluster, linkage, dendrogram
from scipy.spatial.distance import pdist, euclidean, cosine, correlation
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# ----- aggregate POIs -------
def find_poi(df):
    prods_of_focus = list(df[df['product_of_interest_flag']==1]['product_aggregation'].unique())
    focus_txn_dicts = []
    for prod in prods_of_focus:
        poi_dict = {}
        poi_dict['prod_agg'] = prod
        poi_mask = (df['product_aggregation'] == prod) & (df['product_of_interest_flag'] == 1)
        poi_dict['txn_lst'] = list(df[poi_mask]['transaction_id'].unique())
        focus_txn_dicts.append(poi_dict)
    return focus_txn_dicts

#roughly 8 mins. BIG BOY
def poi_aggregate(df, focus_txn_dicts, timer, download):
    prod_agg_dfs = []
    for y, prod_agg_dict in enumerate(focus_txn_dicts):
        poi_agg = prod_agg_dict['prod_agg']
        un_txns = prod_agg_dict['txn_lst']
        if timer:
            print(y)
            print(poi_agg)
        #print(txn_lst)

        # Filter the df to only include transactions in txn_lst
        filtered_df = df[df['transaction_id'].isin(un_txns)]

        #print(filtered_df)
        # Create a new df with all items in the transactions
        prod_agg_df = pd.DataFrame()
        for txn_id in un_txns:
            #print(txn_id)
            inner_df = filtered_df[filtered_df['transaction_id'] == txn_id]
            prod_agg_df = pd.concat([prod_agg_df, inner_df], ignore_index=True, axis = 0)

        # Add the prod_agg to each row of the df
        prod_agg_df['prim_poi'] = ''
        prod_agg_df['prim_poi'] = poi_agg
        prod_agg_dfs.append(prod_agg_df)

    print("POI aggregate finished!")
    main_df = pd.concat(prod_agg_dfs)
    if download:
        main_df.to_csv('assets/POI_aggregates.csv')

    return main_df

# ------ normalized adjacency ----- 
def separate_by_poi (cdf):
    grouped_df = cdf.groupby('prod_agg')
    dfs = {key: value for key, value in grouped_df}

    #populate each with aggregate adjacency sales
    for key, df in dfs.items():
        df_filtered = df[~((df['product_aggregation'] == df['prod_agg']) & (df['product_of_interest_flag'] == 1))]
        sums = df_filtered['dollar_sales'].sum()
        dfs[key]['total_adjacent_sales'] = sums
    return dfs

def get_total_txns(df):
    At = df['transaction_id'].nunique()
    return At

#input dfs!!!
def get_all_products(dfs):
    upa = []
    for key, df in dfs.items():
        upa.extend(df['product_aggregation'].unique())
    return upa

#big boy ~10 minutes. create a dictionary of all products + their transactions
def populate_adjacents (products, defDF, At, timer=False):
    adj_txns = {}
    i = 0
    percent = 0
    temp = 0
    tot_prod = len(products)
    print('Cleaning and Organizing Adjacents')
    print('0%')
    for prod in products:
        fdf = defDF[defDF['product_aggregation'] == prod]
        adj_txns[prod] = {'u_tx': list(fdf["transaction_id"].unique())}
        adj_txns[prod]['category'] = fdf['category'].iloc[0]
        adj_txns[prod]['subcategory'] = fdf['subcategory'].iloc[0]
        if timer:
            i+=1
            percent = int(i/tot_prod*100)
            if temp != percent:
                print(str(percent) + '%')
                temp = percent

    #calculate normalized frequency for all products
    for key in adj_txns:
        count = len(adj_txns[key]['u_tx'])
        adj_txns[key]['count'] = count
        adj_txns[key]['AA'] = count/At
    return adj_txns

#create a POIS dictionary. Key is each POI, holding every adjacent product + exta data
def populate_pois(adj_txns, dfs):
    #calculate total unique tnxs for each POI
    print('Calculating adjacents...')
    pois = {}
    for key in dfs:
        pois[key] = {
            'Atpoi': adj_txns[key]['count'],
            'u_tx': adj_txns[key]['u_tx'],
            'total_adjacent_sales': dfs[key]['total_adjacent_sales'].iloc[0]
        }
    #populate dictionary per POI for each Adj product 
    w= 0
    for poi in pois:
        adj_dic = {}
        for adj in adj_txns:
            if adj != poi:
                l = list(set(adj_txns[adj]['u_tx']) & set(pois[poi]['u_tx']));
                adj_dic[adj] = {
                    "txn_list": l,
                    "txn_count": len(l),
                    "category": adj_txns[adj]['category'],
                    "subcategory": adj_txns[adj]['subcategory']
                    }
        pois[poi]['adjacent'] = adj_dic
        w+=1
    print('POIs Populated!')
    return pois
    
def normalize_adjacency(pois, adj_txns):
    for poi in pois:
        for adj in pois[poi]['adjacent'].keys():
            Atpoi = pois[poi]['Atpoi']
            adj_target = pois[poi]['adjacent'][adj]
            AA = adj_txns[adj]['AA']
            M = (adj_target['txn_count']/Atpoi) 
            normalize_adjacency_calc = round(((M-AA)/AA), 4)
            try:
                pois[poi]['adjacent'][adj]['normalized_adjacency'] = normalize_adjacency_calc*100
            except ZeroDivisionError:
                pass
    print("Normalized adjacents populated!")

#select the top adjacents for each POI based on parameters
def select_top(poit, top, lim):
    top_pois = {}
    for poi in poit:
        top_pois[poi] = {
            'all': [],
            'spirits':  [],
            'beer': [],
            'wine': []
        }
        # Extract the adjacent products and their normalized adjacency values
        adjacents = poit[poi]['adjacent']
        # Sort the adjacents by normalized adjacency in descending order
        sorted_adjacents = list(sorted(adjacents.items(), key=lambda x: x[1].get('normalized_adjacency', float('-inf')), reverse=True))
        # Select the top 5 adjacents
        rank_l = [1,1,1,1]
        #select top 5
        for y in range(len(sorted_adjacents)):
            if sorted_adjacents[y][1]['txn_count'] >= lim and rank_l[0]+rank_l[1]+rank_l[2] <= top*3:
                sorted_adjacents[y][1]['rank']= rank_l[3]
                rank_l[3]+=1
                top_pois[poi]['all'].append(sorted_adjacents[y])
                if sorted_adjacents[y][1]['category'] == 'WINE' and rank_l[0] <= top:
                    sorted_adjacents[y][1]['rank']= rank_l[0]
                    rank_l[0]+=1
                    top_pois[poi]['wine'].append(sorted_adjacents[y])
                elif sorted_adjacents[y][1]['category'] == 'BEER' and rank_l[1] <= top:
                    sorted_adjacents[y][1]['rank']= rank_l[1]
                    rank_l[1]+=1
                    top_pois[poi]['beer'].append(sorted_adjacents[y])
                elif sorted_adjacents[y][1]['category'] == 'SPIRITS' and rank_l[2] <= top:
                    sorted_adjacents[y][1]['rank']= rank_l[2]
                    rank_l[2]+=1
                    top_pois[poi]['spirits'].append(sorted_adjacents[y])
    return top_pois            

#%% 
#########################
#####     MAIN     ######
#########################
global dfs
dfs = []
global products
products = []
global adjacency_dictionary
adjacency_dictionary = []
global pois
pois = []
global tops
tops = []
At = []
cdf = []


######### Sidestep POI Aggregation? ##########
quickStep = False  
##############################################


mdata = 'assets/requirements/shortdata.csv'
aggregated_datafile = 'assets/requirements/aggregations/Aggregated-Master.csv'

num_of_clusters = 4
clusters = [i for i in range(num_of_clusters+1)]
for i in clusters:
    if i==num_of_clusters:
        print(f'Master! {i}')
        cdf.append(pd.read_csv(mdata))

        filtered_df = cdf[i][~((cdf[i]['product_aggregation']=='CHAMPAGNE CANARD DUCHENE CHAMPAGNE') & (cdf[i]['product_of_interest_flag'] == 1))]

        if quickStep: 
            aggregated_df = pd.read_csv(aggregated_datafile)
            quick_step = aggregated_df[~((aggregated_df['product_aggregation']=='CHAMPAGNE CANARD DUCHENE CHAMPAGNE') & (aggregated_df['product_of_interest_flag'] == 1))]
        else:
            #Aggregate POIs
            poi_txn_dicts = find_poi(filtered_df)
            aggregated_df = poi_aggregate(cdf[i], poi_txn_dicts, True, False)
            #rename column
            aggregated_df.rename(columns={'prim_poi': 'prod_agg'}, inplace=True)
            aggregated_df.to_csv(aggregated_datafile)
    else:
        print(f'Cluster: {i}')
        #cdf = pd.read_csv('/Users/alexnichols/Desktop/Loeb.nyc/Champagne/vc_example_data_for_interns_v1.csv')
        cdf.append(pd.read_csv(f'assets/Cluster_{i}_Data.csv'))
        #filter out anything needed 
        filtered_df = cdf[i][~(cdf[i]['product_aggregation']=='CHAMPAGNE CANARD DUCHENE CHAMPAGNE') & cdf[i]['product_of_interest_flag'] == 1]

        if quickStep: 
            aggregated_df = pd.read_csv(f'assets/requirements/aggregations/Cluster_{i}_Aggregated.csv')
            quick_step = aggregated_df[~((aggregated_df['product_aggregation']=='CHAMPAGNE CANARD DUCHENE CHAMPAGNE') & (aggregated_df['product_of_interest_flag'] == 1))]
        else:
            #Aggregate POIs
            poi_txn_dicts = find_poi(filtered_df)
            aggregated_df = poi_aggregate(cdf[i], poi_txn_dicts, True, False)
            #rename column
            aggregated_df.rename(columns={'prim_poi': 'prod_agg'}, inplace=True)
            aggregated_df.to_csv(f'assets/requirements/aggregations/Cluster_{i}_Aggregated.csv')

#if you just ran entire aggregation, set quickstep to True

    dfs.append(separate_by_poi(aggregated_df))
    At.append(get_total_txns(aggregated_df))
    products.append(get_all_products(dfs[i]))
    adjacency_dictionary.append(populate_adjacents(products[i], cdf[i], At[i], True))
    pois.append(populate_pois(adjacency_dictionary[i], dfs[i]))
    normalize_adjacency(pois[i], adjacency_dictionary[i])
    tops.append(select_top(pois[i], 5, 15))





#%%  OUTPUT FILES 
import pickle

with open('assets/tops.pkl', 'wb') as file:
    pickle.dump(tops, file)

with open('assets/adjacency_dictionary.pkl', 'wb') as file:
    pickle.dump(adjacency_dictionary, file)

with open('assets/pois.pkl', 'wb') as file:
    pickle.dump(pois, file)


# %% 
#################################
#  CALCULATE NEIGHBORHOOD DATA  #
#################################


champ_n2 = ['VEUVE CLICQUOT CHAMPAGNE', 'LAURENT PERRIER CHAMPAGNE', 'MOET & CHANDON CHAMPAGNE',  'NICHOLAS FEUILLATTE CHAMPAGNE',  'PIPER HEIDSIECK CHAMPAGNE',  'LOUIS ROEDERER CHAMPAGNE',  'TAITTINGER CHAMPAGNE']
champ_n1 = ['VILLA JOLANDA PROSECCO',  'MIONETTO PROSECCO','RIONDO PROSECCO',  'MUMM NAPA PROSECCO', 'BAREFOOT PROSECCO', 'DOMAINE CHANDON PROSECCO',  'GRUET PROSECCO']


hoods = [champ_n1, champ_n2]

def category_management(initial_csv, pois, neighborhoods):
    nbr_dict = {}
    for i, nlist in enumerate(neighborhoods):
        nbr_dict[i]= {'pois': {}, 'total_sales_of_neighborhood': 0, 'total_adjacent_sales_of_neighborhood': 0}
        for p in nlist:
            if p in list(pois.keys()):
                nbr_dict[i]['pois'][p] = {}

    #get sales for each poi
    market_poi_sales = 0
    market_adjacent_sales = 0
    for poi4 in pois:
        list_of_txns = pois[poi4]['u_tx']
        filtered_df = initial_csv[(initial_csv['transaction_id'].isin(list_of_txns)) & (initial_csv['product_aggregation']==poi4)]
        pois[poi4]['total_poi_sales'] = filtered_df['dollar_sales'].sum()
        market_poi_sales+= pois[poi4]['total_poi_sales']
        market_adjacent_sales +=pois[poi4]['total_adjacent_sales']
        for i in nbr_dict:
            if poi4 in nbr_dict[i]['pois']:
                pois[poi4]['neighborhood'] = i
                nbr_dict[i]['total_sales_of_neighborhood'] += pois[poi4]['total_poi_sales']
                nbr_dict[i]['total_adjacent_sales_of_neighborhood'] += pois[poi4]['total_adjacent_sales']

    #Print and Check Values
    mk_check = 0
    for i in nbr_dict:
        mk_check += nbr_dict[i]['total_sales_of_neighborhood']

    #market and adjacent sales
    for poi1 in pois:
        pois[poi1]['SHARE_BRAND_SALES_of_market'] = pois[poi1]['total_poi_sales']/market_poi_sales*100
        pois[poi1]['SHARE_ADJACENT_SALES_of_market'] = pois[poi1]['total_adjacent_sales']/market_adjacent_sales*100

    #neighborhood share sales
    checker = [0,0,0]
    for i, n in enumerate(neighborhoods):
        for ps in n:
            if ps in pois:
                nbr = pois[ps]['neighborhood']
                pois[ps]['total_sales_of_neighborhood'] = nbr_dict[i]['total_sales_of_neighborhood']
                pois[ps]['total_adjacent_sales_of_neighborhood'] = nbr_dict[i]['total_adjacent_sales_of_neighborhood']
                pois[ps]['SHARE_BRAND_SALES_of_neighborhood'] = pois[ps]['total_poi_sales']/pois[ps]['total_sales_of_neighborhood'] *100
                pois[ps]['SHARE_ADJACENT_SALES_of_neighborhood'] = pois[ps]['total_adjacent_sales']/pois[ps]['total_adjacent_sales_of_neighborhood']*100
                pois[ps]['basket_drive'] = ((pois[ps]['SHARE_ADJACENT_SALES_of_neighborhood'] / pois[ps]['SHARE_BRAND_SALES_of_neighborhood'])-1) *100
                checker[nbr]+=pois[ps]['SHARE_BRAND_SALES_of_neighborhood']
                checker[nbr]+=pois[ps]['SHARE_ADJACENT_SALES_of_neighborhood']
                nbr_dict[i]['pois'][ps]['total_poi_sales'] = pois[ps]['total_poi_sales']
                nbr_dict[i]['pois'][ps]['total_adjacent_sales'] = pois[ps]['total_adjacent_sales']
                nbr_dict[i]['pois'][ps]['SHARE_BRAND_SALES_NBRH'] = pois[ps]['SHARE_BRAND_SALES_of_neighborhood']
                nbr_dict[i]['pois'][ps]['SHARE_ADJACENT_SALES_NBRH'] = pois[ps]['SHARE_ADJACENT_SALES_of_neighborhood']
                nbr_dict[i]['pois'][ps]['BASKET_DRIVE'] = pois[ps]['basket_drive']
               


    #turn output into a dataframe per neighborhood
    neighborhood_df = []
    for i in nbr_dict:
        df_from_dictionary = pd.DataFrame(nbr_dict[i]['pois']).transpose()
        df_from_dictionary = df_from_dictionary.sort_values(by='SHARE_BRAND_SALES_NBRH',ascending=False)
        df_from_dictionary['CUMULATIVE_SHARE_BRAND_SALES'] = df_from_dictionary['SHARE_BRAND_SALES_NBRH'].cumsum()
        neighborhood_df.append(df_from_dictionary)

    return neighborhood_df

global table_stats
table_stats = []
for i, data in enumerate(pois):
    table_stats.append(category_management(cdf[i], pois[i], hoods))



#basket drive: how much adjacent value does a product drive?
# %% RECCOMENDATON ALGORITHM

def apply_recommendation(category_management_output):
    for nhood in category_management_output:
        lower2 = True
        i=0
        nhood['recommendation'] = 'Pending'
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
    return category_management_output

def output_3(df_list, cluster, master_cluster):
    for i, df in enumerate(df_list):
        df = df.dropna()
        if cluster==master_cluster-1:
            df.to_csv('assets/recommendation_'+str(i)+'.csv')
        else:
            df.to_csv('assets/Cluster_'+str(cluster)+'_recommendation_'+str(i)+'.csv')

for i, data in enumerate(table_stats):  
    print("Applying Recommendations")
    table_stats[i] = apply_recommendation(data)   
    output_3(table_stats[i], i, len(table_stats))
# %% 

sets = [set(list(i.keys())) for i in pois]
# Find the intersection of all sets
common_products = set.intersection(*sets)

# Convert the set back to a list if needed
common_products_list = list(common_products)
# %%
