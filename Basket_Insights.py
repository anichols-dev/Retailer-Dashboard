#%%    SETUP FOR BASKET INSIGHTS

#                 INPUTS
#       cdf:                                                   CSV of raw, each line is a product sold Dataset
#       quickStep:                                             Boolean to pull directly from a POI-Aggregated Dataset
#              aggregated_df                                   If quickStep, aggregated_df pulls in the CSV of a POI-Aggregated Dataset
      
#                 OUTPUT
#       
#
#
#

#                  DATA
# 
#        dict_by_date = {
#                             date: {
#                                 prod_agg: {
#                                     'transaction_ids': [list of transaction IDs],
#                                     'dollar_sales': [list of dollar sales],
#                                     'unit_sales': [list of unit sales],
#                                     'category': [list of categories],
#                                     'subcategory': [list of subcategories]
#                                 },
# 
#          dict_by_txn = {
#                             transaction_id: {
#                                     'products': [],
#                                     'dollar_sales': [],
#                                     'unit_sales': [],
#                                     'category': [],
#                                     'subcategory': [],
#                                     'df': None,
#                                     'txn_total': 0
#                                 }
# # 

#%%
from scipy.cluster.hierarchy import ward, fcluster, linkage, dendrogram
from scipy.spatial.distance import pdist, euclidean, cosine, correlation
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime
import pickle

mdata = 'assets/requirements/shortdata.csv'
aggregated_datafile = 'assets/requirements/aggregations/Aggregated-Master.csv'


maindf = pd.read_csv(mdata)

#filter out anything needed 
filtered_df = maindf[~((maindf['product_aggregation']=='CHAMPAGNE CANARD DUCHENE CHAMPAGNE') & (maindf['product_of_interest_flag'] == 1))]

global pois

with open('assets/pois.pkl', 'rb') as file:
    pois = pickle.load(file)

quickStep = True  # Sidestep POI Aggregation? 
quickLists = []


if quickStep: 
    for i in range(len(pois)-1):
        quick_step = pd.read_csv(f'assets/requirements/aggregations/Cluster_{i}_Aggregated.csv')
        quick_step = quick_step[~((quick_step['product_aggregation']=='CHAMPAGNE CANARD DUCHENE CHAMPAGNE') & (quick_step['product_of_interest_flag'] == 1))]
        quickLists.append(quick_step)
    quick_step = pd.read_csv(aggregated_datafile)
    quickLists.append(quick_step)
else:
    pass
    #Aggregate POIs
    #poi_txn_dicts = find_poi(filtered_df)
    #aggregated_df = poi_aggregate(cdf, poi_txn_dicts, True, False)
    #rename column
    #aggregated_df.rename(columns={'prim_poi': 'prod_agg'}, inplace=True)
# Ensure the transaction_datetime column is in datetime format
for i, data in enumerate(pois):
    quickLists[i]['transaction_datetime'] = pd.to_datetime(quickLists[i]['transaction_datetime'])

    # Extract the date part from the transaction_datetime
    quickLists[i]['transaction_date'] = quickLists[i]['transaction_datetime'].dt.date
    quickLists[i]['transaction_time'] = quickLists[i]['transaction_datetime'].dt.time


def select_time(start, end, df, poi):
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
    output = df[df['transaction_datetime'] > start and df['transaction_datetime'] < end]
    return output

#%%  Initialize the nested dictionary


dct_by_date_clustered = []

global test_date

test_date = datetime.date(2021, 1, 28)

# Iterate through the DataFrame to populate the nested dictionary
# 10 seconds
print("Parsing Data by Date....")
for cluster, data in enumerate(pois):

    dict_by_date = {}
    for i, row in quickLists[cluster].iterrows():
        date = row['transaction_date']
        prod_agg = row['prod_agg']
        
        # Initialize the date dictionary if not already present
        if date not in dict_by_date:
            dict_by_date[date] = {}
        
        # Initialize the prod_agg dictionary if not already present
        if prod_agg not in dict_by_date[date]:
            dict_by_date[date][prod_agg] = {
                'transaction_ids': [],
                'products': [],
                'dollar_sales': [],
                'unit_sales': [],
                'category': [],
                'subcategory': [],
                'df': None
            }
        
        # Append the relevant data to the lists
        dict_by_date[date][prod_agg]['transaction_ids'].append(row['transaction_id'])
        dict_by_date[date][prod_agg]['products'].append(row['product_aggregation'])
        dict_by_date[date][prod_agg]['dollar_sales'].append(row['dollar_sales'])
        dict_by_date[date][prod_agg]['unit_sales'].append(row['unit_sales'])
        dict_by_date[date][prod_agg]['category'].append(row['category'])
        dict_by_date[date][prod_agg]['subcategory'].append(row['subcategory'])

    #maybe populate a df here?
    by_date_df = {}
    for date, data in dict_by_date.items():
        if date not in by_date_df:
            by_date_df[date] = {}
        for poi, p_data in data.items():
            if poi not in by_date_df[date]:
                by_date_df[date][poi] = None
            tempdf = pd.DataFrame({
                        'transaction_ids': p_data.get('transaction_ids', []),
                        'products': p_data.get('products', []),
                        'dollar_sales': p_data.get('dollar_sales', []),
                        'unit_sales': p_data.get('unit_sales', []),
                        'category': p_data.get('category', []),
                        'subcategory': p_data.get('subcategory', [])
                    })
            by_date_df[date][poi] = tempdf
    dct_by_date_clustered.append(by_date_df)

#%% DATE STATS

# veuve_data = []

# #Veuve Cliquot 
# for date, data in dict_by_date.items():
#     if data['VEUVE CLICQUOT CHAMPAGNE']:
#         tempdf = pd.DataFrame({
#                 'transaction_ids': data['VEUVE CLICQUOT CHAMPAGNE'].get('transaction_ids', []),
#                 'products': data['VEUVE CLICQUOT CHAMPAGNE'].get('products', []),
#                 'dollar_sales': data['VEUVE CLICQUOT CHAMPAGNE'].get('dollar_sales', []),
#                 'unit_sales': data['VEUVE CLICQUOT CHAMPAGNE'].get('unit_sales', []),
#                 'category': data['VEUVE CLICQUOT CHAMPAGNE'].get('category', []),
#                 'subcategory': data['VEUVE CLICQUOT CHAMPAGNE'].get('subcategory', [])
#             })
#         tempdf = tempdf[tempdf['products']=='VEUVE CLICQUOT CHAMPAGNE']
#         veuve_sales_count = tempdf['unit_sales'].sum()
#         veuve_data.append({'date': date, 'sales_count': veuve_sales_count})
# veuve_data = pd.DataFrame(veuve_data)
# veuve_data['date'] = pd.to_datetime(veuve_data['date'])
# #VIOLIN PLOT!!

# plt.figure(figsize=(10, 6))
# sns.displot(data=veuve_data, x='date', kind='kde')
# plt.title('Frequency of Veuve Clicquot Sales per Day')
# plt.xlabel('Date')
# plt.ylabel('Sales Count')
# plt.xticks(rotation=45)
# plt.show()


# %% BY TRANSACTION
dct_by_txn_clustered = []
print("Parsing Data by Transaction....")

for cluster, data in enumerate(pois):
    dict_by_txn = {}
    for i, row in quickLists[cluster].iterrows():
        txn = row['transaction_id']
        prod_agg = row['prod_agg']
        
        # Initialize the date dictionary if not already present
        if txn not in dict_by_txn:
            dict_by_txn[txn] = {
                'products': [],
                'dollar_sales': [],
                'unit_sales': [],
                'category': [],
                'subcategory': [],
                'df': None,
                'txn_total': 0
            }
        
        # Append the relevant data to the lists
        dict_by_txn[txn]['products'].append(row['product_aggregation'])
        dict_by_txn[txn]['dollar_sales'].append(row['dollar_sales'])
        dict_by_txn[txn]['unit_sales'].append(row['unit_sales'])
        dict_by_txn[txn]['category'].append(row['category'])
        dict_by_txn[txn]['subcategory'].append(row['subcategory'])
        dict_by_txn[txn]['txn_total'] += row['dollar_sales']

    for txn, data in dict_by_txn.items():
        tempdf = pd.DataFrame({
                    'products': data.get('products', []),
                    'dollar_sales': data.get('dollar_sales', []),
                    'unit_sales': data.get('unit_sales', []),
                    'category': data.get('category', []),
                    'subcategory': data.get('subcategory', [])
                })
        dict_by_txn[txn]['df'] = tempdf
    dct_by_txn_clustered.append(dict_by_txn)





# %% POI GENERAL BASKET STATS

def populate_basket_stats(prod_of_in, txn_dict, df, start=0, end=0):
    #variable prepping
    poi_basket_stats = {}
    if start ==0:
        start = quickLists[len(quickLists)-1]['transaction_datetime'].min()
    if end ==0:
        end = quickLists[len(quickLists)-1]['transaction_datetime'].max()

    for poi in prod_of_in:
        #setup dict
        if poi not in poi_basket_stats:
            poi_basket_stats[poi] = {'basket_value': 0, 'items_per_basket': 0}
        basket_v = []
        items = []
        adj_basket_v = []
        adj_items = []

        #filter dataframe
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        limited_df = df[(df['transaction_datetime'] > start) & (df['transaction_datetime'] < end)]
        limited_df = limited_df[limited_df['product_aggregation'] == poi]

        txn_list = limited_df['transaction_id'].unique()
        #sum all info
        for txn in txn_list:
                basket_v.append(txn_dict[txn]['txn_total'])
                items.append(sum(list(txn_dict[txn]['unit_sales'])))
                adjacent_df = txn_dict[txn]['df'][txn_dict[txn]['df']['products'] != poi]
                adj_basket_v.append(sum(list(adjacent_df['dollar_sales'])))
                adj_items.append(sum(list(adjacent_df['unit_sales'])))
        #calculate averages
        txn_count = len(txn_list)
        if txn_count != 0:
            poi_basket_stats[poi]['total_basket_val'] = sum(basket_v)
            poi_basket_stats[poi]['total_items'] = sum(items)
            poi_basket_stats[poi]['total_adjacent_basket_val'] = sum(adj_basket_v)
            poi_basket_stats[poi]['total_adjacent_items'] = sum(adj_items)
            poi_basket_stats[poi]['total_txns'] = txn_count
            poi_basket_stats[poi]['basket_value'] = np.mean(basket_v)
            poi_basket_stats[poi]['items_per_basket'] = np.mean(items)
            poi_basket_stats[poi]['adjacent_value'] = np.mean(adj_basket_v)
            poi_basket_stats[poi]['adjacent_items_per_basket'] = np.mean(adj_items)
    return poi_basket_stats



#%% OUTPUT poi_basket_stats ~ 2 minutes
#def pop_all_times():
clustered_basket_stats = []
print("Calculating Basket Statistics")
for i in range(len(pois)):
    aggregated_df = quickLists[i]
    various_times_basket_stats = {'year': {}, 'quarter':{}, 'all':{}}
    oldest_date = aggregated_df['transaction_datetime'].min()
    youngest_date = aggregated_df['transaction_datetime'].max()


    for year, group in aggregated_df.groupby(aggregated_df['transaction_datetime'].dt.year):
        old = group['transaction_datetime'].min()
        young = group['transaction_datetime'].max()
        various_times_basket_stats['year'][year] = populate_basket_stats(pois[i], dct_by_txn_clustered[i], aggregated_df, old, young)

    for (year, quarter), group in aggregated_df.groupby([aggregated_df['transaction_datetime'].dt.year, aggregated_df['transaction_datetime'].dt.quarter]):
        if year not in various_times_basket_stats['quarter']:
            various_times_basket_stats['quarter'][year] = {}
        if year not in various_times_basket_stats['quarter'][year]:
            various_times_basket_stats['quarter'][year][quarter] = {}
        old = group['transaction_datetime'].min()
        young = group['transaction_datetime'].max()
        various_times_basket_stats['quarter'][year][quarter] = populate_basket_stats(pois[i],  dct_by_txn_clustered[i], aggregated_df, old, young)

    various_times_basket_stats['all'] = populate_basket_stats(pois[i],  dct_by_txn_clustered[i], aggregated_df)
    clustered_basket_stats.append(various_times_basket_stats)

#%%  VISUALIZE TIME TRENDS

years = sorted(clustered_basket_stats[0]['quarter'].keys())
quarters = [1,2,3,4]
basket_values = []

for year in years:
    for quarter in quarters:
        if quarter in clustered_basket_stats[0]['quarter'][year]:
            basket_value = clustered_basket_stats[0]['quarter'][year][quarter]['VEUVE CLICQUOT CHAMPAGNE']['items_per_basket']
            basket_values.append((f"{year}-{quarter}", basket_value))

# Split data into two lists for plotting
quarters_list, values_list = zip(*basket_values)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(quarters_list, values_list, marker='o', linestyle='-', color='b')

# Adding titles and labels
plt.title('Change in Basket Value for VEUVE CLICQUOT CHAMPAGNE Over Time')
plt.xlabel('Time (Year-Quarter)')
plt.ylabel('Basket Value')
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
#plt.show()


#%%



# %% FIGURE BASKET STATS

# Define the metrics to plot
metrics = ['basket_value', 'items_per_basket', 'adjacent_value', 'adjacent_items_per_basket']
titles = ['Basket Value per POI', 'Items per Basket per POI', 'Adjacent Basket Value per POI', 'Adjacent Items per Basket per POI']
y_labels = ['Basket Value', 'Items per Basket', 'Adjacent Basket Value', 'Adjacent Items per Basket']

poi_stats_df = pd.DataFrame.from_dict(clustered_basket_stats[0]['all'], orient='index')
palette = sns.color_palette("tab20", len(poi_stats_df))
poi_colors = {poi: color for poi, color in zip(poi_stats_df.index, palette)}


# # Create the plots using a for loop
# for i, metric in enumerate(metrics):
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=poi_stats_df.index, y=metric, data=poi_stats_df, palette=palette, hue=poi_stats_df.index)
#     plt.title(titles[i])
#     plt.xlabel('Product of Interest')
#     plt.ylabel(y_labels[i])

#     handles = [plt.Rectangle((0,0),1,1, color=poi_colors[poi]) for poi in poi_stats_df.index]
#     plt.legend(handles, poi_stats_df.index, title="POI", bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     #plt.show()

# %%   TIME OF DAY PLOT!!!
clustered_time_df = []
for i in range(len(pois)):
    time_df = quickLists[i][['product_aggregation', 'prod_agg', 'transaction_time', 'transaction_datetime']]
    time_df = time_df[time_df['prod_agg'] == time_df['product_aggregation']]
    time_df['time_history'] = time_df['transaction_time']
    time_df['hour'] = time_df['transaction_datetime'].dt.hour
    time_df['transaction_time'] = time_df['transaction_time'].apply(lambda x: x.hour * 60 + x.minute)
    time_df = time_df.sort_values(by='transaction_time', ascending=True)
    clustered_time_df.append(time_df)


def plot_ridge(data, product_aggregations):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    # Filter data for selected product_aggregations
    filtered_data = data[data['product_aggregation'].isin(product_aggregations)]
    
    
    # Create the ridge plot
    sns.violinplot(x="hour", y="product_aggregation", data=filtered_data, bw_method=.15, density_norm='width', split=True, native_scale=True)
    plt.title('Density of Purchase Times by Product Aggregation')
    plt.xlabel('Time of Day')
    plt.ylabel('Product Aggregation')
    
    plt.xlim(9, 22)

    
    
    #plt.show()

# Example usage
product_aggregations = clustered_time_df[1]['product_aggregation'].unique()
#plot_ridge(time_df, ['VEUVE CLICQUOT CHAMPAGNE', 'RUINART CHAMPAGNE', 'CHAMPAGNE CANARD DUCHENE CHAMPAGNE'])
plot_ridge(clustered_time_df[1], ['VEUVE CLICQUOT CHAMPAGNE'])
plot_ridge(clustered_time_df[1], ['RUINART CHAMPAGNE'])
plot_ridge(clustered_time_df[1], ['VEUVE CLICQUOT CHAMPAGNE', 'POMMERY CHAMPAGNE', 'BILLECART-SALMON CHAMPAGNE'])


# %% DUMP DATA !



with open('assets/clustered_by_date.pkl', 'wb') as file:
    pickle.dump(dct_by_date_clustered, file)

with open('assets/clustered_by_txn.pkl', 'wb') as file:
    pickle.dump(dct_by_txn_clustered, file)

with open('assets/clustered_basket_stats.pkl', 'wb') as file:
    pickle.dump(clustered_basket_stats, file)

with open('assets/clustered_purchase_times.pkl', 'wb') as file:
    pickle.dump(clustered_time_df, file)


# %%
