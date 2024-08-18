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
pd.set_option('display.max_rows', 20)

clusters = pd.read_csv('assets/requirements/clustered_retailers.csv')
clusters = clusters.rename(columns={'RETAILER_ID': 'retailer_id'})

mfile = 'assets/requirements/shortdata.csv'

masta = pd.read_csv(mfile)

# Rename Columns  –  OLD :  NEW
masta = masta.rename(columns={'item_count':'unit_sales'}) 
masta.to_csv(mfile, index=False)

number_of_clusters = sorted(list(clusters['labels_from_model'].unique()))

merged = pd.merge(masta, clusters, on="retailer_id", how='left')
merged = merged.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y', 'category_hh_idx', 'log_txn_count', 'log_itemcount', 'log_dollar_sales', 'log_cat_hh', 'norm_log_txn_count', 'norm_log_itemcount', 'norm_log_dollar_sales', 'norm_log_cat_hh', 'tsne_1', 'tsne_2', 'tsne_3'])
# %%
clustered_dfs = []
for i in number_of_clusters:
    temp = merged[merged['labels_from_model']==i]
    temp.to_csv(f'assets/Cluster_{i}_Data.csv', index=False)
    clustered_dfs.append(temp)
# %%
