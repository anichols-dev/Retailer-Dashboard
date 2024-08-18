#%%
import pandas as pd

file = pd.read_csv('maindata.csv')

file = file.head(100000)

file.to_csv('shortdata.csv', index = False)