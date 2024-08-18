#%%
import pandas as pd

mdf = pd.read_csv('assets/maindata.csv')

shortdf = mdf.sort_values(by="transaction_datetime")
shortdf = shortdf.head(150000)

mdf.to_csv('assets/requirements/shortdata.csv')
# %%
