# README for Retailer Dashboard and Computation
#### Alex Nichols, Harris Lencz, Cole Clark
#### Reach out to my email below with questions


## Overview

Mobile_Retail_Dashboard is a Python project designed to perform various computations related to retailer data. The main script, `run_computation.py`, orchestrates the execution of several Python scripts to generate insights and reports.
Assuming a MainData.csv with 2 million rows, run time is roughly 25 minutes. 

## Features

- Cluster_Retailers.py: Clustering retailers (lightspeed)
- Populate_Adjacents.py: Populating adjacent data (~17 min)
- Neighborhood_Adjacents.py: Computing neighborhood adjacents (~1 min)
- Basket_Insights.py: Generating basket insights (~5 min)
- Market_Share.py: Calculating market share (lightspeed)

## Requirements

- Python 3.6 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Install the required packages (this will do automatically when running run_computation.py):
    ```sh pip install -r requirements.txt ```

### Required Inputs (Stored in Assets / Requirements)

1. **Clustered Retailers**: Path to a csv of retailerID and Cluster
2. **Main Data**: Path to CSV with row as product, with columns for TransactionID, Product_of_Interest_Flag, etc

### Optional Inputs (Stored in Assets / Requirements / Aggregations  ||  Program will save data at this incremental step for future efficiency. Find quickStep boolean in Populate_Adjacents to toggle this step)

For i in range(number of clusters):
1. **Cluster i Aggregated by POI**: Path to the first optional input file.
2. **Main Data Aggregated by POI**: Path to the second optional input file.


### Computation

```sh python run_computation_computation.py ```

### Dashboard

```streamlit run Retail_Dashoard.py```


### anichols26@amherst.edu


##### Final Comments
I learned so much on this project. Thank you Cole and Harris for your contributions.
