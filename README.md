# The Amydga Retailer Dashboard 
#### Alex Nichols, Harris Lencz, Cole Clark 
#### Reach out to my email below with questions


## Overview

Mobile_Retail_Dashboard is a Python project designed to perform various computations on historical transaction data. The main script, `run_computation.py`, orchestrates several Python programs to compute insights and reports.
Assuming a MainData.csv with 2 million rows (75 MB), run time is roughly **25 minutes.**

I built this end-to-end application to expand the coverage of **Amydga**, an intelligence system for the Alcohol Industry. This dashboard helped Amygda add 350 retailers to their network (and counting).

## Run It!

### Inputs
_Store the 2 main data files in /Assets/Requirements:_
1. **Clustered Retailers**: CSV of retailerID and Cluster
2. **Main Data**: CSV with row as product, with columns for TransactionID, Product_of_Interest_Flag, etc

_Ensure the following programs pull from the appropriate file names:_
1. ```Cluster_Retailers.py```
2. ```Populate_Adjacents.py```
3. ```Basket_Insights.py```

### Computation

To run all computation:  ```python3 run_computation.py ```

### Dashboard

To run the streamlit application:   ```streamlit run Retail_Dashoard.py```



## Program Descriptions

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

### Optional Inputs 
##### Program will save data at this incremental step for future efficiency. Find quickStep boolean in Populate_Adjacents to toggle this step

For ```number of clusters``` files:
1. **Cluster {i} Aggregated by POI**: Path to the first optional input file.
2. **Main Data Aggregated by POI**: Path to the second optional input file.


### anichols26@amherst.edu


##### Final Comments
I learned so much on this project. Thank you Cole and Harris for your contributions.
