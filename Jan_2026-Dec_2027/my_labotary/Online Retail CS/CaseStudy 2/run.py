import streamlit as st
from libs.overview import *
from libs.dmf import *              # for demand forecasting
from libs.replenishment import *    # for replenishment optimization
import pandas as pd

## Loading dataset
df = pd.read_csv('data/replmn_data.csv')
# Extract data quality metrics
data_dim = df.shape
max_miss_rows = df.isnull().sum(axis=1).max()
n_miss_cols = df.isnull().sum(axis=0).sum()
n_dupl_rows = df.duplicated().sum()
total_order_qty = f"{df['order_qty_units'].sum():,}"
total_cost_usd = f"${df['total_cost_usd'].sum():,.0f}"
total_stockout = df['stockout'].sum()
df_null = pd.pivot_table(df.loc[df['region'].isnull()], index=['warehouse'], columns='policy', aggfunc='size' )

st.set_page_config(page_title="Replenishment recommendation", page_icon="🎯", layout="wide")

ovv, relp_opt, fore, po_eval, _ = st.tabs(
    ["OVERVIEW", "REPLENISHMENT OPTIMIZATION", "FORECASTING", "POLICY EVALUATION", ""]
)

with ovv:
    # Tầng 1
    with st.expander("**:red[DESCRIPTION]**", expanded=True):
        data_description()
        st.dataframe(df.head().set_index(['date', 'sku_id', 'warehouse']), width='stretch')

    # Tầng 2
    with st.expander("**:red[DATA QUALITY]**", expanded=True):
        data_quality(df_null, data_dim, max_miss_rows, n_miss_cols, n_dupl_rows, total_order_qty, total_cost_usd, total_stockout)
        st.write(" ")

with fore:
    demand_forecast_eda(df)

with relp_opt:
    repl_eda(df)
    repl_optimization(df)
