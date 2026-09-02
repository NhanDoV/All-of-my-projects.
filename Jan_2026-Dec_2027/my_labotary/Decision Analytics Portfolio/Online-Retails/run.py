import time
import pandas as pd
import streamlit as st

# For data quality and EDA functions
from libs.data_quality import *
from libs.EDA import *
from libs.RFM import *

# For loading CSS
from pathlib import Path

t0 = time.time()

def load_css():
    css = Path("assets/style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# =============================================================================
# LOADING DATASET
# =============================================================================
df = pd.read_csv('assets/OnlineRetail.csv', encoding='ISO-8859-1')

# Preprocessing: Convert 'InvoiceDate' to datetime and count duplicate rows
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract main info
n_dup_rows = df.duplicated().sum()
data_dim = df.shape
memory_usg = f"{(df.memory_usage(deep=True) / 1024**2).sum():.2f} (MB)"

# Remove duplicated data
df.drop_duplicates(inplace=True)

# =============================================================================
# PAGE CONFIGURATION 
# =============================================================================
st.set_page_config(layout="wide")
load_css()
st.title("ONLINE RETAIL")

ovv, basic_EDA, pareto, rfm, FE_n_SF = st.tabs(
    ["Data Understanding", "EDA", "Pareto Analysis", "RFM Analysis", "Feature_Engineering & Sale_Forecasting"]
)

# =============================================================================
# ovv: DATA OVERVIEW | QUALITY & BUZ-UNDERSTANDING
# =============================================================================
with ovv:
    data_overview_show(df)
    quality_and_metric_show(df, data_dim, memory_usg, n_dup_rows)
    intuition(df)

# =============================================================================
# basic_EDA: 
# =============================================================================

# Update a new column 'revenue' calculated as UnitPrice * Quantity
df['revenue'] = df['UnitPrice'] * df['Quantity']

# Extract date, month to aggregate revenue and other metrics by day
df['Date'] = df['InvoiceDate'].dt.date
df['Month'] = df['InvoiceDate'].dt.month

with basic_EDA:
    basic_EDA_show(df)

# =============================================================================
# pareto: PARETO ANALYSIS
# =============================================================================
pareto_df = get_pareto_df(df)

with pareto:
    pareto_analysis(pareto_df)
    customer_analytic(df)

# =============================================================================
# rfm: RFM ANALYSIS
# =============================================================================
rfm_df = get_RFM_table(df)

with rfm:
    rfm_show(rfm_df)
    customer_segmentation(rfm_df)

# =============================================================================
# FE_n_SF: FEATURE ENGINEERING & SALE FORECASTING
# =============================================================================
monthly_sales = get_monthly_trend(df)

with FE_n_SF:
    #feature_engineering(df)
    sale_forecasting(monthly_sales)