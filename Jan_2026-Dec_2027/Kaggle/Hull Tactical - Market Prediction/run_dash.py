import pandas as pd
import streamlit as st
from libs.EDA import *
import yfinance as yf

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

sp500 = yf.Ticker('^GSPC')
df_sp500 = sp500.history(period="max", start='1990-01-01')

def reconstruct_price(pct_changes, initial_price):
    prices = [initial_price]
    
    for pct_change in pct_changes:
        next_price = prices[-1] * (1 + pct_change)
        prices.append(next_price)
    
    return prices

prices = reconstruct_price(train_df['forward_returns'].values, 100)
train_df['price'] = prices[:-1]
train_df['SP500'] = df_sp500['Close'].values[:len(train_df)]

# ======= Title of the page =======
st.set_page_config(layout="wide")
st.title("📚 Hull Prediction")

basic_EDA, advanced_EDA, feature_engineering_and_predict = st.tabs(
    ["Basic EDA", "Advanced Analytic", "Feature Engineering & Predict"]
)

with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with basic_EDA:
    c1, _, c2 = st.columns([3, 0.1, 5])
    with c1:
        st.write("#### Description & reference")
        overview_description(train_df, test_df)

    with c2:
        st.write("#### 1. High-lighted metrics")
        get_metric_overviews(train_df, test_df)

        st.write("------------")
        st.write("#### 2. Data type distribution")
        get_dtype_distribution(train_df, test_df)

    st.write("------------")
    st.write("#### 3. Target analysis")
    get_main_target_analytic(train_df)

    st.write("------------")
    st.write("#### 4. Missing data analysis")
    get_missing_data_report(train_df, test_df)

    st.write("------------")
    st.write("#### 5. Correlative features analytic")
    with st.expander('Show', expanded = True):
        get_correlative_features_analysis(train_df, test_df)

with advanced_EDA:
    dummy_feat_report(train_df, test_df)

    st.write("------------")
    market_and_reference_report(train_df)

    st.write("------------")    
    c1, _, c2 = st.columns([2, 0.1, 3])
    with c1:
        feat_avl_report(train_df)
    with c2:
        volatity_report(train_df)

    st.write("------------")
    target_relationship_report(train_df)