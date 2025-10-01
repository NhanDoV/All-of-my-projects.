import streamlit as st
from libs.data_preprocessing import *

st.set_page_config(page_title="Solar Balancing", layout="wide")
st.title("Solar Balancing")
home, advanced_analytic, sentiment_anal = st.tabs((
    "**HOME**", 
    "**ADVANCED ANALYTICS**", 
    "**Sentiment the Bitch-EVN**"
))
# store to a excel file named all_data.xlsx from this line; new-data named new_batch.xlsx to advoid memory
all_db = load_all_excels("data")

with home:
    with st.expander("Time-series form"):
        time_series_show(all_db)
    c1, _, c2 = st.columns([2, 0.1, 3])    
    with c1:
        with st.expander("Data Overview"):
            overview(all_db)
    with c2:
        with st.expander("Main charts"):
            summary(all_db)

with advanced_analytic:
    EDA(all_db)
    
with sentiment_anal:
    sentiment_analytic()