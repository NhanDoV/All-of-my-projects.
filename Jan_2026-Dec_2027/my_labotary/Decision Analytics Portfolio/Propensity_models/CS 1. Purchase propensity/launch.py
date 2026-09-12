import streamlit as st
import pandas as pd
import seaborn as sns
from libs.ovv import *
from libs.eda import *
import matplotlib.pyplot as plt

df = pd.read_csv('data/training_sample.csv')
# df = df.drop(columns = ['Unnamed: 0', 'propensity'])
data_dim = str(df.shape)
n_null_cols = (df.isnull().sum() > 0).sum()
max_null_rows = df.isnull().sum().max()
n_dupl_row = df.duplicated().sum()

n_ordered = f"{df['ordered'].sum():,} \t ({df['ordered'].mean():.2f} %)"
add_to_basket = f"{(df['basket_add_list'] + df['basket_add_detail']).sum():,} \t ({(df['basket_add_list'] + df['basket_add_detail']).mean():.2f} %)"
n_icon_click = f"{df['basket_icon_click'].sum():,} \t ({df['basket_icon_click'].mean():.2f} %)"

viewed_checkout = f"{df['saw_checkout'].sum():,} \t ({df['saw_checkout'].mean():.2f} %)"
signed_in = f"{df['UserID'].nunique():,}"          # chỉ có số nguyên
wishlist_add = f"{df['detail_wishlist_add'].sum():,} \t ({df['detail_wishlist_add'].mean():.3f} %)"

st.set_page_config(page_title="Purchase Propensity", page_icon="🎯👤", layout="wide")

ovv, basket, wishlst_prdfun, cntxt, heatmap, pred = st.tabs(
    ["OVERVIEW", "BASKET ANALYTIC", "WISHLIST & PRODUCT FUNNEL", "OTHER CONTEXT", "HEATMAP SUMMARY", "TRAIN & PREDICT"]
)

with ovv:
    color1 = "#091f4b"   # header
    color2 = "#3c3cdd"   # hàng xen kẽ 1
    color3 = "#7b7bd9"   # hàng xen kẽ 2
    with st.expander("**:blue[TỔNG QUAN]**", expanded=True):
        c1, c2 = st.columns([8, 9], border=True)
        with c1:
            st.markdown(f"""<span style="color: {color3}; font-weight:bold; font-size:19px; text-align:center;"> METRIC </span>""", 
                        unsafe_allow_html=True)
            # Add data_dim as a detail at header Description; 
            get_metric_table(data_dim, n_null_cols, max_null_rows, n_dupl_row, 
                             n_ordered, n_icon_click, add_to_basket, viewed_checkout, signed_in, wishlist_add)

        with c2:
            st.markdown(f"""<span style="color:{color3}; font-weight:bold; font-size:19px; text-align:center;"> DATA DESCRIPTION </span>""", 
                        unsafe_allow_html=True)
            abstract_descr(color1, color2, color3)

        # Layer 2
        st.dataframe(df.head(2), hide_index=True)

    comment()

# ================================================
basket_df = df[['basket_icon_click', 'closed_minibasket_click', 'basket_add_list', 'basket_add_detail', 'ordered']]
with basket:
    c1, c2 = st.columns([5, 4], gap='small')
    with c1:
        basket_view(basket_df)
    with c2:
        basket_comment(basket_df)

# ================================================
wish_list_df = df[['detail_wishlist_add', 'list_size_dropdown', 'image_picker' , 'ordered']]
with wishlst_prdfun:
    c1, c2 = st.columns([3, 4], gap='small')
    with c1:
        with st.expander("WISHLIST REPORT", expanded=True):
            pass

# ================================================
with cntxt:
    with st.expander("", expanded=True):
        pass

# ================================================
df_corr = df.copy().drop(columns='UserID').corr()
with heatmap:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df_corr, ax=ax)
    st.pyplot(fig)