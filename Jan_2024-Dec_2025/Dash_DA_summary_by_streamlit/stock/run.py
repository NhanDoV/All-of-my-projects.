import streamlit as st
from helper.data_loader import load_data

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Bitcoin Analytics",
    layout="wide",
)

# ======================================================
# LOAD GLOBAL CSS
# ======================================================
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ======================================================
# LOAD DATA (GLOBAL CACHE)
# ======================================================
# NOTE:
# - Data is loaded once
# - Pages (home / predict) can call load_data again
#   but Streamlit cache will prevent reloading

df = load_data("btc_data.csv")

# ======================================================
# SIDEBAR (APP-LEVEL ONLY)
# ======================================================
st.sidebar.title("₿ Bitcoin")
st.sidebar.markdown("Bitcoin market analysis playground ☀️")

st.sidebar.divider()

st.sidebar.markdown("### Pages")
st.sidebar.markdown("- 🏠 **HOME**")
st.sidebar.markdown("- 🔮 **PREDICT**")

st.sidebar.divider()

st.sidebar.caption(
    "This app provides high-level market insights.\n"
    "Not financial advice."
)