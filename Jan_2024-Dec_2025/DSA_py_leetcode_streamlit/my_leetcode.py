import matplotlib.pyplot as plt
import streamlit as st
from libs.lc_scripts import *
import random
import itertools

st.set_page_config(page_title = "MY LEETCODE", layout="wide")
python_tab, sql_tab = st.tabs((
    "**PYTHON**", 
    "**SQL**"
))

# ---------- Python scripts ----------
with python_tab:
    with st.expander("Check valid triangle"):
        c1, c2 = st.columns([4, 3])
        with c1:
            input_ls_arr = st.text_area("Input list of array here")

with sql_tab:
    with st.expander("Check valid triangle"):
        c1, c2 = st.columns([4, 3])
        with c1:
            input_ls_arr = st.text_area("Input list of array here")