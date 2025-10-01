import streamlit as st
from libs.home import *

st.set_page_config(page_title="Funny simulation", layout="wide")
home, graphs, sequences, simulations = st.tabs((
    "**HOME**", 
    "**Funny graphs (cts equation)**", 
    "**Funny-Serial/Sequence**", 
    "**Real-life simulations**"
))
# ---------- HOME ----------
with home:
    bg_home = get_base64_of_bin_file("assets/bg.jpg")
    homepage_render_wrt_background(bg_home)
# ---------- GRAPHS ----------
with graphs:
    from libs.graphs_eq import *
    bg_graphs = get_base64_of_bin_file("assets/tab1.jpg")
    graph_render_wrt_background(bg_graphs)    
    run()
# ---------- SEQUENCES ----------
with sequences:
    from libs.numeric import *
    bg_sequences = get_base64_of_bin_file("assets/tab2.jpg")
    nums_render_wrt_background(bg_sequences)
    run()
# ---------- SIMULATIONS ----------
with simulations:
    from libs.realife_sims import *
    bg_sims = get_base64_of_bin_file("assets/tab3.jpg")
    realifesims_render_wrt_background(bg_sims)
    run()