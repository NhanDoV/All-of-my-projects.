import streamlit as st
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def homepage_render_wrt_background(bg_home):
    st.markdown(f"""
        <style>
        .home {{
            background-image: url("data:image/jpg;base64,{bg_home}");
            background-size: cover;
            background-position: center;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }}
        </style>
        <div class="home">
            <h1>Funny Simulation Playground 🎲</h1>
            <p>This is a small app to play with mathematical visualizations, fun graphs, and simulations.</p>
            <ul>
                <li><b>Funny graphs</b>: draw shapes from math equations (heart, spiral, batman logo…)</li>
                <li><b>Funny Sequences</b>: explore special number sequences.</li>
                <li><b>Real-life simulations</b>: simulate games, physics, and social dynamics.</li>
            </ul>
            <p><i>Now, feel free to jump in and start your journey through this playground.</i></p>
        </div>
    """, unsafe_allow_html=True)