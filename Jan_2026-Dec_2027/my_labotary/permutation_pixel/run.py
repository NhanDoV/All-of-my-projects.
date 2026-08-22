import streamlit as st
import cv2
import numpy as np
import random

from helper import generate_points, pairwise_perm_blocks

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Image Block Permutation",
    layout="wide"
)

# =========================
# CSS FORM
# =========================
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# =========================
# TITLE
# =========================
st.title("🧩 Image Block Permutation")

# =========================
# SIDEBAR CONFIG
# =========================
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

n_max = st.sidebar.slider("Max Areas", 10, 500, 200)
block_size = st.sidebar.slider("Block Size (d)", 20, 200, 80)

run_btn = st.sidebar.button("🚀 Run Permutation")

# =========================
# MAIN LOGIC
# =========================
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    if run_btn:
        n_areas = random.randint(n_max // 2, n_max)

        points = generate_points(w, h, n_areas, block_size)
        result = pairwise_perm_blocks(img_rgb, points, block_size)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(img_rgb, use_container_width=True)

        with col2:
            st.subheader("Permuted")
            st.image(result, use_container_width=True)

    else:
        st.info("👉 Click **Run Permutation** to start")

else:
    st.warning("👉 Please upload an image")