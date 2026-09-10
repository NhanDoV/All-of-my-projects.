import streamlit as st

st.set_page_config(
    page_title="GAME WITH NHAN",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== CSS ======================
st.markdown("""
<style>
    /* Ẩn mục launch */
    [data-testid="stSidebarNav"] ul li:first-child {
        display: none !important;
    }

    /* Thu hẹp sidebar */
    section[data-testid="stSidebar"] {
        width: 210px !important;
        min-width: 210px !important;
        max-width: 210px !important;
    }

    /* Card game */
    .game-card {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        transition: all 0.25s ease;
        border: 1px solid #3a3a55;
        height: 100%;
    }
    .game-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.45);
        border-color: #ffd700;
    }
    .game-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 14px 0 8px;
        color: #ffd700;
    }
</style>
""", unsafe_allow_html=True)

# ====================== GIAO DIỆN ======================
st.title("🎮 GAME WITH NHAN")
st.markdown("### Chọn game bạn muốn chơi")

games = [
    {
        "title": "Tower of Hanoi",
        "icon": "🏛️",
        "desc": "Cổ điển - Di chuyển tháp đĩa sang cột đích",
        "page": "pages/1__TowerHanoi__.py",
    },
    {
        "title": "Code Breaker",
        "icon": "🎨",
        "desc": "Đoán dãy màu bí mật trong số lượt giới hạn",
        "page": "pages/2__GuessBalls__.py",
    },
    {
        "title": "Coming Soon",
        "icon": "🧩",
        "desc": "Game mới đang được phát triển",
        "page": None,
    },
]

cols = st.columns(3, gap="large")

for idx, game in enumerate(games):
    with cols[idx]:
        st.markdown(f"""
        <div class="game-card">
            <div style="font-size: 3.6rem;">{game['icon']}</div>
            <div class="game-title">{game['title']}</div>
            <p style="color:#bbb; font-size:0.95rem; min-height:48px;">{game['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        if game["page"]:
            if st.button("▶ Chơi ngay", key=f"play_{idx}", use_container_width=True):
                st.switch_page(game["page"])
        else:
            st.button("Coming Soon", key=f"soon_{idx}", disabled=True, use_container_width=True)