from pathlib import Path
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
        border: 1px solid #3a3a55;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .game-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.45);
        border-color: #ffd700;
    }

    .game-icon {
        font-size: 3.6rem;
        margin-bottom: 8px;
    }

    .game-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 14px 0 8px;
        color: #ffd700;
    }

    .game-desc {
        color: #bbb;
        font-size: 0.95rem;
        min-height: 48px;
        margin-bottom: 16px;
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
        "module": "tower_hanoi",  # ← Tên file không có .py
    },
    {
        "title": "Code Breaker",
        "icon": "🎨",
        "desc": "Đoán dãy màu bí mật trong số lượt giới hạn",
        "module": "guess_balls",  # ← Tên file không có .py
    },
    {
        "title": "Coming Soon",
        "icon": "🧩",
        "desc": "Game mới đang được phát triển",
        "module": None,
    },
]

cols = st.columns(3, gap="large")

for idx, game in enumerate(games):
    with cols[idx]:
        st.markdown(f"""
        <div class="game-card">
            <div>
                <div class="game-icon">{game['icon']}</div>
                <div class="game-title">{game['title']}</div>
                <div class="game-desc">{game['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if game["module"]:
            if st.button(
                label="▶ Chơi ngay",
                key=f"play_{idx}",
                use_container_width=True,
            ):
                # Lưu game được chọn vào session state
                st.session_state.selected_game = game["module"]
                st.rerun()
        else:
            st.button(
                label="Coming Soon",
                key=f"soon_{idx}",
                disabled=True,
                use_container_width=True,
            )

# ====================== LOAD GAME ======================
if "selected_game" in st.session_state:
    selected = st.session_state.selected_game

    # Nút back
    if st.button("← Quay lại menu", key="back_btn"):
        del st.session_state.selected_game
        st.rerun()

    st.divider()

    # Import và chạy game
    if selected == "tower_hanoi":
        st.markdown("## 🏛️ Tower of Hanoi")
        # Import code game
        from tower_hanoi import run as run_tower_hanoi
        run_tower_hanoi()

    elif selected == "guess_balls":
        st.markdown("## 🎨 Code Breaker")
        # Import code game
        from guess_balls import run as run_guess_balls
        run_guess_balls()