import streamlit as st
import random

# ============================================
# GAME LOGIC
# ============================================
def create_board(rows, cols, num_mines):
    """Tạo lưới game và đặt mìn ngẫu nhiên"""
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    mines = set()

    while len(mines) < num_mines:
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
        if (r, c) not in mines:
            mines.add((r, c))

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for r in range(rows):
        for c in range(cols):
            if (r, c) in mines:
                board[r][c] = -1
            else:
                count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in mines:
                        count += 1
                board[r][c] = count

    return board, mines


def reveal_cell(board, revealed, r, c, rows, cols):
    """Mở ô (bao gồm flood fill nếu ô trống)"""
    if revealed[r][c]:
        return

    revealed[r][c] = True

    if board[r][c] == 0:
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not revealed[nr][nc]:
                reveal_cell(board, revealed, nr, nc, rows, cols)


def check_win(revealed, mines, rows, cols):
    """Kiểm tra thắng: tất cả ô không phải mìn đều đã mở"""
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in mines and not revealed[r][c]:
                return False
    return True


# ============================================
# STREAMLIT APP
# ============================================
st.set_page_config(page_title="💣 Minesweeper", page_icon="💣", layout="wide")

CELL_UNREVEALED_BG   = "#180764"   # ô chưa mở
CELL_REVEALED_BG     = "#1333B3"   # ô đã mở
CELL_HOVER_BG        = "#5d4898"   # hover
CELL_BORDER          = "#242274"
CELL_BORDER_REVEALED = "#786ae6"
CELL_TEXT            = "#efe9e9"

METRIC_BG            = "#2a1a5e"   # nền metric (tím đậm vừa)
METRIC_BORDER        = "#786ae6"   # viền metric

st.markdown(f"""
    <style>
        div[data-testid="stHorizontalBlock"] button {{
            height: 42px !important;
            min-height: 42px !important;
            width: 100% !important;
            border-radius: 4px !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            border: 2px solid {CELL_BORDER} !important;
            background-color: {CELL_UNREVEALED_BG} !important;
            color: {CELL_TEXT} !important;
            box-shadow: inset 1px 1px 0 #ffffff, inset -1px -1px 0 #6e6e6e !important;
            transition: all 0.1s ease;
        }}
        div[data-testid="stHorizontalBlock"] button:hover:enabled {{
            background-color: {CELL_HOVER_BG} !important;
            border-color: #555 !important;
        }}
        div[data-testid="stHorizontalBlock"] button:disabled {{
            background-color: {CELL_REVEALED_BG} !important;
            border: 1px solid {CELL_BORDER_REVEALED} !important;
            box-shadow: none !important;
            color: {CELL_TEXT} !important;
            opacity: 1 !important;
        }}
        /* ===== METRIC ===== */
        div[data-testid="stMetric"] {{
            background-color: {METRIC_BG} !important;
            border: 1px solid {METRIC_BORDER} !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            text-align: center !important;                 /* căn giữa toàn bộ */
        }}
        div[data-testid="stMetric"] label {{
            color: {CELL_TEXT} !important;
            font-size: 29px!important;
            display: block !important;
            text-align: center !important;                 /* căn giữa label */
            width: 100% !important;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: {CELL_TEXT} !important;
            font-size: 39px!important;
        }}
        /* ===== SELECTBOX ===== */
        div[data-baseweb="select"] {{
            width: 100% !important;
        }}

        div[data-baseweb="select"] > div {{
            min-height: 38px !important;
            height: 38px !important;
            background-color: #1e1460 !important;
            border: 1px solid #786ae6 !important;
            border-radius: 8px !important;
        }}

        /* Selected value */
        div[data-baseweb="select"] [data-testid="stMarkdownContainer"],
        div[data-baseweb="select"] span {{
            color: #efe9e9 !important;
        }}

        /* Dropdown arrow */
        div[data-baseweb="select"] svg {{
            fill: #efe9e9 !important;
            width: 18px !important;
            height: 18px !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ---------- Layout ----------
config_col, play_col = st.columns([3, 5], gap="large")

with config_col:
    st.title("💣 Minesweeper")
    with st.container(border=True):
        st.markdown("##### ⚙️ Cài đặt bàn chơi")
        crow, ccol, cmine = st.columns(3)
        with crow:
            ROWS = st.number_input(
                "Số hàng",
                min_value=6, max_value=12,
                value=6,
                label_visibility="collapsed",
                help="Chọn số hàng"
            )
            st.caption("Hàng (rows)")
        with ccol:
            COLS = st.number_input(
                "Số cột",
                min_value=8, max_value=16,
                value=10,
                label_visibility="collapsed",
                help="Chọn số cột"
            )
            st.caption("Cột (cols)")
        with cmine:
            NUM_MINES = st.number_input(
                "Số mìn",
                min_value=10, max_value=20,
                value=12,
                label_visibility="collapsed",
                help="Chọn số mìn"
            )
            st.caption("Mìn")

    st.markdown("---")
    st.markdown("""
    **Cách chơi:**
    - Bật **Chế độ cắm cờ** rồi click để đặt / gỡ 🚩
    - Tắt chế độ cắm cờ rồi click để mở ô
    - Mở hết ô không có mìn để thắng!
    """)

    # Flag mode
    flag_mode = st.checkbox("🚩 Chế độ cắm cờ (Flag mode)", value=False)

    # ---------- Khởi tạo / reset board khi thay đổi size ----------
    need_new_board = (
        "board" not in st.session_state
        or st.session_state.get("rows") != ROWS
        or st.session_state.get("cols") != COLS
        or st.session_state.get("num_mines") != NUM_MINES
    )

    if need_new_board:
        st.session_state.board, st.session_state.mines = create_board(ROWS, COLS, NUM_MINES)
        st.session_state.revealed = [[False] * COLS for _ in range(ROWS)]
        st.session_state.flagged = [[False] * COLS for _ in range(ROWS)]
        st.session_state.game_over = False
        st.session_state.won = False
        st.session_state.first_click = True
        st.session_state.rows = ROWS
        st.session_state.cols = COLS
        st.session_state.num_mines = NUM_MINES

    # Số mìn còn lại (an toàn vì đã init ở trên)
    flags_count = sum(sum(row) for row in st.session_state.flagged)
    mines_left = NUM_MINES - flags_count
    st.markdown("---")
    c1, c2 = st.columns([2, 1], gap='large')
    with c1:
        st.metric("💣 Số mìn còn lại", f"\t {mines_left} ", border=True)
    with c2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        if st.button("🔄 Chơi lại", use_container_width=True):
            st.session_state.board, st.session_state.mines = create_board(ROWS, COLS, NUM_MINES)
            st.session_state.revealed = [[False] * COLS for _ in range(ROWS)]
            st.session_state.flagged = [[False] * COLS for _ in range(ROWS)]
            st.session_state.game_over = False
            st.session_state.won = False
            st.session_state.first_click = True
            st.rerun()

with play_col:
    with st.expander("**:blue[GAME PLAY]**", expanded=True):
        # Trạng thái game
        if st.session_state.game_over:
            if st.session_state.won:
                st.success("🎉 CHÚC MỪNG! BẠN THẮNG RỒI! 🎉")
            else:
                st.error("💥 BÙM! BẠN THUA RỒI! 💥")

        # Render lưới
        for r in range(ROWS):
            cols = st.columns(COLS)
            for c in range(COLS):
                with cols[c]:
                    # Label
                    if st.session_state.revealed[r][c]:
                        val = st.session_state.board[r][c]
                        if val == -1:
                            label = "💣"
                        elif val == 0:
                            label = " "
                        else:
                            label = str(val)
                    elif st.session_state.flagged[r][c]:
                        label = "🚩"
                    else:
                        label = " "

                    disabled = st.session_state.game_over or st.session_state.revealed[r][c]

                    if st.button(label, key=f"cell_{r}_{c}", disabled=disabled, use_container_width=True):
                        if st.session_state.game_over:
                            st.stop()

                        # ----- FLAG MODE -----
                        if flag_mode:
                            if not st.session_state.revealed[r][c]:
                                st.session_state.flagged[r][c] = not st.session_state.flagged[r][c]
                            st.rerun()

                        # ----- OPEN MODE -----
                        else:
                            # Không mở ô đã flag
                            if st.session_state.flagged[r][c]:
                                st.stop()

                            # First click an toàn
                            if st.session_state.first_click:
                                st.session_state.first_click = False
                                while st.session_state.board[r][c] == -1:
                                    st.session_state.board, st.session_state.mines = create_board(ROWS, COLS, NUM_MINES)

                            # Mở ô
                            if st.session_state.board[r][c] == -1:
                                # Trúng mìn
                                st.session_state.game_over = True
                                st.session_state.won = False
                                for mr, mc in st.session_state.mines:
                                    st.session_state.revealed[mr][mc] = True
                            else:
                                reveal_cell(
                                    st.session_state.board,
                                    st.session_state.revealed,
                                    r, c, ROWS, COLS
                                )
                                if check_win(st.session_state.revealed, st.session_state.mines, ROWS, COLS):
                                    st.session_state.game_over = True
                                    st.session_state.won = True

                            st.rerun()
