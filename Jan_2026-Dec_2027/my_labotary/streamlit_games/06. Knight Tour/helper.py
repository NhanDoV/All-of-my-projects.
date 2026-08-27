import random
import streamlit as st

def game_descr():
    st.markdown("""
        <style>
        [data-testid="stExpander"] summary p {
            color: violet !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    with st.expander("Luật chơi", expanded=True):
        c1, c2 = st.columns([3, 2])    
        with c1:
            st.markdown('''
                            <div style="background:#E8F5E9;padding:12px;border-radius:10px;text-align:center;color:#2E7D32">
                                <b> Quân mã đi theo hình chữ L: <br> (x, y) -> {(x+2, y+1), (x-2, y+1), (x+1, y+2), (x+2, y+1), etc } </b>
                            </div>
                        ''', unsafe_allow_html=True)
            st.write(" ")
            st.markdown('''
                            <div style="background:#E8F5E9;padding:12px;border-radius:10px;text-align:center;color:#2E7D32">
                                <b> Ô có <span style="color: green;"> ♞ </span> là vị trí hiện tại của quân mã </b>
                            </div>
                        ''', unsafe_allow_html=True)
            st.write(" ")
            st.markdown('''
                            <div style="background:#E8F5E9;padding:12px;border-radius:10px;text-align:center;color:#2E7D32">
                                <b> Đi qua toàn bộ ô trên bàn cờ để chiến thắng. </b>
                            </div>
                        ''', unsafe_allow_html=True)
        with c2:
            st.write(" ")
            st.markdown('''
                            <div style="background:#E3F2FD;padding:12px;border-radius:10px;text-align:center;color:#1565C0">
                                <b> Không được đi lại vào ô đã đi qua </b>
                            </div>
                        ''', 
                    unsafe_allow_html=True)
            st.write(" ")
            st.markdown('''
                            <div style="background:#E3F2FD;padding:12px;border-radius:10px;text-align:center;color:#1565C0">
                                <b> Ô có 
                                        <span style="color: blue;"> 
                                            <span style="font-size:1em">♞<span style="color:red;margin-left:-0.7em">✕</span></span>                        
                                        </span> 
                                    là các ô đã đi qua </b>
                            </div>
                        ''', 
                    unsafe_allow_html=True)
            st.write(" ")
            st.markdown('''
                            <div style="background:#E3F2FD;padding:12px;border-radius:10px;text-align:center;color:#1565C0">
                                <b> Nút Back sẽ hủy nước đi gần nhất. </b>
                            </div>
                        ''', 
                    unsafe_allow_html=True)

# ===================================== GAME RULEs ==========================================
def start_new_game(board_size: int):
    start = (
        random.randint(0, board_size - 1),
        random.randint(0, board_size - 1),
    )

    st.session_state.board_size = board_size
    st.session_state.path = [start]
    st.session_state.message = (
        f"Bắt đầu tại hàng {start[0] + 1}, cột {start[1] + 1}."
    )
    st.session_state.won = False
    st.session_state.balloons_fired = False


def is_knight_move(source, target):
    dr = abs(source[0] - target[0])
    dc = abs(source[1] - target[1])
    return (dr, dc) in {(1, 2), (2, 1)}


def legal_moves(board_size, path):
    current = path[-1]
    visited = set(path)
    moves = []

    for row in range(board_size):
        for col in range(board_size):
            target = (row, col)

            if target not in visited and is_knight_move(current, target):
                moves.append(target)

    return moves

def board_css(board_size, path):
    visited = set(path)
    current = path[-1]
    available = set(legal_moves(board_size, path))

    styles = [
        """
        <style>
            .st-key-board [data-testid="stHorizontalBlock"] {
                gap: 2 !important;
            }

            .st-key-board [data-testid="stButton"] {
                width: 100% !important;
                margin: 0 !important;
            }

            .st-key-board [data-testid="stButton"] > button {
                width: 100% !important;
                aspect-ratio: 1 / 1 !important;
                min-height: 0 !important;
                padding: 0 !important;
                margin: 0 !important;
                border-radius: 0 !important;
                font-size: clamp(1.2rem, 3vw, 3rem) !important;
                font-weight: 700 !important;
                transition: 0.15s ease-in-out !important;
            }

            .st-key-board [data-testid="stButton"] > button:hover:not(:disabled) {
                filter: brightness(0.94);
                transform: scale(0.98);
            }

            .st-key-board [data-testid="stButton"] > button:disabled {
                opacity: 1 !important;
                cursor: default !important;
            }

            .st-key-board [data-testid="stButton"] > button * {
                color: inherit !important;
            }
        """
    ]

    for row in range(board_size):
        for col in range(board_size):
            cell = (row, col)
            key = f".st-key-cell_{row}_{col}"

            base_color = "#f8fbff" if (row + col) % 2 == 0 else "#b9dcf5"
            text_color = "#172033"
            border = "1px solid rgba(60, 80, 110, 0.18)"
            shadow = "none"

            if cell in visited and cell != current:
                text_color = "#d9485f"

            if cell in available:
                border = "3px solid #22a06b"
                shadow = "inset 0 0 0 1px rgba(255,255,255,0.65)"

            if cell == current:
                border = "4px solid #e58f00"
                shadow = "inset 0 0 0 2px #fff3cd"
                text_color = "#111827"

            styles.append(
                f"""
                {key} [data-testid="stButton"] > button {{
                    background-color: {base_color} !important;
                    color: {text_color} !important;
                    border: {border} !important;
                    box-shadow: {shadow} !important;
                }}
                """
            )

    styles.append("</style>")
    st.markdown("\n".join(styles), unsafe_allow_html=True)

def render_move_log(path):
    if len(path) <= 1:
        st.caption("Chưa có nước đi nào.")
        return

    n = len(path)
    n_col = n // 10 + 1          # <10 → 1 cột, 10-19 → 2 cột, 20-29 → 3 cột...

    # Tạo danh sách dòng
    lines = [f"**0.** From: `({path[0][0]+1}, {path[0][1]+1})`"]
    for i, (r, c) in enumerate(path[1:], 1):
        lines.append(f"**{i}.** Đi tới: `({r+1}, {c+1})`")

    if n_col == 1:
        # Ngắn thì in dọc bình thường
        st.markdown("  \n".join(lines))
    else:
        # Dài thì chia cột
        cols = st.columns(n_col)
        for idx, line in enumerate(lines):
            with cols[idx % n_col]:
                st.markdown(line)