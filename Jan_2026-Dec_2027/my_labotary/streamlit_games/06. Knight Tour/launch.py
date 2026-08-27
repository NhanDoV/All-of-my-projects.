import streamlit as st
from helper import *

st.set_page_config(page_title="Knight-Tour", page_icon="🐴", layout="wide",)

MIN_BOARD_SIZE = 5
MAX_BOARD_SIZE = 8

if "board_size" not in st.session_state:
    start_new_game(5)

c1, c2 = st.columns([4, 5], gap="large")

with c1:
    st.title(":yellow[♞ Knight Tour]")
    selected_size = st.number_input(
        "Chọn kích thước bàn cờ",
        min_value=MIN_BOARD_SIZE,
        max_value=MAX_BOARD_SIZE,
        value=st.session_state.board_size,
        step=1,
    )

    if selected_size != st.session_state.board_size:
        start_new_game(selected_size)
        st.rerun()

    col_back, col_reset = st.columns(2)

    with col_back:
        if st.button(
            "↩ Back",
            use_container_width=True,
            disabled=len(st.session_state.path) <= 1,
        ):
            st.session_state.path.pop()
            st.session_state.won = False
            st.session_state.balloons_fired = False
            st.session_state.message = "Đã quay lại nước đi trước."
            st.rerun()

    with col_reset:
        if st.button("↻ New game", use_container_width=True):
            start_new_game(st.session_state.board_size)
            st.rerun()

    st.divider()

    total_cells = st.session_state.board_size ** 2
    move_count = len(st.session_state.path) - 1
    
    _, c_stats, _, c_logs, _ = st.columns([1, 2, 2, 2, 1])
    with c_stats:
        st.metric("Số nước đã đi", move_count)
    with c_logs:
        st.metric("Ô đã thăm", f"{len(st.session_state.path)} / {total_cells}")

    if st.session_state.won:
        st.success("Chúc mừng! Bạn đã hoàn thành Knight Tour. 🎉")

        if not st.session_state.balloons_fired:
            st.balloons()
            st.session_state.balloons_fired = True

    else:
        st.info(st.session_state.message)

        if len(st.session_state.path) < total_cells:
            moves_left = legal_moves(
                st.session_state.board_size,
                st.session_state.path,
            )

            if not moves_left:
                st.warning(
                    "Mã không còn nước đi hợp lệ. "
                    "Hãy dùng Back để thử hướng khác."
                )

    game_descr()

with c2:
    board_size = st.session_state.board_size
    path = st.session_state.path
    current = path[-1]
    visited = set(path)

    board_css(board_size, path)

    st.markdown(
        f"<span style='color: #89CFF0; font-weight: 600; font-size:29px'>"
        f"Chessboard {board_size} x {board_size}"
        f"</span>",
        unsafe_allow_html=True,
    )

    with st.container(key="board"):
        for row in range(board_size):
            columns = st.columns(board_size, gap="small")

            for col in range(board_size):
                cell = (row, col)

                if cell == current:
                    label = ":green[♞]"
                elif cell in visited:
                    label = ''':blue[♞❌]'''
                else:
                    label = " "

                with columns[col]:
                    with st.container(key=f"cell_{row}_{col}"):
                        clicked = st.button(
                            label,
                            key=f"move_{row}_{col}",
                            help=f"Hàng {row + 1}, cột {col + 1}",
                            use_container_width=True,
                            disabled=st.session_state.won
                        )

                        if clicked:
                            if cell in visited:
                                st.session_state.message = (
                                    "Ô này đã được đi qua. "
                                    "Hãy chọn một ô chưa thăm."
                                )

                            elif is_knight_move(current, cell):
                                st.session_state.path.append(cell)

                                total_cells = board_size ** 2
                                move_count = len(st.session_state.path) - 1

                                st.session_state.message = (
                                    f"Nước đi {move_count}: "
                                    f"hàng {cell[0] + 1}, cột {cell[1] + 1}."
                                )

                                if len(st.session_state.path) == total_cells:
                                    st.session_state.won = True

                            else:
                                st.session_state.message = (
                                    "Nước đi không hợp lệ. "
                                    "Mã chỉ được đi theo hình chữ L."
                                )

                            st.rerun()

    st.divider()
    st.markdown(
        "<span style='color: #B0E0E6; font-size: 23px;'> Lịch sử nước đi </span>",
        unsafe_allow_html=True,
    )

    with st.container():
        render_move_log(st.session_state.path)