import streamlit as st
import string
from helper import *
from game_states import *
from gen_solutions import *
import pandas as pd

# =================================== CONFIG =======================================
st.set_page_config(page_title="N-Queens Puzzle", page_icon="♛", layout="wide",)
play, sol = st.tabs(["PLAY", "VIEW ALL SOLUTIONS"])

with play:
    # ================================ INITIAL LOAD ====================================
    if "n" not in st.session_state:
        initialize_game(n=4)

    # ============================ SIDEBAR / TOP CONTROLS ===============================
    st.title("♛ N-Queens Puzzle")
    st.caption("Đặt đúng N queens sao cho không có hai quân nào cùng hàng, cột hoặc đường chéo.")

    selected_n = st.selectbox("Chọn kích thước bàn cờ", 
                            options=list(range(4, 9)), 
                            index=st.session_state.n - 4)
    if selected_n != st.session_state.n:
        initialize_game(selected_n)

    # ================================ MAIN LAYOUT ========================================
    left_col, right_col = st.columns([2, 5], gap="large")

    n = st.session_state.n
    queens = st.session_state.queens
    conflict_queens = get_conflict_queens(queens)
    attacked_cells = get_attacked_cells(queens, n)
    hint_cell = st.session_state.hint_cell

    queens_as_labels = sorted(
        [pos_to_label(pos) for pos in queens],
        key=lambda x: (x[0], int(x[1:]))
    )

    with left_col:
        st.write("##### Trạng thái")

        if st.session_state.status == "won":
            st.success("🎉 PASS — Bạn đã giải thành công!")
        elif conflict_queens:
            st.error(f"⚠️ Có {len(conflict_queens)} queen đang xung đột.")
        elif len(queens) == n:
            st.warning("Đã đủ queens nhưng vẫn còn xung đột.")
        else:
            st.info("Đang chơi...")

        # Row info
        lc1, lc2 = st.columns(2)
        with lc1:
            st.metric(label="Queens đã đặt", value=f"{len(queens)} / {n}")
        with lc2:
            st.metric(label="Số nước thao tác", value=len(st.session_state.history))

        # Log rows
        st.write("##### Tọa độ queens")

        if queens_as_labels:
            st.code(", ".join(queens_as_labels), language=None)
        else:
            st.caption("Chưa đặt queen nào.")

        # Hint rows
        ll, lr = st.columns([3, 2])
        with ll:
            st.subheader("Hướng dẫn icon")
            st.markdown("♛ : Queen hợp lệ")
            st.markdown("♛⚠️ : Queen đang conflict")
            st.markdown("💡 : Gợi ý nước đi")
            st.markdown("· : Ô đang bị queen tấn công")

        with lr:
            st.button(
                "↩️ Undo",
                on_click=undo_last_move,
                disabled=len(st.session_state.history) == 0,
                use_container_width=True,
            )

            st.button(
                "💡 Hint",
                on_click=show_hint,
                disabled=st.session_state.status == "won",
                use_container_width=True,
            )

            st.button(
                "✅ Show solution",
                on_click=load_solution,
                use_container_width=True,
            )

            if st.button("🔄 Reset board", use_container_width=True):
                initialize_game(st.session_state.n)
                st.rerun()


    # =========================================================
    # BOARD
    # =========================================================
    with right_col:
        st.subheader(f"Bàn cờ {n} × {n}")

        st.caption(
            "Click ô trống để đặt queen. Click vào queen để gỡ queen. "
            "Các ô có hậu bị xung đột sẽ hiện cảnh báo."
        )

        # Header: Y\X | 1 | 2 | ... | n
        header_cols = st.columns(n + 1, gap="small")

        header_cols[0].markdown(
            "<div style='text-align:center; font-weight:bold;'>Y \\ X</div>",
            unsafe_allow_html=True,
        )

        for col in range(n):
            header_cols[col + 1].markdown(
                f"<div style='text-align:center; font-weight:bold;'>{col + 1}</div>",
                unsafe_allow_html=True,
            )

        # Board rows: a -> h
        for row in range(n):
            row_cols = st.columns(n + 1, gap="medium")
            st.markdown("""
                <style>
                div[data-testid="column"] button {
                    width: 50px !important;
                    height: 50px !important;
                    min-width: 50px !important;
                    max-width: 50px !important;
                    text-align: center !important;
                    padding: 0 !important;
                }
                </style>
            """, unsafe_allow_html=True)
            row_label = string.ascii_lowercase[row]

            row_cols[0].markdown(
                f"""
                    <div style='text-align:center; font-weight:bold;'>
                        {row_label}
                    </div>""",
                unsafe_allow_html=True,
            )

            for col in range(n):
                position = (row, col)
                cell_label = pos_to_label(position)

                is_queen = position in queens
                is_conflict = position in conflict_queens
                is_hint = position == hint_cell
                is_attacked = position in attacked_cells

                # Nội dung icon hiển thị trong button
                if is_conflict:
                    icon = "♛⚠️"
                elif is_queen:
                    icon = "♛"
                elif is_hint:
                    icon = "💡"
                elif is_attacked:
                    icon = "·"
                else:
                    icon = " "

                # Queen hợp lệ/hint dùng primary để dễ quan sát.
                # Queen conflict vẫn là secondary nhưng có warning icon.
                button_type = "primary" if (is_queen and not is_conflict) or is_hint else "secondary"                
                row_cols[col + 1].button(
                    label=icon,
                    key=f"cell_{cell_label}",
                    help=f"Ô {cell_label}",
                    type=button_type,
                    use_container_width=True,
                    on_click=toggle_queen,
                    args=(position,),
                )

        st.divider()

        if st.session_state.status == "won":
            st.balloons()
            st.success(
                f"Hoàn thành N-Queens với N = {n}. "
                f"Tổng số thao tác: {len(st.session_state.history)}."
            )
        elif conflict_queens:
            conflict_labels = sorted(
                [pos_to_label(pos) for pos in conflict_queens],
                key=lambda x: (x[0], int(x[1:])),
            )
            st.warning(
                "Queens conflict: " + ", ".join(conflict_labels)
            )
        elif len(queens) == n:
            st.info("Bạn đã có đủ queens. Hệ thống đang kiểm tra xung đột...")

with sol:
    sol = AllSolution()
    ans = [sol.solveNQueens(n) for n in range(1, 9)]
    n_res = [len(res) for res in ans]
    df = pd.DataFrame({
        'n_queens': list(range(1, 9)),
        'n.solutions': n_res
    })

    with st.expander("**:violet[DEFINITION - RULE PLAY]**", expanded=True):
        rule_play()

    with st.expander("**:violet[Results overview]**", expanded=True):
        c1, c2 = st.columns([1, 6], border=True)
        with c1:
            st.table(df.set_index('n_queens'))

        with c2:
            st.subheader("**:blue[Display number of possible results]**")
            left, right = st.columns(2, border=True)
            with left:
                n_queens = st.selectbox("Select number of queens", [str(n) for n in range(1, 9)])
                n_queens = int(n_queens)
                results = ans[n_queens - 1]
                at_most = len(results)

            if at_most > 1:
                with right:
                    nums = st.number_input("Select number of possible results", 
                                            min_value=2, max_value=min(4, at_most), 
                                            help="At most 4 to optimize display view.mode")
                # view mode func here
                view_mode(nums, results)
            else:
                with right:
                    st.warning(f"When n_queens = {n_queens}, we DO NOT HAVE ANY POSSIBLE SOLUTION FOR THIS PUZZLE")