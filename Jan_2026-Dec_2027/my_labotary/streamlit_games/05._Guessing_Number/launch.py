import streamlit as st
import random
import dataclasses
from typing import List, Dict

st.set_page_config(page_title="Guessing a Number", page_icon="🕵️‍♂️🔢", layout="wide")

c1, c2 = st.columns([1, 2], gap='medium', border=True)

with c1:
    st.title(":green[🕵️‍♂️ Guessing a Number 🔢]")
    st.markdown("""
    <div style="line-height: 1.7; color: #e0e0e0;">

    #### <span style="color:#7CFC00;">Description</span>

    <span style="color:#87CEFA;">Trò chơi đoán số:</span>

    - Máy sẽ <span style="color:#7CFC00; font-weight:600;">ngẫu nhiên chọn một số</span> trong khoảng <span style="color:#00BFFF; font-weight:600;">1 → 1000</span>.
    - Bạn nhập số đoán vào ô input bên phải.
    - Sau mỗi lần đoán:
      - <span style="color:#FFD700;">Quá thấp</span> → hiện cảnh báo màu vàng
      - <span style="color:#00BFFF;">Quá cao</span> → hiện thông báo màu xanh
      - <span style="color:#7CFC00; font-weight:600;">Đúng</span> → bạn thắng và thấy số lần đoán
    - Bảng <span style="color:#87CEFA; font-weight:600;">Lịch sử các số đã đoán</span> sẽ lưu lại toàn bộ lần đoán của bạn trong ván hiện tại.
    - Nhấn nút <span style="color:#7CFC00; font-weight:600;">NEW GAME</span> để bắt đầu ván mới (số mới + xóa lịch sử).

    </div>
    """, unsafe_allow_html=True)

with c2:
    HI = 1000

    @dataclasses.dataclass
    class GameState:
        number: int
        num_guesses: int = 0
        game_number: int = 0
        game_over: bool = False
        history: List[Dict] = dataclasses.field(default_factory=list)

    if "state" not in st.session_state:
        st.session_state.state = GameState(random.randint(1, HI))

    state = st.session_state.state

    # ===== 2 CỘT: ĐOÁN | LỊCH SỬ =====
    guess_col, his_col = st.columns([1, 2], gap="medium")

    # ----- Cột trái: ô đoán (được căn giữa theo chiều dọc) -----
    with guess_col:
        n = len(state.history)

        # Công thức spacer: càng nhiều dòng lịch sử → càng đẩy ô đoán xuống
        # (xấp xỉ nửa chiều cao bảng)
        spacer = max(0, (n - 1) // 2)

        for _ in range(spacer):
            st.write("")          # dòng trống

        if not state.game_over:
            guess = st.text_input(
                f"Guess a number between 1 and {HI}",
                key=f"guess_{state.game_number}"
            )

            if guess:
                try:
                    guess_int = int(guess)
                    state.num_guesses += 1

                    if guess_int < state.number:
                        result = "Too low ⬇️"
                        st.warning(f"{guess_int} is too low")
                    elif guess_int > state.number:
                        result = "Too high ⬆️"
                        st.info(f"{guess_int} is too high")
                    else:
                        result = "Correct ✅"
                        st.success(f"You win! It only took you {state.num_guesses} tries")
                        state.game_over = True

                    state.history.append({
                        "Lần đoán": state.num_guesses,
                        "Số bạn đoán": guess_int,
                        "Kết quả": result
                    })

                except ValueError:
                    st.error("Please guess a *number*")
        else:
            st.success(
                f"Game over! The number was **{state.number}**. "
                f"You guessed in **{state.num_guesses}** tries."
            )

    # ----- Cột phải: bảng lịch sử -----
    with his_col:
        st.markdown(
            """### <span style="color:cyan">📜 Lịch sử các số đã đoán</span>""",
            unsafe_allow_html=True
        )

        if state.history:
            if len(state.history) > 8:
                left, right = st.columns(2)
                with left:
                    st.caption("8 nước đầu tiên")
                    st.dataframe(
                        state.history[:8],
                        width='stretch',
                        hide_index=True
                    )
                with right:
                    st.caption("Các nước còn lại")
                    st.dataframe(
                        state.history[8:],
                        width='stretch',
                        hide_index=True
                    )
            else:
                st.dataframe(
                    state.history,
                    width='stretch',
                    hide_index=True
                )
        else:
            st.caption("Chưa có nước đi nào. Hãy bắt đầu đoán!")

    # ===== NÚT NEW GAME (dưới cùng, chính giữa) =====
    st.markdown("<br>", unsafe_allow_html=True)

    _, col_center, _ = st.columns([1, 1, 1])
    with col_center:
        if st.button("NEW GAME", type="primary", use_container_width=True):
            state.number = random.randint(1, HI)
            state.num_guesses = 0
            state.game_number += 1
            state.game_over = False
            state.history = []
            st.rerun()