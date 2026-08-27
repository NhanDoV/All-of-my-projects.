import streamlit as st
import random
import dataclasses
from typing import List, Dict

st.set_page_config(page_title="Guessing a Number", page_icon="🕵️‍♂️🔢", layout="wide")

c1, c2 = st.columns([2, 3])

with c1:
    st.title("🕵️‍♂️ Guessing a Number 🔢")
    st.markdown("""
    #### Description

    Trò chơi đoán số:

    - Máy sẽ **ngẫu nhiên chọn một số** trong khoảng **1 → 1000**.
    - Bạn nhập số đoán vào ô input bên phải.
    - Sau mỗi lần đoán:
      - **Quá thấp** → hiện cảnh báo màu vàng
      - **Quá cao** → hiện thông báo màu xanh
      - **Đúng** → bạn thắng và thấy số lần đoán
    - Bảng **Lịch sử các nước đi** sẽ lưu lại toàn bộ lần đoán của bạn trong ván hiện tại.
    - Nhấn nút **NEW GAME** để bắt đầu ván mới (số mới + xóa lịch sử).
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

    # Khởi tạo state nếu chưa có
    if "state" not in st.session_state:
        st.session_state.state = GameState(random.randint(1, HI))

    state = st.session_state.state

    if st.button("NEW GAME", type="primary"):
        state.number = random.randint(1, HI)
        state.num_guesses = 0
        state.game_number += 1
        state.game_over = False
        state.history = []          # xóa lịch sử khi chơi ván mới

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

                # Lưu vào lịch sử
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

    # ===== Bảng lịch sử các nước đã đi =====
    st.markdown("### 📜 Lịch sử các nước đã đi")

    if state.history:
        if len(state.history) > 8:
            left, right = st.columns(2)

            with left:
                st.caption("8 nước đầu tiên")
                st.dataframe(
                    state.history[:8],
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                st.caption("Các nước còn lại")
                st.dataframe(
                    state.history[8:],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.dataframe(
                state.history,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.caption("Chưa có nước đi nào. Hãy bắt đầu đoán!")