import random
from functools import lru_cache
import streamlit as st

st.set_page_config(page_title="Stone Game I", page_icon="🪨", layout="wide")

def computer_best_move(nums):
    """Trả về 'left' hoặc 'right' theo minimax tối ưu."""

    # Nếu chỉ còn 1 viên đá thì lấy luôn bên trái (hoặc phải đều như nhau)
    if len(nums) == 1:
        return "left"

    @lru_cache(maxsize=None)
    def best_score_diff(l, r):
        if l == r:
            return nums[l]
        take_left = nums[l] - best_score_diff(l + 1, r)
        take_right = nums[r] - best_score_diff(l, r - 1)
        return max(take_left, take_right)

    left_value = nums[0] - best_score_diff(1, len(nums) - 1)
    right_value = nums[-1] - best_score_diff(0, len(nums) - 2)
    return "left" if left_value >= right_value else "right"

def new_game():
    n = random.randrange(12, 21, 2)  # chỉ sinh số chẵn từ 2 đến 20
    st.session_state.nums = [random.randint(1, 20) for _ in range(n)]
    st.session_state.you_score = 0
    st.session_state.computer_score = 0
    st.session_state.turn = "you"
    st.session_state.history = []
    st.session_state.game_over = False

def take_stone(side, player):
    stone = st.session_state.nums.pop(0) if side == "left" else st.session_state.nums.pop()
    if player == "you":
        st.session_state.you_score += stone
        label = "You"
    else:
        st.session_state.computer_score += stone
        label = "Computer"
    st.session_state.history.append(f"{label} lấy {side}: {stone}")
    if not st.session_state.nums:
        st.session_state.game_over = True
        return
    st.session_state.turn = "computer" if player == "you" else "you"

def computer_turn():
    if st.session_state.turn != "computer" or st.session_state.game_over:
        return
    move = computer_best_move(tuple(st.session_state.nums))
    take_stone(move, "computer")

if "nums" not in st.session_state:
    new_game()

st.title(":green[🪨 Stone Game I]")
st.caption("Bạn là Alice · Computer là Bob · Chọn đá bên trái hoặc phải")

col_left, col_right, newgame = st.columns(3)
col_left.metric("You (Alice)", st.session_state.you_score)
col_right.metric("Computer (Bob)", st.session_state.computer_score)
with newgame:
    if st.button("🔄 Ván mới"):
        new_game()
        st.rerun()

st.divider()

if st.session_state.game_over:
    if st.session_state.you_score > st.session_state.computer_score:
        st.success("🎉 You win!")
    elif st.session_state.you_score < st.session_state.computer_score:
        st.error("🤖 Computer wins!")
    else:
        st.info("🤝 Draw!")
    st.write(f"**Kết quả:** You = {st.session_state.you_score} | Computer = {st.session_state.computer_score}")
else:
    st.subheader("Các viên đá còn lại")
    # Hiển thị theo grid n//2 cột
    n_cols = 1 + len(st.session_state.nums) // 2

    for i in range(0, len(st.session_state.nums), n_cols):
        row = st.columns(n_cols)
        for j, col in enumerate(row):
            idx = i + j
            if idx < len(st.session_state.nums):
                col.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        margin:5px;
                        padding:12px 8px;
                        border-radius:10px;
                        color:white;
                        background:#4F46E5;
                        font-size:20px;
                        font-weight:600;">
                        {st.session_state.nums[idx]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    st.caption(f"Còn {len(st.session_state.nums)} viên đá")

    if st.session_state.turn == "you":
        st.info("Đến lượt bạn: chọn viên bên trái hoặc bên phải.")
        col1, col2 = st.columns(2)
        if col1.button(f"⬅️ Lấy trái ({st.session_state.nums[0]})", use_container_width=True):
            take_stone("left", "you")
            st.rerun()
        if col2.button(f"Lấy phải ({st.session_state.nums[-1]}) ➡️", use_container_width=True):
            take_stone("right", "you")
            st.rerun()
    else:
        st.warning("Computer đang chọn nước đi tối ưu...")
        computer_turn()
        st.rerun()

st.divider()
with st.expander("Lịch sử lượt chơi"):
    if st.session_state.history:
        col_you, col_bot = st.columns(2)

        with col_you:
            st.markdown("**You (Alice)**")
            for i, item in enumerate(st.session_state.history, start=1):
                if item.startswith("You"):
                    st.write(f"{i}. {item}")

        with col_bot:
            st.markdown("**Computer (Bob)**")
            for i, item in enumerate(st.session_state.history, start=1):
                if item.startswith("Computer"):
                    st.write(f"{i}. {item}")
    else:
        st.caption("Chưa có lượt nào.")
