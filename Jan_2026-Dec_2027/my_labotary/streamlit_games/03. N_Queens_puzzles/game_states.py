import streamlit as st
from helper import *

def initialize_game(n: int):
    """Khởi tạo hoặc reset hoàn toàn game."""
    st.session_state.n = n
    st.session_state.queens = set()
    st.session_state.history = []
    st.session_state.hint_cell = None
    st.session_state.solution = find_one_solution(n)
    st.session_state.status = "playing"

def update_game_status():
    """Cập nhật trạng thái thắng/chơi."""
    n = st.session_state.n
    queens = st.session_state.queens
    conflicts = get_conflict_queens(queens)

    if len(queens) == n and len(conflicts) == 0:
        st.session_state.status = "won"
    else:
        st.session_state.status = "playing"


def toggle_queen(position: tuple[int, int]):
    """
    Click ô:
    - Nếu đã có queen: remove.
    - Nếu trống: add.
    - Ghi lịch sử để hỗ trợ Undo.
    """
    if position in st.session_state.queens:
        st.session_state.queens.remove(position)

        st.session_state.history.append({
            "action": "remove",
            "position": position,
        })
    else:
        st.session_state.queens.add(position)

        st.session_state.history.append({
            "action": "add",
            "position": position,
        })

    st.session_state.hint_cell = None
    update_game_status()


def undo_last_move():
    """Đảo ngược action gần nhất."""
    if not st.session_state.history:
        return

    last_move = st.session_state.history.pop()
    action = last_move["action"]
    position = last_move["position"]

    if action == "add":
        st.session_state.queens.discard(position)

    elif action == "remove":
        st.session_state.queens.add(position)

    st.session_state.hint_cell = None
    update_game_status()


def show_hint():
    """
    Gợi ý một nước đi hợp lệ.
    Ưu tiên ô thuộc nghiệm solver và chưa được đặt.
    """
    queens = st.session_state.queens
    solution = st.session_state.solution

    if solution:
        for pos in solution:
            if pos not in queens:
                if not any(is_attacking(pos, queen) for queen in queens):
                    st.session_state.hint_cell = pos
                    return

    valid_moves = get_valid_moves(queens, st.session_state.n)

    if valid_moves:
        st.session_state.hint_cell = valid_moves[0]
    else:
        st.session_state.hint_cell = None


def load_solution():
    """Đặt bàn cờ thành một nghiệm hoàn chỉnh."""
    if st.session_state.solution:
        st.session_state.queens = set(st.session_state.solution)
        st.session_state.history = []
        st.session_state.hint_cell = None
        update_game_status()
