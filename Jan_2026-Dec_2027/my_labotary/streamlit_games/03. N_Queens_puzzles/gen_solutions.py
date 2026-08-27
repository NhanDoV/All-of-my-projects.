import streamlit as st
import matplotlib.pyplot as plt

class AllSolution:
    def solveNQueens(self, n: int) -> list[list[str]]:
        col = [False] * n
        posDiag = [False] * (n * 2)
        negDiag = [False] * (n * 2)
        res = []
        board = [["."] * n for i in range(n)]

        def backtrack(r):
            if r == n:
                copy = ["".join(row) for row in board]
                res.append(copy)
                return
            
            for c in range(n):
                if col[c] or posDiag[r + c] or negDiag[r - c + n]:
                    continue
                col[c] = True
                posDiag[r + c] = True
                negDiag[r - c + n] = True
                board[r][c] = "Q"

                backtrack(r + 1)

                col[c] = False
                posDiag[r + c] = False
                negDiag[r - c + n] = False
                board[r][c] = "."

        backtrack(0)
        return res

def rule_play():
    st.markdown(
        "The goal is to place **N queens on an N × N chessboard** such that **no two queens can attack each other**.",
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div style="background:#E8F5E9;padding:12px;border-radius:10px;text-align:center;color:#2E7D32"><b>One ♛ per row</b></div>', 
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#E3F2FD;padding:12px;border-radius:10px;text-align:center;color:#1565C0"><b>One ♛ per column</b></div>', 
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div style="background:#FFF3E0;padding:12px;border-radius:10px;text-align:center;color:#EF6C00"><b>No two ♛ can share the same diagonal</b></div>', 
                    unsafe_allow_html=True)
    with c4:
        st.markdown('<div style="background:#F3E5F5;padding:12px;border-radius:10px;text-align:center;color:#7B1FA2"><b>♛ attack horizontally, vertically, and diagonally</b></div>', 
                    unsafe_allow_html=True)

def make_chessboard(board):
    n = len(board)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Chessboard
    for row in range(n):
        for col in range(n):
            color = 'white' if (row + col) % 2 == 0 else 'gray'

            ax.add_patch(
                plt.Rectangle(
                    (col, n - row - 1),
                    1, 1,
                    facecolor=color
                )
            )

            # Queen
            if board[row][col] == 'Q':
                ax.text(
                    col + 0.5,
                    n - row - 0.5,
                    '♛',
                    fontsize=40,
                    ha='center',
                    va='center'
                )

    # Grid
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))

    ax.grid(True, linewidth=1)

    # Remove labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    st.pyplot(fig)

def view_mode(n, res):
    """ n only has 3 values: 2, 3, 4 """
    for i, col_i in enumerate(st.columns(n)):
        with col_i:
            case_i = st.number_input('Case idx:', min_value=0, max_value=len(res), value=i, step=1, key=i)
            make_chessboard(res[case_i])
