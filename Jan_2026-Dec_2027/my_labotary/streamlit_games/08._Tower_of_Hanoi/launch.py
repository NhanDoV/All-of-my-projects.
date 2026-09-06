import streamlit as st
import random
from typing import List, Tuple, Optional

# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────
ROD_COLORS = ["red", "blue", "green", "violet", "orange", "yellow"]

st.set_page_config(
    page_title="Tower of Hanoi",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
def init_game(n_rods: int, n_disks: int):
    colors = random.sample(ROD_COLORS, n_rods)
    target = random.choice([i for i in range(n_rods) if i != 0])
    rods: List[List[int]] = [[] for _ in range(n_rods)]
    rods[0] = list(range(n_disks, 0, -1))  # lớn dưới → nhỏ trên

    st.session_state.n_rods = n_rods
    st.session_state.n_disks = n_disks
    st.session_state.rod_colors = colors
    st.session_state.target_rod = target
    st.session_state.rods = rods
    st.session_state.selected = None
    st.session_state.moves = 0
    st.session_state.history = []
    st.session_state.move_stack = []
    st.session_state.won = False
    st.session_state.hint_msg = ""
    st.session_state.show_balloons = False

def ensure_state():
    if "rods" not in st.session_state:
        init_game(3, 3)

# ────────────────────────────────────────────────
# Game logic
# ────────────────────────────────────────────────
def can_move(src: int, dst: int) -> bool:
    rods = st.session_state.rods
    if not rods[src]:
        return False
    disk = rods[src][-1]
    if not rods[dst]:
        return True
    return disk < rods[dst][-1]

def do_move(src: int, dst: int) -> bool:
    if not can_move(src, dst):
        st.toast("Illegal move – larger disk cannot sit on a smaller one", icon="⚠️")
        return False
    disk = st.session_state.rods[src].pop()
    st.session_state.rods[dst].append(disk)
    st.session_state.moves += 1
    st.session_state.history.append(f"#{st.session_state.moves}: disk {disk}  →  Rod {dst + 1}")
    st.session_state.move_stack.append((src, dst, disk))
    st.session_state.hint_msg = ""
    if len(st.session_state.rods[st.session_state.target_rod]) == st.session_state.n_disks:
        st.session_state.won = True
        st.session_state.show_balloons = True
    return True

def undo_move():
    if not st.session_state.move_stack:
        st.toast("Nothing to undo", icon="ℹ️")
        return
    src, dst, disk = st.session_state.move_stack.pop()
    st.session_state.rods[dst].pop()
    st.session_state.rods[src].append(disk)
    st.session_state.moves -= 1
    if st.session_state.history:
        st.session_state.history.pop()
    st.session_state.won = False
    st.session_state.show_balloons = False
    st.session_state.selected = None
    st.session_state.hint_msg = ""

def on_rod_click(idx: int):
    if st.session_state.won:
        return
    sel = st.session_state.selected
    if sel is None:
        if st.session_state.rods[idx]:
            st.session_state.selected = idx
    else:
        if sel == idx:
            st.session_state.selected = None
        else:
            do_move(sel, idx)
            st.session_state.selected = None

# ────────────────────────────────────────────────
# Hint
# ────────────────────────────────────────────────
def find_next_move_classic(n: int, source: int, target: int, aux: int,
                           rods: List[List[int]]) -> Optional[Tuple[int, int]]:
    sequence = []
    def hanoi(k, s, t, a):
        if k == 0:
            return
        hanoi(k - 1, s, a, t)
        sequence.append((s, t))
        hanoi(k - 1, a, t, s)
    hanoi(n, source, target, aux)

    current = [list(r) for r in rods]
    for src, dst in sequence:
        if current[src] and (not current[dst] or current[src][-1] < current[dst][-1]):
            disk = current[src].pop()
            current[dst].append(disk)
            if can_move(src, dst):
                return src, dst
    return None

def get_hint() -> str:
    rods = st.session_state.rods
    n = st.session_state.n_disks
    target = st.session_state.target_rod
    n_rods = st.session_state.n_rods

    source = 0
    if target == 0:
        source = 1 if n_rods > 1 else 0
    aux_candidates = [i for i in range(n_rods) if i != source and i != target]
    aux = aux_candidates[0] if aux_candidates else source

    move = find_next_move_classic(n, source, target, aux, rods)
    if move:
        s, d = move
        return f"💡 Hint: Move top disk from **Rod {s+1}** → **Rod {d+1}**"
    for s in range(n_rods):
        for d in range(n_rods):
            if s != d and can_move(s, d):
                return f"💡 Hint: Try Rod {s+1} → Rod {d+1}"
    return "💡 No legal moves left (or already solved)"

# ────────────────────────────────────────────────
# Render board
# ────────────────────────────────────────────────
def render_board() -> str:
    rods = st.session_state.rods
    colors = st.session_state.rod_colors
    target = st.session_state.target_rod
    selected = st.session_state.selected
    n_disks = st.session_state.n_disks
    n_rods = len(rods)

    rod_parts = []
    for i in range(n_rods):
        color = colors[i]
        is_sel = "selected" if selected == i else ""

        disks_html = ""
        for d in rods[i]:  # lớn → nhỏ + column-reverse = nhỏ trên
            w = 28 + (d / n_disks) * 62
            disks_html += f'<div class="disk" style="width:{w}%;">{d}</div>'

        badge = '<div class="badge">🎯 TARGET</div>' if target == i else ""

        rod_parts.append(
            f'<div class="rod {is_sel}">'
            f'{badge}'
            f'<div class="label">Rod {i+1}</div>'
            f'<div class="pole" style="background:radial-gradient(circle at 50% 30%, {color} 96%, transparent 99%)"></div>'
            f'<div class="stack">{disks_html}</div>'
            f'<div class="base" style="background:radial-gradient(circle, {color} 95%, transparent 99%)"></div>'
            f'</div>'
        )

    board = "".join(rod_parts)

    css = """
    <style>
    .board{
        display:flex;justify-content:space-evenly;align-items:flex-end;
        gap:6px ;padding:28px 4px 24px;
        background:radial-gradient(#0f0f23 0%, #1a1a3a 100%);
        border-radius:16px; min-height:420px;
        box-shadow:inset 0 0 50px rgba(0,0,0,.45);overflow:hidden;
    }
    .rod{flex:1;max-width:180px;display:flex;flex-direction:column;
         align-items:center;position:relative;padding-top:12px;}
    .rod.selected{outline:3px solid #ffd700;outline-offset:8px;border-radius:14px;}
    .label{color:#e0e0e0;font-size:13px;font-weight:600;margin-bottom:6px;letter-spacing:.4px;}
    .badge{position:absolute;top:-9px;background:#ffd700;color:#111;
           font-size:10px;font-weight:700;padding:2px 9px;border-radius:12px;z-index:5;white-space:nowrap;}
    .pole{width:9px;height:320px;border-radius:5px 5px 0 0;
          box-shadow:2px 0 5px rgba(0,0,0,.35);z-index:1;}
    .stack{position:absolute;bottom:26px;width:100%;display:flex;
           flex-direction:column-reverse;align-items:center;z-index:2;pointer-events:none;}
    .disk{height:26px;background:linear-gradient(180deg,#ffffff 0%,#f0f0f0 35%,#c8c8c8 100%);
          clip-path:polygon(7% 0%,93% 0%,100% 100%,0% 100%);
          border:1.5px solid #999;border-radius:3px;box-shadow:0 2px 4px rgba(0,0,0,.28);
          margin:2.5px auto;display:flex;align-items:center;justify-content:center;
          font-size:11px;font-weight:600;color:#333;user-select:none;}
    .base{width:100%;height:20px;border-radius:5px;margin-top:-3px;
          box-shadow:0 3px 7px rgba(0,0,0,.3);z-index:0;}
    </style>
    """
    return css + f'<div class="board">{board}</div>'

# ────────────────────────────────────────────────
# UI helpers
# ────────────────────────────────────────────────
def game_description():
    st.markdown("""
                ### 🏛️ Tower of Hanoi
                **Rules**
                1. Move only **one disk** per turn.  
                2. You may only take the **top-most** disk of a rod.  
                3. A larger disk **cannot** be placed on a smaller disk.  
                4. Goal: move the entire tower onto the **🎯 TARGET** rod.
                5. <span style="color:#ffd700; font-weight:700;">How to play:</span> Click the **Rod i** bounded-box to select the top disk → then click **Rod j** to move it there (click the same Rod again to deselect).
    """, unsafe_allow_html=True)

def game_historic():
    st.markdown("#### Move history")
    hist = st.session_state.get("history", [])
    if not hist:
        st.caption("No moves yet.")
    else:
        recent = hist[-14:]
        # chia 14 dòng gần nhất thành 3 cột
        cols = st.columns(3)
        for i, line in enumerate(recent):
            with cols[i % 3]:
                st.text(line)
        if len(hist) > 14:
            st.caption(f"… +{len(hist)-14} earlier")

# ────────────────────────────────────────────────
# Layout (chạy trực tiếp)
# ────────────────────────────────────────────────
ensure_state()

left, right = st.columns([3, 5], gap="large")

with left:
    st.markdown("---")
    descr_col, params_col = st.columns([3, 2], gap='medium')
    with descr_col:
        game_description()        
    with params_col:
        n_rods = st.selectbox("Number of rods", [3, 4, 5], index=0, key="ui_rods")

        max_d = n_rods + 2
        min_d = max(3, n_rods - 1)
        n_disks = st.selectbox(
            "Number of disks",
            list(range(min_d, max_d + 1)),
            index=0,
            key="ui_disks",
        )


        if st.button("🔄 New Game", type="primary", use_container_width=True):
            init_game(n_rods, n_disks)
            st.rerun()

        hint, back = st.columns(2, gap='large')
        with hint:
            if st.button("💡 Hint", use_container_width=True):
                st.session_state.hint_msg = get_hint()
                st.rerun()
        with back:
            if st.button("↩️ Back", use_container_width=True):
                undo_move()
                st.rerun()

    if st.session_state.get("hint_msg"):
        st.info(st.session_state.hint_msg)

    st.markdown("---")
    game_historic()

with right:
    st.markdown("### Play Game")

    if (st.session_state.n_rods != n_rods or st.session_state.n_disks != n_disks):
        init_game(n_rods, n_disks)
        st.rerun()

    # Nút chọn cột
    n = st.session_state.n_rods
    btn_cols = st.columns(n)
    for i in range(n):
        with btn_cols[i]:
            label = f"Rod {i+1}"
            if st.session_state.selected == i:
                label = f"▶ {label}"
            if st.button(label, key=f"btn_{i}", use_container_width=True):
                on_rod_click(i)
                st.rerun()

    st.markdown(render_board(), unsafe_allow_html=True)

    sel = st.session_state.selected
    st.markdown(
        f"**Moves:** `{st.session_state.moves}` &nbsp;|&nbsp; "
        f"**Target:** `Rod {st.session_state.target_rod+1}` &nbsp;|&nbsp; "
        f"**Selected:** `{sel+1 if sel is not None else '—'}`"
    )

    if st.session_state.won:
        st.success(f"🎉 Solved in **{st.session_state.moves}** moves!")
        if st.session_state.show_balloons:
            st.balloons()
            st.session_state.show_balloons = False   # chỉ bay 1 lần
        if st.button("Play again", type="primary"):
            init_game(st.session_state.n_rods, st.session_state.n_disks)
            st.rerun()

with st.expander("Instruction", expanded=True):
    c1, c2, c3 = st.columns(3, gap='large')
    with c1:
        st.image('hint1.png')
    with c2:
        st.image('hint2.png')
    with c3:
        st.image('hint3.png')
