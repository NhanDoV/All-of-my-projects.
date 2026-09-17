import streamlit as st
import random

st.set_page_config(
    page_title="Maze Adventure",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== EMOJI ====================
WALL   = "🧱"
ROAD   = "⬜"
PLAYER = "🧑"
GOAL   = "🏁"
TREE   = "🌲"
RIVER  = "🌊"
DEMON  = "😈"
EMPTY  = "　"   # full-width space để căn đẹp

# ==================== MAPS (built-in) ====================
# 0 = road, 1 = wall, 2 = goal, 3 = tree, 4 = river, 5 = demon

LEVEL1 = [  # basic: wall + road + goal
    [1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,2,1],
    [1,1,1,1,1,1,1,1,1,1,1],
]

LEVEL2 = [  # medium: + tree + river
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,3,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,4,1,0,1],
    [1,0,1,0,0,0,0,0,1,4,1,0,1],
    [1,0,1,1,1,3,1,0,1,4,1,0,1],
    [1,0,0,0,0,0,1,0,0,4,0,0,1],
    [1,1,1,1,1,0,1,1,1,4,1,0,1],
    [1,0,0,3,0,0,0,0,1,4,1,0,1],
    [1,0,1,1,1,1,1,0,1,4,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
]

LEVEL3 = [  # hard: + demon
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,3,0,0,5,0,0,1],
    [1,0,1,0,1,0,1,1,1,4,1,0,1,1],
    [1,0,1,0,0,0,0,0,1,4,1,0,0,1],
    [1,0,1,1,1,3,1,0,1,4,1,5,0,1],
    [1,0,0,0,0,0,1,0,0,4,0,0,0,1],
    [1,1,1,1,1,0,1,1,1,4,1,0,1,1],
    [1,0,0,3,0,0,0,0,1,4,1,0,0,1],
    [1,0,1,1,1,1,1,0,1,4,1,5,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

MAPS = {1: LEVEL1, 2: LEVEL2, 3: LEVEL3}

# ==================== HELPER ====================
def cell_to_emoji(val, is_player=False):
    if is_player:
        return PLAYER
    return {
        0: ROAD,
        1: WALL,
        2: GOAL,
        3: TREE,
        4: RIVER,
        5: DEMON,
    }.get(val, ROAD)

def render_maze(maze, player_pos):
    rows = []
    for r, row in enumerate(maze):
        line = []
        for c, val in enumerate(row):
            is_player = (r, c) == player_pos
            line.append(cell_to_emoji(val, is_player))
        rows.append("".join(line))
    return "\n".join(rows)

def find_start(maze):
    # tìm vị trí road đầu tiên gần góc trên-trái
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == 0:
                return (r, c)
    return (1, 1)

def is_valid(maze, pos):
    r, c = pos
    if not (0 <= r < len(maze) and 0 <= c < len(maze[0])):
        return False
    return maze[r][c] != 1  # không đi vào tường

def get_demon_positions(maze):
    """Return list of (r, c) where cell == 5"""
    positions = []
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == 5:
                positions.append((r, c))
    return positions

def move_demons(maze, player_pos, level):
    """Move each demon randomly 1-2 steps (only on road/tree). Returns new maze."""
    if level < 3:
        return maze

    new_maze = [row[:] for row in maze]
    demons = get_demon_positions(new_maze)

    # clear old demon positions first
    for r, c in demons:
        new_maze[r][c] = 0          # turn into road

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up down left right

    for r, c in demons:
        # try to move 1 or 2 steps
        steps = random.randint(1, 2)
        moved = False

        # shuffle directions so movement feels random
        random.shuffle(directions)

        for dr, dc in directions:
            nr, nc = r, c
            valid_path = True
            for _ in range(steps):
                nr += dr
                nc += dc
                # out of bound or blocked?
                if not (0 <= nr < len(new_maze) and 0 <= nc < len(new_maze[0])):
                    valid_path = False
                    break
                cell = new_maze[nr][nc]
                # only allow road (0) or tree (3). Don't step on wall/river/goal/other demons/player
                if cell in (1, 2, 4, 5) or (nr, nc) == player_pos:
                    valid_path = False
                    break
            if valid_path:
                new_maze[nr][nc] = 5
                moved = True
                break

        if not moved:
            # stay in place if no valid move
            new_maze[r][c] = 5

    return new_maze

def can_move(maze, pos, level):
    r, c = pos
    cell = maze[r][c]
    if cell == 1:          # wall
        return False
    if level >= 2 and cell == 4:  # river (level 2+)
        return False
    if level >= 3 and cell == 5:  # demon (level 3)
        return False
    return True

# ==================== SESSION STATE ====================
if "level" not in st.session_state:
    st.session_state.level = 1
if "maze" not in st.session_state:
    st.session_state.maze = [row[:] for row in MAPS[1]]
if "player" not in st.session_state:
    st.session_state.player = find_start(st.session_state.maze)
if "won" not in st.session_state:
    st.session_state.won = False
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "message" not in st.session_state:
    st.session_state.message = ""

def reset_level(level):
    st.session_state.level = level
    st.session_state.maze = [row[:] for row in MAPS[level]]
    st.session_state.player = find_start(st.session_state.maze)
    st.session_state.won = False
    st.session_state.game_over = False
    st.session_state.message = ""

def move(dr, dc):
    if st.session_state.won or st.session_state.game_over:
        return
    r, c = st.session_state.player
    new_pos = (r + dr, c + dc)
    maze = st.session_state.maze
    level = st.session_state.level

    if not is_valid(maze, new_pos):
        st.session_state.message = "🚫 Không thể đi hướng đó!"
        return

    cell = maze[new_pos[0]][new_pos[1]]

    # kiểm tra va chạm đặc biệt
    if level >= 2 and cell == 4:  # river
        st.session_state.game_over = True
        st.session_state.message = "🌊 Bạn bị cuốn trôi bởi dòng sông!"
        return
    if level >= 3 and cell == 5:  # demon
        st.session_state.game_over = True
        st.session_state.message = "😈 Bạn bị quỷ dữ tấn công!"
        return

    # di chuyển thành công
    st.session_state.player = new_pos
    st.session_state.message = ""

    # sau khi player di chuyển → quỷ cũng di chuyển (nếu level ≥ 3)
    if level >= 3:
        st.session_state.maze = move_demons(
            st.session_state.maze,
            st.session_state.player,
            level
        )
        # kiểm tra xem player có bị quỷ đụng sau khi quỷ di chuyển không
        pr, pc = st.session_state.player
        if st.session_state.maze[pr][pc] == 5:
            st.session_state.game_over = True
            st.session_state.message = "😈 Bạn bị quỷ dữ tấn công!"
            return

    # thắng?
    if cell == 2:
        st.session_state.won = True
        st.session_state.message = f"🎉 Level {level} hoàn thành!"

# ==================== UI ====================
st.title("🗺️ Maze Adventure – Rule-based Game")

descr_col, play_col = st.columns([1, 2], gap="large")

with descr_col:
    st.header("📖 Mô tả trò chơi")
    st.markdown("""
**Mục tiêu:** Đưa nhân vật 🧑 đến đích 🏁  

**Các vật thể:**
- 🧱 Tường (không đi qua được)
- ⬜ Đường đi
- 🌲 Rừng cây (chỉ trang trí, đi được)
- 🌊 Sông (level 2+ → chết nếu bước vào)
- 😈 Quỷ dữ (level 3 → chết nếu chạm)
- 🏁 Đích

**Luật chơi:**
- Level 1 (Basic): Chỉ tường + đường + đích
- Level 2 (Medium): Thêm rừng & sông
- Level 3 (Hard): Thêm quỷ dữ
    """)

with play_col:

    st.subheader(f"Level : **{st.session_state.level}**")

    # st.header("🎮 Khu vực chơi")
    # Hiển thị maze
    maze_str = render_maze(st.session_state.maze, st.session_state.player)
    st.markdown(
        f"""
        <div style="
            font-family: monospace;
            font-size: 28px;
            line-height: 1.15;
            background: #1a1a2e;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            letter-spacing: 2px;
            overflow-x: auto;
        ">
        {maze_str.replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")  # spacing

    # Nút điều khiển
    col_up, col_left, col_down, col_right, col_reset = st.columns(5)

    with col_up:
        if st.button("⬆️ Lên", use_container_width=True):
            move(-1, 0)
            st.rerun()
    with col_left:
        if st.button("⬅️ Trái", use_container_width=True):
            move(0, -1)
            st.rerun()
    with col_down:
        if st.button("⬇️ Xuống", use_container_width=True):
            move(1, 0)
            st.rerun()
    with col_right:
        if st.button("➡️ Phải", use_container_width=True):
            move(0, 1)
            st.rerun()
    with col_reset:
        if st.button("🔄 Reset Level", use_container_width=True):
            reset_level(st.session_state.level)
            st.rerun()

    # Thông báo
    if st.session_state.message:
        if st.session_state.won:
            st.success(st.session_state.message)
            st.balloons()
        elif st.session_state.game_over:
            st.error(st.session_state.message)
        else:
            st.warning(st.session_state.message)

    # Nút sau khi thắng / thua
    if st.session_state.won:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.level < 3:
                if st.button("➡️ Next Level", type="primary", use_container_width=True):
                    reset_level(st.session_state.level + 1)
                    st.rerun()
            else:
                st.success("🏆 Bạn đã hoàn thành tất cả 3 level!")
        with c2:
            if st.button("🆕 New Game (Level 1)", use_container_width=True):
                reset_level(1)
                st.rerun()

    if st.session_state.game_over:
        st.markdown("---")
        if st.button("🆕 Chơi lại Level này", type="primary", use_container_width=True):
            reset_level(st.session_state.level)
            st.rerun()
        if st.button("🏠 Về Level 1", use_container_width=True):
            reset_level(1)
            st.rerun()