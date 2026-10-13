import streamlit as st
import streamlit.components.v1 as components
import json
import random

st.set_page_config(
    page_title="IsMatched",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Item pools (already enough for largest board: 5×4 = 10 pairs) ────────────
COLOR_ITEMS = [
    {"value": "#ef4444", "label": "Red",    "type": "color"},
    {"value": "#f97316", "label": "Orange", "type": "color"},
    {"value": "#eab308", "label": "Yellow", "type": "color"},
    {"value": "#22c55e", "label": "Green",  "type": "color"},
    {"value": "#3b82f6", "label": "Blue",   "type": "color"},
    {"value": "#8b5cf6", "label": "Purple", "type": "color"},
    {"value": "#ec4899", "label": "Pink",   "type": "color"},
    {"value": "#14b8a6", "label": "Teal",   "type": "color"},
    {"value": "#a855f7", "label": "Violet", "type": "color"},
    {"value": "#84cc16", "label": "Lime",   "type": "color"},
    {"value": "#06b6d4", "label": "Cyan",   "type": "color"},
    {"value": "#f43f5e", "label": "Rose",   "type": "color"},
]

ANIMAL_ITEMS = [
    {"value": "🐶", "label": "Dog",    "type": "animal"},
    {"value": "🐱", "label": "Cat",    "type": "animal"},
    {"value": "🦊", "label": "Fox",    "type": "animal"},
    {"value": "🐼", "label": "Panda",  "type": "animal"},
    {"value": "🐸", "label": "Frog",   "type": "animal"},
    {"value": "🐵", "label": "Monkey", "type": "animal"},
    {"value": "🦁", "label": "Lion",   "type": "animal"},
    {"value": "🐯", "label": "Tiger",  "type": "animal"},
    {"value": "🐨", "label": "Koala",  "type": "animal"},
    {"value": "🐰", "label": "Rabbit", "type": "animal"},
    {"value": "🐻", "label": "Bear",   "type": "animal"},
    {"value": "🐷", "label": "Pig",    "type": "animal"},
]

GRID_OPTIONS = {
    "2x4": (2, 4),
    "2x5": (2, 5),
    "3x4": (3, 4),
    "4x2": (4, 2),
    "4x3": (4, 3),
    "4x4": (4, 4),
    "4x5": (4, 5),
    "5x2": (5, 2),
    "5x4": (5, 4),
}

# ── Custom CSS to match dark theme ───────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px;}
    
    /* Dark background */
    .stApp {background-color: #0f1117;}
    
    /* Section headers */
    h1, h2, h3 {color: #e2e8f0 !important; font-family: 'Segoe UI', system-ui, sans-serif;}
    
    /* Labels */
    .stSelectbox label, .stSlider label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Slider track */
    .stSlider [data-baseweb="slider"] {color: #ef4444;}
    
    /* Current setup box */
    .setup-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 1.5rem;
        border: 1px solid #334155;
    }
    .setup-box h4 {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0 0 0.6rem 0;
        font-weight: 600;
    }
    .setup-box ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .setup-box li {
        color: #60a5fa;
        font-size: 0.88rem;
        padding: 0.15rem 0;
    }
    .setup-box li::before {
        content: "• ";
        color: #3b82f6;
    }
    
    /* Logo */
    .logo-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .logo-row span {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }
</style>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 3], gap="large")

with col_left:
    st.markdown(
        '<div class="logo-row">'
        '<svg width="28" height="28" viewBox="0 0 24 24" fill="none">'
        '<path d="M6 6h4v4H6V6zm8 0h4v4h-4V6zM6 14h4v4H6v-4zm8 0h4v4h-4v-4z" fill="#a78bfa"/>'
        '<rect x="2" y="2" width="20" height="20" rx="5" stroke="#a78bfa" stroke-width="2" fill="none"/>'
        '</svg>'
        '<span>IsMatched</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Game settings")

    matching_type = st.selectbox(
        "Matching type",
        options=["Color matched", "Animal matched"],
        index=0,
        key="matching_type",
    )

    grid_label = st.selectbox(
        "Grid size",
        options=list(GRID_OPTIONS.keys()),
        index=list(GRID_OPTIONS.keys()).index("4x4"),
        key="grid_size",
    )
    rows, cols = GRID_OPTIONS[grid_label]
    total_cards = rows * cols
    pairs = total_cards // 2

    turns = st.slider(
        "Number of turns",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        key="turns",
    )

    # Current setup summary
    type_key = "color-matched" if matching_type == "Color matched" else "animal-matched"
    st.markdown(
        f"""
        <div class="setup-box">
            <h4>Current setup</h4>
            <ul>
                <li>Type: {type_key}</li>
                <li>Grid: {grid_label}</li>
                <li>Cards: {total_cards}</li>
                <li>Pairs: {pairs}</li>
                <li>Turns: {turns}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Prepare board data for JS ────────────────────────────────────────────────
item_pool = COLOR_ITEMS if matching_type == "Color matched" else ANIMAL_ITEMS
selected = random.sample(item_pool, pairs)
# Duplicate for pairs then shuffle
board = selected + selected
random.shuffle(board)

board_json = json.dumps(board)
is_color = matching_type == "Color matched"

# Dynamic card size so larger grids still fit nicely
max_dim = max(rows, cols)
if max_dim >= 5:
    card_px = 70
elif max_dim >= 4:
    card_px = 80
else:
    card_px = 90
comp_height = rows * (card_px + 12) + 220

# ── HTML / JS game component ─────────────────────────────────────────────────
game_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f1117;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #e2e8f0;
    padding: 0;
    overflow: hidden;
  }}
  .game-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0;
  }}
  .hint {{
    color: #64748b;
    font-size: 0.85rem;
    text-align: center;
  }}
  .start-btn {{
    width: 100%;
    max-width: 720px;
    background: transparent;
    border: 1px solid #334155;
    border-radius: 10px;
    color: #94a3b8;
    padding: 0.7rem 1rem;
    font-size: 0.9rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    transition: all 0.2s;
  }}
  .start-btn:hover {{
    border-color: #475569;
    color: #e2e8f0;
    background: #1e293b;
  }}
  .board-area {{
    display: flex;
    align-items: stretch;
    gap: 2.1rem;          /* padding between grid and status boxes */
    margin-top: 0.5rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat({cols}, 70px);
    grid-template-rows: repeat({rows}, 70px);
    gap: 14px;
  }}
  .card {{
    width: {card_px}px;
    height: {card_px}px;
    border-radius: 12px;
    cursor: pointer;
    position: relative;
    perspective: 600px;
    user-select: none;
  }}
  .card-inner {{
    width: 100%;
    height: 100%;
    transition: transform 0.35s ease;
    transform-style: preserve-3d;
    position: relative;
  }}
  .card.flipped .card-inner,
  .card.matched .card-inner {{
    transform: rotateY(180deg);
  }}
  .card-face {{
    position: absolute;
    inset: 0;
    backface-visibility: hidden;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
  }}
  .card-back {{
    background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 50%, #6d28d9 100%);
    background-image:
      radial-gradient(circle at 20% 30%, rgba(255,255,255,0.08) 0%, transparent 50%),
      radial-gradient(circle at 80% 70%, rgba(255,255,255,0.06) 0%, transparent 40%),
      linear-gradient(135deg, #4c1d95, #6d28d9);
    color: #c4b5fd;
    font-weight: 700;
    font-size: 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  .card-front {{
    background: #1e293b;
    transform: rotateY(180deg);
    border: 2px solid #334155;
  }}
  .card-front.color-face {{
    /* solid color fill handled by JS */
  }}
  .card.matched .card-front {{
    border-color: #22c55e;
    box-shadow: 0 0 12px rgba(34,197,94,0.35);
  }}
  .card.mismatch .card-front {{
    border-color: #ef4444;
    animation: shake 0.4s ease;
  }}
  @keyframes shake {{
    0%,100% {{ transform: rotateY(180deg) translateX(0); }}
    25% {{ transform: rotateY(180deg) translateX(-4px); }}
    75% {{ transform: rotateY(180deg) translateX(4px); }}
  }}
  .status-col {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;  /* top ↔ bottom */
    align-self: stretch;             /* cao bằng grid */
    gap: 0;
    padding-top: 0;
  }}
  .stat-box {{
    background: #fff;
    border-radius: 10px;
    padding: 0.6rem 1.1rem;
    text-align: center;
    min-width: 110px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
  }}
  .stat-label {{
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .stat-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #2563eb;
    line-height: 1.2;
  }}
  .msg {{
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 0.3rem;
    min-height: 1.2em;
    text-align: center;
  }}
  .overlay {{
    position: absolute;
    inset: 0;
    background: rgba(15,17,23,0.75);
    display: none;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    z-index: 10;
  }}
  .overlay.show {{ display: flex; }}
  .overlay-box {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
  }}
  .overlay-box h3 {{
    font-size: 1.3rem;
    margin-bottom: 0.4rem;
  }}
  .overlay-box p {{
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 1rem;
  }}
  .overlay-box button {{
    background: #3b82f6;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-size: 0.9rem;
    cursor: pointer;
  }}
  .overlay-box button:hover {{ background: #2563eb; }}
  .board-wrapper {{ position: relative; }}
</style>
</head>
<body>
<div class="game-wrap">
  <p class="hint">Flip the cards and find all matching pairs.</p>
  <button class="start-btn" id="btnNew">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 12a9 9 0 11-6.2-8.6"/>
      <polyline points="21 3 21 9 15 9"/>
    </svg>
    Start new game
  </button>

  <div class="board-area">
    <div class="board-wrapper">
      <div class="grid" id="grid"></div>
      <div class="overlay" id="overlay">
        <div class="overlay-box">
          <h3 id="overlayTitle"></h3>
          <p id="overlayMsg"></p>
          <button id="overlayBtn">Play again</button>
        </div>
      </div>
    </div>
    <div class="status-col">
      <div class="stat-box">
        <div class="stat-label">Turns left</div>
        <div class="stat-value" id="turnsVal">{turns}</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Matched</div>
        <div class="stat-value" id="matchedVal">0/{pairs}</div>
      </div>
    </div>
  </div>
  <p class="msg" id="msg">Select two cards to find a match.</p>
</div>

<script>
(function() {{
  const BOARD = {board_json};
  const IS_COLOR = {str(is_color).lower()};
  const PAIRS = {pairs};
  const MAX_TURNS = {turns};
  const ROWS = {rows};
  const COLS = {cols};

  let turnsLeft = MAX_TURNS;
  let matchedCount = 0;
  let flipped = [];
  let lock = false;
  let gameOver = false;

  const grid = document.getElementById('grid');
  const turnsEl = document.getElementById('turnsVal');
  const matchedEl = document.getElementById('matchedVal');
  const msgEl = document.getElementById('msg');
  const overlay = document.getElementById('overlay');
  const overlayTitle = document.getElementById('overlayTitle');
  const overlayMsg = document.getElementById('overlayMsg');

  function buildBoard() {{
    grid.innerHTML = '';
    BOARD.forEach((item, idx) => {{
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.idx = idx;
      card.dataset.value = item.value;

      const inner = document.createElement('div');
      inner.className = 'card-inner';

      const back = document.createElement('div');
      back.className = 'card-face card-back';
      back.textContent = '?';

      const front = document.createElement('div');
      front.className = 'card-face card-front';
      if (IS_COLOR) {{
        front.style.background = item.value;
        front.style.border = 'none';
      }} else {{
        front.textContent = item.value;
        front.style.fontSize = '2rem';
      }}

      inner.appendChild(back);
      inner.appendChild(front);
      card.appendChild(inner);
      card.addEventListener('click', () => onCardClick(card));
      grid.appendChild(card);
    }});
  }}

  function onCardClick(card) {{
    if (lock || gameOver) return;
    if (card.classList.contains('flipped') || card.classList.contains('matched')) return;
    if (flipped.length >= 2) return;

    card.classList.add('flipped');
    flipped.push(card);

    if (flipped.length === 1) {{
      msgEl.textContent = 'Choose one more card.';
    }}

    if (flipped.length === 2) {{
      lock = true;
      turnsLeft--;
      turnsEl.textContent = turnsLeft;

      const [a, b] = flipped;
      const match = a.dataset.value === b.dataset.value;

      if (match) {{
        setTimeout(() => {{
          a.classList.add('matched');
          b.classList.add('matched');
          matchedCount++;
          matchedEl.textContent = matchedCount + '/' + PAIRS;
          flipped = [];
          lock = false;
          msgEl.textContent = 'Nice match! Keep going.';

          if (matchedCount === PAIRS) {{
            endGame(true);
          }} else if (turnsLeft <= 0) {{
            endGame(false);
          }}
        }}, 400);
      }} else {{
        a.classList.add('mismatch');
        b.classList.add('mismatch');
        msgEl.textContent = 'Not a match. Try again.';
        setTimeout(() => {{
          a.classList.remove('flipped', 'mismatch');
          b.classList.remove('flipped', 'mismatch');
          flipped = [];
          lock = false;
          if (turnsLeft <= 0) {{
            endGame(false);
          }} else {{
            msgEl.textContent = 'Select two cards to find a match.';
          }}
        }}, 800);
      }}
    }}
  }}

  function endGame(won) {{
    gameOver = true;
    overlay.classList.add('show');
    if (won) {{
      overlayTitle.textContent = '🎉 You won!';
      overlayMsg.textContent = 'All ' + PAIRS + ' pairs matched with ' + turnsLeft + ' turns left.';
    }} else {{
      overlayTitle.textContent = '💥 Out of turns';
      overlayMsg.textContent = 'You matched ' + matchedCount + '/' + PAIRS + ' pairs. Try again!';
    }}
  }}

  function reset() {{
    // Force Streamlit to re-run by reloading the component parent
    // Since board is generated server-side, we just ask Streamlit to rerun
    window.parent.postMessage({{type: 'streamlit:setComponentValue', value: Date.now()}}, '*');
    // Fallback: simple page reload of the iframe content via Streamlit
    location.reload();
  }}

  document.getElementById('btnNew').addEventListener('click', () => {{
    // Trigger Streamlit rerun by changing a dummy query – simplest reliable way:
    // just reload the whole page so Streamlit regenerates a new shuffled board
    window.parent.location.reload();
  }});
  document.getElementById('overlayBtn').addEventListener('click', () => {{
    window.parent.location.reload();
  }});

  buildBoard();
}})();
</script>
</body>
</html>
"""

with col_right:
    components.html(game_html, height=comp_height, scrolling=False)