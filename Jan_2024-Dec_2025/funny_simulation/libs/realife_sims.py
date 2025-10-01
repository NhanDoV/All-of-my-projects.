import streamlit as st
from itertools import product

# =========================================================================================================
#--------------------------------------------- Decoration -------------------------------------------------
# .........................................................................................................
def realifesims_render_wrt_background(bg_img):
    st.markdown(f"""
        <style>
        .simulations {{
            background-image: url("data:image/jpg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }}
        </style>
        <div class="simulations">
            <h2 style="color: #C724B1;"> Real-life Simulations 🌍</h2>
            <p>Choose a domain and run toy simulations.</p>
        </div>
    """, unsafe_allow_html=True)

# =========================================================================================================
#--------------------------------------------- Board Game -------------------------------------------------
# .........................................................................................................
class Board:
    def Knight_move(self, x: int, y: int) -> list:
        """
            Return valid knight moves inside board and not yet visited.
        """
        moves = list(product([x-1, x+1],[y-2, y+2])) + list(product([x-2,x+2],[y-1,y+1]))
        moves = [(x,y) for x,y in moves if x >= 0 and y >= 0 and x < 8 and y < 8]
        return moves

# =========================================================================================================
#--------------------------------------------- Board Game -------------------------------------------------
# .........................................................................................................
class polygon_transformation:
    def is_valid_triangle(self, a: float, b: float, c: float) -> bool:
        return (a + b > c) * (a + c > b) * (b + c > a)

# =========================================================================================================
#-------------------------------------------- RUN & LAUNCH ------------------------------------------------
# .........................................................................................................
domain_topic_dict = {
    "Chess / Board Games": ["Knight’s tour", "8 Queens", "Tic-Tac-Toe"],
    "Physics": ["Projectile motion", "Brownian motion", "Pendulum"],
    "Card Games": ["Poker simulation", "Blackjack", "Monty Hall"],
    "Biology": ["Predator-Prey (Lotka–Volterra)", "SIR Model"],
    "Social / Economics": ["Prisoner’s dilemma", "Auction toy model"],
    "Construction / Transformation": ["Making stick to polygon", ""]
}

def run():
    c1, _, c2 = st.columns([9, 1, 9])
    with c1:
        domain = st.selectbox("Choose a domain", list(domain_topic_dict.keys()))
    with c2:
        topic_name = st.selectbox("Select problem", domain_topic_dict.get(domain, []))
    