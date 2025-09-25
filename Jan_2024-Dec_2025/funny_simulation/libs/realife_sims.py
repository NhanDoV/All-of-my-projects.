import streamlit as st
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
    def Knight_move(self):
        pass

# =========================================================================================================
#-------------------------------------------- RUN & LAUNCH ------------------------------------------------
# .........................................................................................................
domain_topic_dict = {
    "Chess / Board Games": ["Knight’s tour", "8 Queens", "Tic-Tac-Toe"],
    "Physics": ["Projectile motion", "Brownian motion", "Pendulum"],
    "Card Games": ["Poker simulation", "Blackjack", "Monty Hall"],
    "Biology": ["Predator-Prey (Lotka–Volterra)", "SIR Model"],
    "Social / Economics": ["Prisoner’s dilemma", "Auction toy model"]
}

def run():
    c1, _, c2 = st.columns([9, 1, 9])
    with c1:
        domain = st.selectbox("Choose a domain", list(domain_topic_dict.keys()))
    with c2:
        topic_name = st.selectbox("Select problem", domain_topic_dict.get(domain, []))
    