import random
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

DIGITS = "012345"
K = 4

# ============================================================
# GAME LOGIC
# ============================================================

def generate_secret():
    """Generate a random 4-digit secret code."""
    return "".join(random.choices(DIGITS, k=K))


def calculate_score(secret, guess):
    """
    Calculate:
        white = correct digit + correct position
        red   = correct digit + wrong position
    """

    white = 0
    red = 0

    # Convert to lists so we can mark used digits
    secret_remaining = list(secret)
    guess_remaining = list(guess)

    # --------------------------------------------------------
    # Step 1: White
    # --------------------------------------------------------

    for i in range(K):
        if guess[i] == secret[i]:
            white += 1

            # Mark both positions as already used
            secret_remaining[i] = None
            guess_remaining[i] = None

    # --------------------------------------------------------
    # Step 2: Red
    # --------------------------------------------------------

    for i in range(K):

        if guess_remaining[i] is None:
            continue

        if guess_remaining[i] in secret_remaining:

            red += 1

            # Mark the matched secret digit as used
            idx = secret_remaining.index(guess_remaining[i])
            secret_remaining[idx] = None

    return white, red


def validate_guess(guess):
    """Validate user's input."""

    if not guess:
        return False, "Please enter a guess."

    if len(guess) != K:
        return False, f"Please enter exactly {K} digits."

    if not all(char in DIGITS for char in guess):
        return False, "Digits must be between 0 and 5."

    return True, ""

# ============================================================
# RULES
# ============================================================
def scoring_descr():
    with st.expander("##### 🎯 Scoring", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div style="background-color: rgba(255,255,255,0.08); padding:10px;">
                    <span style="color:white; font-weight:bold;">⚪ White</span>
                    <ul>
                        <li><span style="color:#D1FFBD">✓ Correct digit</span></li>
                        <li><span style="color:#D1FFBD">✓ Correct position</span></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div style="background-color: rgba(239,68,68,0.10); padding:10px;">
                    <span style="color:red"; font-weight:bold;> 🔴 Red </span>
                    <ul>
                        <li> <span style="color:#D1FFBD"> ✓ Correct digit  </span> </li>
                        <li> <span style="color:#FAC898"> ✗ Wrong position </span> </li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.write(" ")

def get_example():
    c1, c2 = st.columns([5, 4])
    with c1:
        with st.expander("##### 💡 Example", expanded=True):
            _, col1, _, col2, _ = st.columns([1, 2, 1, 2, 1])
            with col1:
                st.metric(label="**🔐 Secret**", value = "1234")
            with col2:
                st.metric(label="🎲 Your guess", value = "1442")

    with c2:
        with st.expander("##### 📊 Result", expanded=True):
            get_result()
            get_explaination()

def get_result():
    c1, c2 = st.columns([4, 3])
    with c1:
        st.markdown(
            """
            <div style="
                padding: 8px 12px;
                border-radius: 8px;
                background-color: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.15);
            ">
                ⚪ <b>White = 1</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div style="
                padding: 8px 12px;
                border-radius: 8px;
                background-color: rgba(239,68,68,0.10);
                border: 1px solid rgba(239,68,68,0.25);
            ">
                🔴 <b>Red = 2</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write(" ")

def get_explaination():
    st.markdown("**Why?**")
    st.markdown(
        """
        - **1** is in the correct position → **⚪ 1 White**
        - One **4** is in the wrong position → **🔴 1 Red**
        - **2** is in the wrong position → **🔴 1 Red**
        """
    )

def get_rule_descr():
    with st.expander("📖 How to play", expanded=True):
        st.write(
            f"""
                I, the computer, have chosen a secret **{K}-digit number**.
                - Each digit is between **0 and 5**.
                - Your goal is to guess the secret number.
            """
        )
        scoring_descr()
        get_example()
