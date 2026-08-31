import streamlit as st
from helper import *

# =========================================== SESSION STATE
if "secret_code" not in st.session_state:
    st.session_state.secret_code = generate_secret()

if "previous_guesses" not in st.session_state:
    st.session_state.previous_guesses = []

if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "game_number" not in st.session_state:
    st.session_state.game_number = 0

# =========================================== NEW GAME
def new_game():
    st.session_state.secret_code = generate_secret()
    st.session_state.previous_guesses = []
    st.session_state.game_over = False
    st.session_state.game_number += 1

# =========================================== PAGE CONFIG & TITLE
st.set_page_config(page_title="Mastermind", page_icon="🎯", layout="wide")
st.title("🎯 Mastermind")

c1, c2 = st.columns([3, 2], border=False)
with c1:
    get_rule_descr()

with c2:
    # NEW GAME BUTTON

    if st.button("🔄 NEW GAME", use_container_width=True):
        new_game()
        st.rerun()
    # ============================================================
    # GAME
    # ============================================================

    if not st.session_state.game_over:

        st.subheader("Make your guess")
        guess = st.text_input(
            f"Enter a {K}-digit number using digits 0–5:",
            max_chars=K,
            key=f"guess_input_{st.session_state.game_number}"
        )

        if st.button(
            "🎯 GUESS",
            use_container_width=True
        ):

            guess = guess.strip()

            # ----------------------------------------------------
            # Validate
            # ----------------------------------------------------

            valid, error_message = validate_guess(guess)

            if not valid:
                st.error(error_message)

            else:
                # ------------------------------------------------
                # Calculate score
                # ------------------------------------------------

                white, red = calculate_score(
                    st.session_state.secret_code,
                    guess
                )

                # ------------------------------------------------
                # Save guess
                # ------------------------------------------------

                st.session_state.previous_guesses.append(
                    {
                        "guess": guess,
                        "white": white,
                        "red": red
                    }
                )

                # ------------------------------------------------
                # Win condition
                # ------------------------------------------------

                if white == K:

                    st.session_state.game_over = True

                    st.success("🎉 YOU GOT IT!")

                    st.balloons()

                    st.write(
                        f"It took you "
                        f"**{len(st.session_state.previous_guesses)} guesses**."
                    )

                else:

                    st.info(
                        f"⚪ White: **{white}**   "
                        f"🔴 Red: **{red}**"
                    )


    # ============================================================
    # GAME OVER
    # ============================================================

    else:

        st.success("🎉 GAME OVER!")

        st.write(
            f"The secret number was:"
        )

        st.code(
            st.session_state.secret_code,
            language="text"
        )

        st.write(
            f"You solved it in "
            f"**{len(st.session_state.previous_guesses)} guesses**."
        )

        if st.button(
            "🎮 PLAY AGAIN",
            use_container_width=True
        ):
            new_game()
            st.rerun()


    # ============================================================
    # GUESS HISTORY
    # ============================================================

    if st.session_state.previous_guesses:

        st.divider()

        st.subheader("📋 Guess History")

        # Latest guess first
        for i, result in enumerate(
            reversed(st.session_state.previous_guesses),
            start=1
        ):

            st.write(
                f"**Guess #{len(st.session_state.previous_guesses) - i + 1}**"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Guess",
                    result["guess"]
                )

            with col2:
                st.metric(
                    "⚪ White",
                    result["white"]
                )

            with col3:
                st.metric(
                    "🔴 Red",
                    result["red"]
                )