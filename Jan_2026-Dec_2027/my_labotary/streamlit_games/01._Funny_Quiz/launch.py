import streamlit as st
from helper import *

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Mini Quiz Game",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>

.block-container {
    max-width: 100%;
    padding-top: 2.5rem;
    padding-bottom: 1rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* ---------- Title ---------- */

.game-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
    color: #FFFFFF
}

.game-subtitle {
    font-size: 16px;
    color: #666;
    margin-bottom: 25px;
}


/* ---------- Left panel ---------- */

.left-panel {
    background: #f7f9fc;
    border-radius: 15px;
    padding: 22px;
    border: 1px solid #e6eaf0;
}

.rule-box {
    background: #0d3b0d;
    border-radius: 12px;
    padding: 16px;
    margin-top: 10px;
    border: 1px solid #e6eaf0;
}

.score-box {
    background: #0a7a0a;
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
    text-align: center;
    border: 1px solid #e6eaf0;
}

.score-number {
    font-size: 36px;
    font-weight: 800;
}

.result-box {
    background: #033003;
    border-radius: 12px;
    padding: 18px;
    margin-top: 15px;
    border: 1px solid #e6eaf0;
}

/* ---------- Question ---------- */

.question-card {
    background: #1A421E;
    border: 1px solid #e6eaf0;
    border-radius: 15px;
    padding-top: 9px;
    padding-bottom: 6px;
    padding-left: 23px;
    padding-right: 29px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.question-number {
    font-size: 18px;
    font-weight: 700;
    color: #f0b090;
    margin-bottom: 6px;
}

.question-text {
    font-size: 18px;
    font-family: 'Fira Code', 'Consolas', monospace;     
    font-weight: 700;
    padding-left: 9px;
    margin-bottom: 6px;
}


/* ---------- Divider ---------- */

.section-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 15px;
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "game_started" not in st.session_state:
    st.session_state.game_started = False

if "questions" not in st.session_state:
    st.session_state.questions = []

if "score" not in st.session_state:
    st.session_state.score = 0

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "results" not in st.session_state:
    st.session_state.results = []

left_col, right_col = st.columns([6, 5], gap="large")

with left_col:
    st.markdown(
        '<div class="game-title">🧠 Mini Quiz Game 📝</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="game-subtitle">'
            'Test your “deep knowledge” of Korea with 4 random questions'
        '</div>',
        unsafe_allow_html=True
    )
    with st.expander("🎮 **Game Rules**", expanded=True):
        cl, cr = st.columns(2)
        with cl:
            st.markdown('''
                            <div style="background:#E8F5E9;padding:12px;border-radius:10px;text-align:center;color:#2E7D32">
                                <b> Select 1 answer for each question. </b>
                            </div>
                        ''', unsafe_allow_html=True)
        with cr:
            st.markdown('''
                            <div style="background:#C6F3F5;padding:12px;border-radius:10px;text-align:center;color:#0C5659">
                                <b> Click SUBMIT ANSWERS to see your score </b>
                            </div>
                        ''', unsafe_allow_html=True)
        st.write("")

    _, c1, _, c2, _ = st.columns([1, 2, 3, 5, 1], gap='medium')
    with c1:
        if st.session_state.game_started:
            button_text = "🎮 NEW GAME"
            for _ in range(20):
                st.write(" ")
        else:
            st.write(" ")
            button_text = "▶️ PLAY"

        if st.button(
            button_text,
            use_container_width=True,
            type="primary"
        ):
            new_game()
            st.rerun()

    with c2:
        if st.session_state.submitted:
            score = st.session_state.score
            if score == 4:
                message = "🎉 Perfect! Excellent job!"
            elif score == 3:
                message = "👏 Great job!"
            elif score == 2:
                message = "💪 Keep practicing!"
            else:
                message = "📚 Better luck next time!"
            st.markdown(
                f"""
                    <div class="result-box">

                    <div style="font-size:20px; font-weight:800;">
                    🎉 Final Result
                    </div>

                    <br>

                    <div style="font-size:30px; font-weight:800;">
                    {score} / 4
                    </div>

                    <div style="margin-top:8px;">
                    {message}
                    </div>

                    </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "### 📊 Answer Summary"
        )
        for i, result in enumerate(st.session_state.results):
            if result["correct"]:
                st.success(
                    f"Question {i + 1}: ✅ Correct"
                )
            else:
                selected = result["selected"]
                if selected is None:
                    selected_text = "No answer"
                else:
                    selected_text = selected
                st.error(
                    f"Question {i + 1}: ❌ Incorrect"
                )
                st.caption(
                    f"Your answer: {selected_text}  |  "
                    f"Correct answer: {result['answer']}"
                )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

# =============================================================================
# RIGHT COLUMN
# =============================================================================
with right_col:
    st.markdown(
        '<div class="section-title">📋 Questions </div>',
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # Game has not started
    # -------------------------------------------------------------------------
    if not st.session_state.game_started:
        st.info(
            "👈 Click **PLAY** to start the game."
        )

    # -------------------------------------------------------------------------
    # Game started
    # -------------------------------------------------------------------------
    else:
        for idx, question in enumerate(
            st.session_state.questions
        ):

            st.markdown(
                f"""
                <div class="question-card">

                <div class="question-number">
                    QUESTION {idx + 1} / 4
                </div>

                <div class="question-text">
                    {question["question"]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # 2 × 2 answer layout
            col1, col2 = st.columns(2)
            for option_idx, option in enumerate(question["options"]):
                target_col = (col1 if option_idx % 2 == 0 else col2 )
                with target_col:
                    st.checkbox(option, key=f"q_{idx}_{option_idx}")

            # st.write("")

        # ---------------------------------------------------------------------
        # SUBMIT
        # ---------------------------------------------------------------------

        if st.button(
            "✅ SUBMIT ANSWERS",
            use_container_width=True,
            type="primary"
        ):

            # Check if each question has exactly one answer
            invalid_questions = []

            for idx, question in enumerate(
                st.session_state.questions
            ):

                selected_count = sum(
                    st.session_state.get(
                        f"q_{idx}_{option_idx}",
                        False
                    )
                    for option_idx in range(4)
                )

                if selected_count != 1:
                    invalid_questions.append(idx + 1)

            if invalid_questions:

                questions_text = ", ".join(
                    map(str, invalid_questions)
                )

                st.warning(
                    f"⚠️ Please select exactly one answer "
                    f"for question(s): {questions_text}"
                )

            else:
                submit_answers()
                st.rerun()