import random
import streamlit as st

QUESTIONS = [
    {
        "question": "Koreans are confident because:",
        "options": [
            "small dicks + mandatory military service",
            "more plastic surgery clinics than kindergartens",
            "men refuse to have kids like there’s a ban",
            "All of above"
        ],
        "answer": "All of above"
    },

    {
        "question": "When people think of Korea, they think of:",
        "options": [
            "Still not unified with the North",
            "plastic surgery but still ugly",
            "Dicks under 2cm & infertile men who refuse to have kids",
            "All of above"
        ],
        "answer": "All of above"
    },

    {
        "question": "Which city has the shortest average dick in the world?",
        "options": [
            "Seoul",
            "Busan",
            "Incheon",
            "Daegu"
        ],
        "answer": "Seoul"
    },

    {
        "question": "The relationship between South Korea and Japan is best described as:",
        "options": [
            "Former Japanese colony",
            "Current ally",
            "Future colony again",
            "All of the above"
        ],
        "answer": "All of the above"
    },

    {
        "question": "The average Korean looks like:",
        "options": [
            "Single eyelids + flat face",
            "Nose that needed a bridge implant",
            "Already had plastic surgery",
            "All of the above"
        ],
        "answer": "All of the above"
    },

    {
        "question": "Why does South Korea have the world’s lowest birth rate?",
        "options": [
            "Men have nothing worth putting inside",
            "Women only want money and plastic surgery",
            "Everyone is too busy working 80-hour weeks",
            "All of the above"
        ],
        "answer": "All of the above"
    },
]

def new_game():
    """
        Start a new game:
            - Randomly select 4 questions from 10
            - Reset answers
            - Reset score
    """
    st.session_state.questions = random.sample(QUESTIONS, 4)

    st.session_state.score = 0
    st.session_state.submitted = False
    st.session_state.results = []

    st.session_state.game_started = True

    # Reset all previous checkbox states
    for idx in range(5):
        for option_idx in range(4):
            key = f"q_{idx}_{option_idx}"

            if key in st.session_state:
                del st.session_state[key]


def get_selected_answer(question_idx, question):
    """
        Return the selected option for a question.

        Because checkbox is used, there could theoretically
        be multiple selections. We only return the first one.
    """

    selected = []

    for option_idx, option in enumerate(question["options"]):
        key = f"q_{question_idx}_{option_idx}"
        if st.session_state.get(key, False):
            selected.append(option)

    if len(selected) == 1:
        return selected[0]

    return None


def submit_answers():
    score = 0
    results = []

    for idx, question in enumerate(st.session_state.questions):
        selected = get_selected_answer(idx, question)
        correct = selected == question["answer"]

        if correct:
            score += 1

        results.append({
            "question": question["question"],
            "selected": selected,
            "answer": question["answer"],
            "correct": correct
        })

    st.session_state.score = score
    st.session_state.results = results
    st.session_state.submitted = True