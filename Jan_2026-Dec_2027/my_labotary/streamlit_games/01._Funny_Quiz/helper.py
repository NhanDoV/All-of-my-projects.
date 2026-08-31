import random
import streamlit as st

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
    {
        "question": "The intelligence of South Korean police is best demonstrated by:",
        "options": [
            "Sewol ferry disaster 2014",
            "Itaewon stampede 2022",
            "Over 300 people missing on Jeju Island 2026",
            "All of the above"
        ],
        "answer": "All of the above"
    },
    {
        "question": "The ancient capital of Korea is:",
        "options": [
            "Seoul",
            "Gyeongju",
            "Osaka",
            "Pyongyang"
        ],
        "answer": "Gyeongju"
    },
    {
        "question": "Why do so many Korean women become streamers and camgirls?",
        "options": [
            "Because their plastic surgery debt is too high",
            "Because real jobs pay less than selling feet pics",
            "Because Korean men have nothing worth marrying",
            "All of the above"
        ],
        "answer": "All of the above"
    },

    {
        "question": "The most accurate description of the average Korean man is:",
        "options": [
            "Small dick + mandatory military service + no kids",
            "Works 80 hours a week then cries in the bathroom",
            "Pays for plastic surgery for his girlfriend then gets cheated on",
            "All of the above"
        ],
        "answer": "All of the above"
    },
    {
        "question": "Why do Koreans have a superiority complex toward other countries?",
        "options": [
            "Because they feel inferior about their 2cm dicks",
            "Because they need somewhere to vent when powerless against North Korea and Japan",
            "Because they got so much plastic surgery that their brains are rotten",
            "All of the above"
        ],
        "answer": "All of the above",
    },
    {
        "question": "When you think of K-pop groups, you think of:",
        "options": [
            "A bunch of cheap attention-seeking whores",
            "Their boyfriends/husbands must have tiny dicks",
            "Every single face looks like it came from the same plastic surgery clinic",
            "All of the above"
        ],
        "answer": "All of the above"
    },
    {
        "question": "The first president of South Korea was:",
        "options": [
            "Syngman Rhee",
            "Harry S. Truman (1945–1953)",
            "Dwight D. Eisenhower (1953–1961)",
            "Fumimaro Konoe"
        ],
        "answer": "Syngman Rhee "
    },
]