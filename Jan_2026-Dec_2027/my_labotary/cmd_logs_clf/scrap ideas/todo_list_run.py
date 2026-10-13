import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="To-do list",
    page_icon="📝",
    layout="centered"
)

# =========================
# MODERN COLOR SCHEME
# =========================
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    background-attachment: fixed;
}

/* Title */
h1 {
    color: #e0e7ff !important;
    text-align: center;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Input & select boxes */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background-color: #1e1b4b !important;
    color: #e0e7ff !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.3) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

/* Checkbox text */
.stCheckbox label {
    color: #e0e7ff !important;
    font-size: 1.05rem !important;
}

/* Divider */
hr {
    border-color: #4f46e5 !important;
    opacity: 0.4;
}

/* Delete button style */
button[kind="secondary"] {
    background: #1e1b4b !important;
    border: 1px solid #4f46e5 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📝 To-do list")

# =========================
# CATEGORY OPTIONS (common todo types)
# =========================
CATEGORIES = {
    "📝 General": "📝",
    "💻 Code / Tech": "💻",
    "🎤 Meeting / Interview": "🎤",
    "🛒 Shopping": "🛒",
    "📚 Study / Learn": "📚",
    "🏃 Health / Sport": "🏃",
    "💼 Work": "💼",
    "🏠 Home": "🏠",
    "💡 Idea": "💡",
    "📅 Plan": "📅",
}

# =========================
# SESSION STATE
# =========================
if "todos" not in st.session_state:
    st.session_state.todos = [
        {
            "task": "Viết TODO list hôm nay",
            "icon_category": "📝",
            "done": True
        },
        {
            "task": "Luyện tập Pysparks",
            "icon_category": "💻",
            "done": True
        },
        {
            "task": "Chuẩn bị interview với WQ",
            "icon_category": "🎤",
            "done": False
        },
        {
            "task": "Bán JiXinPin cho Trump địt",
            "icon_category": "🛒",
            "done": False
        },
    ]

# =========================
# ADD NEW TASK (2 columns: text + category)
# =========================
col_text, col_cat, col_btn = st.columns([5, 3, 1.2])

with col_text:
    new_task = st.text_input(
        label=" ",
        placeholder="Add to-do item...",
        label_visibility="collapsed",
        key="new_task_input"
    )

with col_cat:
    selected_cat = st.selectbox(
        label=" ",
        options=list(CATEGORIES.keys()),
        label_visibility="collapsed",
        key="new_cat_select"
    )

with col_btn:
    if st.button("➕ Add", use_container_width=True):
        if new_task.strip():
            st.session_state.todos.append({
                "task": new_task.strip(),
                "icon_category": CATEGORIES[selected_cat],
                "done": False
            })
            st.rerun()

st.divider()

# =========================
# DISPLAY TASKS (icon | checkbox | delete)
# =========================
for i, todo in enumerate(st.session_state.todos):
    col_icon, col_check, col_del = st.columns([1, 8, 1])

    with col_icon:
        # Show category icon nicely centered
        st.markdown(
            f"<div style='text-align:center; font-size:1.4rem; padding-top:4px;'>{todo['icon_category']}</div>",
            unsafe_allow_html=True
        )

    with col_check:
        todo["done"] = st.checkbox(
            todo["task"],
            value=todo["done"],
            key=f"todo_{i}"
        )

    with col_del:
        if st.button("🗑️", key=f"delete_{i}", use_container_width=True):
            st.session_state.todos.pop(i)
            st.rerun()

st.divider()

# =========================
# DELETE ALL CHECKED
# =========================
_, c, _ = st.columns([1, 1, 1])
with c:
    if st.button("🧹 Delete all checked", use_container_width=True):
        st.session_state.todos = [t for t in st.session_state.todos if not t["done"]]
        st.rerun()