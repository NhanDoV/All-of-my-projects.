import streamlit as st

st.set_page_config(page_title="Math MCQ Repo", layout="wide")

home, by_topic, by_mock = st.tabs(("**HOME**", "**Topics**", "**Mock Exam**"))

with home:
    st.subheader("📘 Welcome")
    st.write(
        """
        This app collects answers to multiple-choice questions from several high school math exams 
        (in a style close to GMAT).  
        You can explore the questions by topic or try full mock exam sets.
        """
    )

with by_topic:
    st.subheader("📂 Browse by Topic")
    st.write(
        """
        Select a topic below to review questions and answers:
        - **Algebra**  
        - **Sequences**  
        - **Statistics**  
        """
    )
    # (Here you can later add st.selectbox or st.radio for topic selection)

with by_mock:
    st.subheader("📝 Mock Exams")
    st.write(
        """
        Explore full sets of mock exams.  
        Perfect if you want to practice under test-like conditions.
        """
    )
    # (Here you can later add options for mock exam sets)