import streamlit as st
import pandas as pd    # dùng để in ra bảng biểu suất / phí

with st.sidebar:
    topic = st.radio('**:red[Kiểu kiểm định]**', 
                     ['z-test', 't-test', 'var-test', 'distribution-test'])
    alpha = st.number_input('Mức ý nghĩa')
    alter = st.radio("**Kiểu đối thuyết**",
                     ['two-sided', 'left', 'right'])

# Main app
st.title(':rainbow[Statistical Testing Hypothesis]')
st.set_page_config(page_title="Statistical Testing Hypothesis", page_icon="⚡", layout="wide")

c1, c2 = st.columns([2, 5], gap='large')
with c1:
    # sử dụng if-else logic cho mean test, prop test, var test hoặc diff test
    val = st.number_input('Input sample value ()')

    with st.expander('Final conclusion'):
        st.success('Tell sth here if not evidence to reject H0')
        st.warning('Tell sth here if reject H0')

with c2:
    with st.expander('Descriptuve stats explanation table'):
        st.write(
            """
                H0 : problem statement
                H1 : 
                p_value : conclusion if
                acceptance : formulation
                Confidence interval:formulation
            """
        )