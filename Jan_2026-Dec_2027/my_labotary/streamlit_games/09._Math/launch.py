import streamlit as st
import pandas as pd    # dùng để in ra bảng biểu suất / phí

with st.sidebar:
    topic = st.radio('**:red[Topic]**', 
                     ['lượng giác', 'hình học', 'xstk', 
                      'game theory', 'set theory', 
                      'ode', 'pde', 'mixed multi topic',
                      'brainteaster',
                    ])
    level = st.radio('Độ khó', ['Dễ', 'Trung bình', 'Khó', 'Cực khó'])    

# Main app
# Đẩy các câu hỏi folder có sẵn từ repo này về https://github.com/NhanDoV/WorldQuant-math-quiz/blob/main/Math-test/MathWQ_summary/libs/home.py
# Tổ chức lại folder theo Path để chứa relative các hình mô phỏng, cái nào done thì kéo về
st.set_page_config(page_title="Math Cheater", page_icon="⚡", layout="wide")

# Lưu ý rằng mỗi câu sẽ có format định dạng chia cột / hình bố trí khác nhau nên ta không ưu tiên tạo dict tổng hợp
if topic == "sth":
    if level == "Dễ":
        with st.expander("Câu 1"):
            # Bố trí trục dọc
            st.write('Đề bài here')
            st.write('Đáp án here')

        with st.expander("Câu 2"):
            # Bố trí ngang
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write('Đề bài here')
            with c2:
                st.write('Đáp án here')
    
    # tương tự cho các level khác

if topic == "lượng giác":
    if level == "Dễ":
        with st.expander("Câu 1"):
            # Bố trí ngang
            st.write('Trong các phát biểu sau, phát biểu nào là đúng?')
            options = ["$A. \\; \\tan(\\pi) > \\sin(\\pi)$", 
                       "$B. \\; \\sin\\left( 1 / 10000 \\right) > \\cos\\left( 1 / 10000 \\right)$", 
                       "$C. \\; 3^{3^{100000}} > 4^{2^{100000}}$", 
                       "$D. \\; \\log_{2^{1000}}\\left( 3^{100} \\right) < \\log_{3^{10000}}\\left( 2^{1000} \\right)$", 
                       "$E.$ Tất cả đều SAI"]
            columns = st.columns([0.9, 1.5, 0.8, 1.5, 0.9], gap='medium')
            for idx, option in enumerate(options):
                col = columns[idx]
                with col:
                    st.write(option)                
            
            see_answer = st.button("Xem lời giải")
            if see_answer:
                cleft, cright = st.columns([3, 2], gap='medium')
                with cleft:
                    st.write("""
                - Dễ dàng thấy rằng :red[$A$] và :red[$B$] là các đáp án :red[SAI], bởi $\\begin{cases} \\tan \\pi = 0 \\\\ \\sin \\pi = 0 \\end{cases} \\text{ and } \\begin{cases} \\sin(1/10000) & \\to & 0 \\\\ \\cos(1 / 10000) & \\to & 1 \\end{cases}$
                
                - **:green[$C$ ĐÚNG]**. Thật vậy, ta thực hiện **:blue[logarithm]** liên tiếp 2 lần cả hai vế để so sánh, ta có

                $\\qquad \\begin{array}{ccl} \\ln \\left( \\ln  3^{\\left( 3^{100000} \\right)} \\right) &=& 100000 \\cdot (\\ln 3) + \\ln(\\ln 3) \\\\ & > & 100000 \\cdot (\\ln 2) + \\ln(\\ln 4) \\, = \\, \\ln \\left( \\ln  4^{\\left( 2^{100000} \\right)} \\right) \\end{array}$

                Vậy, :green[$3^{3^{100000}} > 4^{2^{100000}}$]
                    """)
                with cright:
                    st.write("""
                - :red[$D$ SAI]. Lưu ý rằng :orange[$\\log_a b = \\ln b / \\ln a$], do đó

                $ \\left. \\begin{array}{c} \\ln_{\\left( 2^{1000} \\right)}(3^{100}) &=& \\dfrac{100 \\cdot \\ln 3}{1000 \\cdot \\ln 2} &=& \\dfrac{1}{10} \\cdot \\left( \\log_2 3 \\right) \\\\ \\\\ \\ln_{\\left( 3^{10000} \\right)}(2^{1000}) &=& \\dfrac{1000 \\cdot \\ln 2}{10000 \\cdot \\ln 3} &=& \\dfrac{1}{10} \\cdot \\left( \\log_3 2 \\right) \\end{array} \\right \\rbrace $

                Mà $\\left( \\log_3 2 \\right) \\; < \\; 1 \\; < \\; \\left( \\log_2 3 \\right)$

                $ \\Rightarrow \\quad \\ln_{\\left( 2^{1000} \\right)}(3^{100}) > \\ln_{\\left( 3^{10000} \\right)}(2^{1000}) $
                """)

        with st.expander("Câu 2"):
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write("Giải phương trình lượng giác sau:")
                st.latex("\\quad \\dfrac{\\cos x \\left( 1 - 2 \\sin x \\right)}{\\left( 1 - \\sin x \\right) \\left( 1 + 2 \\sin x \\right)} \\, = \\, \\sqrt{3}")
            with c2:
                st.write('Đáp án here')
