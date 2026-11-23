import streamlit as st
import pandas as pd    

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="🏆Math Odyssey🚀",
    page_icon="∑",
    layout="wide"
)
st.title("🏆Math Odyssey🚀")

desc1, desc2, desc3, desc4 = st.columns(4, gap='large')
with desc1:
    st.write(":yellow[A journey through mathematics]")
with desc2:
    st.write(":green[Challenge your mathematical thinking.]")
with desc3:
    st.write(":blue[Expand your problem-solving skills.]")
with desc4:
    st.write(":red[Master the art of mathematical reasoning.]")

# ==========================================================
# CUSTOM UI CONFIGURATION
# ==========================================================
st.markdown("""
        <style>
        /* 1. FIX FONT SIZE: st.write() */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            font-size: 13px !important;
            line-height: 1.4 !important;
        }

        /* 2. FIX FONT SIZE: st.latex() */
        .katex {
            font-size: 14px !important;
        }

        /* Display equations */
        .katex-display > .katex {
            font-size: 15px !important;
        }

        /* 3. FIX SIDEBAR WIDTH */
        section[data-testid="stSidebar"] {
            width: 200px !important;
            min-width: 200px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 200px !important;
        }
        </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    topic = st.radio('**:blue[Topic]**', 
                     ['lượng giác', 'hình học', 'XSTK', 
                      'game theory', 'set theory', 
                      'ODE & PDE', 'Tích phân', 'mixed multi topic',
                      'brainteaster',
                    ])
    level = st.radio('**:red[Độ khó]**', ['Dễ', 'Trung bình', 'Khó', 'Cực khó'])

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
        with st.expander("**:green[Câu 1]**"):
            c1, c2 = st.columns([1, 2], gap='large')
            with c1:
                st.write("Tính $\\sin \\dfrac{\\pi}{12}$")
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))
                if see_answer:
                    st.write('Dễ thấy rằng $\\dfrac{\\pi}{12} = \\dfrac{\\pi}{3} - \\dfrac{\\pi}{4}$')

        with st.expander("**:green[Câu 2]**"):
            c1, c2 = st.columns([0.9, 3], gap='medium')
            with c1:
                st.write("Rút gọn biểu thức lượng giác sau:")
                st.latex("\\cos \\dfrac{\\pi}{7} \\; - \\; \\cos \\dfrac{2\\pi}{7} \\; + \\; \\cos \\dfrac{3\\pi}{7}")
                trig_key = f"see_answer_{level}_{topic}_2"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                st.write("Đặt $A = \\cos \\dfrac{\\pi}{7} \\; - \\; \\cos \\dfrac{2\\pi}{7} \\; + \\; \\cos \\dfrac{3\\pi}{7}$")
                med = st.selectbox("Để xử lý tổng này, ta có thể nhân nó với một đại lượng khác, chẳng hạn", 
                                    ["2sin (π/7)", "2sin (2π/7)", "2sin (3π/7)"])
                
                if see_answer:
                    with c2:
                        cleft, cright = st.columns([5, 6])
                        with cleft:
                            if med == "2sin (π/7)":
                                st.success("$\\begin{cases} \\cos \\alpha \\cdot \\sin \\beta = \\dfrac{1}{2} \\left( \\sin (\\alpha + \\beta) - \\sin (\\alpha - \\beta) \\right) \\\\ \\sin \\alpha \\; = \\; \\sin (\\pi - \\alpha) \\end{cases}$")
                                st.write("Ta có,")                                
                                st.latex("""
                                    \\begin{array}{rcl} 
                                        A_1 &:=& 2 \\cos \\dfrac{\\pi}{7} \\cdot \\sin \\dfrac{\\pi}{7} \\; = \\; \\sin \\dfrac{2 \\pi}{7}
                                        \\tiny{ \\qquad } \\\\ \\\\
                                        A_2 &:=& 2 \\cos \\dfrac{2 \\pi}{7} \\cdot \\sin \\dfrac{\\pi}{7} \\; = \\; \\sin \\dfrac{3\\pi}{7} - \\sin \\dfrac{\\pi}{7}
                                        \\tiny{ \\qquad } \\\\ \\\\
                                        A_3 &:=& 2 \\cos \\dfrac{3 \\pi}{7} \\cdot \\cos \\dfrac{\\pi}{7} \\; = \\; \\sin \\dfrac{4\\pi}{7} - \\sin \\dfrac{2\\pi}{7}
                                    \\end{array}
                                     """)
                            else:
                                st.success("$\\begin{cases} \\cos \\alpha \\cdot \\sin \\beta = \\dfrac{1}{2} \\left( \\sin (\\alpha + \\beta) - \\sin (\\alpha - \\beta) \\right) \\\\ \\sin \\alpha \\; = \\; \\sin (\\pi - \\alpha) \\qquad \\sin \\left(- \\alpha \\right) \\; = \\; - \\sin \\alpha \\end{cases}$")
                                st.write("Ta có,")
                                if med == "2sin (2π/7)":
                                    st.latex("""
                                        \\begin{array}{rcl} 
                                            A_1 &:=& 2 \\cos \\dfrac{\\pi}{7} \\cdot \\sin \\dfrac{2\\pi}{7} \\; = \\; \\sin \\dfrac{3 \\pi}{7} - \\sin \\left( \\dfrac{-\\pi}{7} \\right)
                                            \\tiny{ \\qquad } \\\\ \\\\
                                            A_2 &:=& 2 \\cos \\dfrac{2 \\pi}{7} \\cdot \\sin \\dfrac{2\\pi}{7} \\; = \\; \\sin \\dfrac{4\\pi}{7}
                                            \\tiny{ \\qquad } \\\\ \\\\
                                            A_3 &:=& 2 \\cos \\dfrac{3 \\pi}{7} \\cdot \\sin \\dfrac{2\\pi}{7} \\; = \\; \\sin \\dfrac{5\\pi}{7} - \\sin \\dfrac{\\pi}{7}
                                        \\end{array}
                                        """)
                                else:
                                    st.latex("""
                                        \\begin{array}{rcl} 
                                            A_1 &:=& 2 \\cos \\dfrac{\\pi}{7} \\cdot \\sin \\dfrac{3\\pi}{7} \\; = \\; \\sin \\dfrac{4 \\pi}{7} - \\sin \\left( \\dfrac{-2\\pi}{7} \\right)
                                            \\tiny{ \\qquad } \\\\ \\\\
                                            A_2 &:=& 2 \\cos \\dfrac{2 \\pi}{7} \\cdot \\sin \\dfrac{3\\pi}{7} \\; = \\; \\sin \\dfrac{5\\pi}{7} - \\sin \\left( \\dfrac{-\\pi}{7} \\right)
                                            \\tiny{ \\qquad } \\\\ \\\\
                                            A_3 &:=& 2 \\cos \\dfrac{3 \\pi}{7} \\cdot \\sin \\dfrac{3\\pi}{7} \\; = \\; \\sin \\dfrac{6\\pi}{7}
                                        \\end{array}
                                        """)
                        with cright:
                            st.write("- Rút gọn $A_1 - A_2 + A_3$, ta được")
                            if med == "2sin (π/7)":
                                st.latex("""
                                    \\begin{array}{rcl}
                                    2 A \\cdot \\sin \\dfrac{\\pi}{7} & = & \\sin \\dfrac{\\pi}{7} + \\left[ \\sin \\dfrac{4 \\pi}{7} - \\sin \\dfrac{3 \\pi}{7} \\right]
                                    \\\\ \\\\
                                            & = & \\quad \\sin \\dfrac{\\pi}{7}
                                    \\end{array}
                                    """)
                                st.write("- Mà $\\sin \\dfrac{\\pi}{7} \\neq 0,$ chia 2 về cho $\\sin \\dfrac{\\pi}{7}$,")
                            elif med == "2sin (2π/7)":
                                st.latex("""
                                    \\begin{array}{rcl}
                                    2 A \\cdot \\sin \\dfrac{\\pi}{7} & = & \\sin \\dfrac{5\\pi}{7} + \\left[ \\sin \\dfrac{3 \\pi}{7} - \\sin \\dfrac{4 \\pi}{7} \\right] - \\left[ \\sin \\dfrac{\\pi}{7} + \\sin \\dfrac{-\\pi}{7} \\right]
                                    \\\\ \\\\
                                            & = & \\quad \\sin \\dfrac{2\\pi}{7}
                                    \\end{array}
                                    """)
                                st.write("- Mà $\\sin \\dfrac{2\\pi}{7} \\neq 0,$ chia 2 về cho $\\sin \\dfrac{2\\pi}{7}$,")
                            else:
                                st.latex("""
                                    \\begin{array}{rcl}
                                    2 A \\cdot \\sin \\dfrac{\\pi}{7} & = & \\sin \\dfrac{4\\pi}{7} + \\left[ \\sin \\dfrac{2 \\pi}{7} - \\sin \\dfrac{5 \\pi}{7} \\right] - \\left[ \\sin \\dfrac{6\\pi}{7} + \\sin \\dfrac{-\\pi}{7} \\right]
                                    \\\\ \\\\
                                            & = & \\quad \\sin \\dfrac{3\\pi}{7}
                                    \\end{array}
                                    """)
                                st.write("- Mà $\\sin \\dfrac{3\\pi}{7} \\neq 0,$ chia 2 về cho $\\sin \\dfrac{3\\pi}{7}$,")

                            st.success("$\\qquad \\qquad \\qquad \\qquad \\boxed{ A \\; = \\; \\dfrac{1}{2} }$")

    elif level == "Trung bình":
        with st.expander("**:green[Câu 1]**"):
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
            
            trig_key = f"see_answer_{level}_{topic}_1"
            see_answer = st.button("Xem lời giải", key=hash(trig_key))

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

        with st.expander("**:green[Câu 2]**"):
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write("Giải phương trình lượng giác sau:")
                st.latex("\\quad \\dfrac{\\cos x \\left( 1 - 2 \\sin x \\right)}{\\left( 1 - \\sin x \\right) \\left( 1 + 2 \\sin x \\right)} \\, = \\, \\sqrt{3}")
                trig_key = f"see_answer_{level}_{topic}_2"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                if see_answer:
                    st.latex("\\text{TXĐ: } \\color{Red} \\sin x \\not \\in \\left \\lbrace -\\dfrac{1}{2}, 1 \\right \\rbrace \\qquad \\qquad")
                    st.write("$\\Leftrightarrow x \\not \\in \\left \\lbrace -\\dfrac{\\pi}{6} + 2k\\pi, \\dfrac{\\pi}{2} + 2k\\pi \\right \\rbrace, k \\in \\mathbb{Z}$")
                    with c2:
                        cleft, cright = st.columns([3, 2], gap='medium')
                        with cleft:
                            st.write("- PT đã cho tương đương với:")
                            st.latex("""
                                \\begin{array}{lcl}
                                \\quad \\left( 1 - 2 \\sin x \\right) \\cos x &=& \\sqrt{3} \\left( 1 + 2 \\sin x \\right) \\left( 1 - \\sin x \\right) \\\\
                                    \\qquad \\color{Cyan} \\tiny{(\\text{vì}  \\sin 2x = 2 \\sin x \\cos x )} \\\\
                                \\Rightarrow \\quad \\cos x - \\sin 2x &=& \\sqrt{3} \\left( 1 - 2 \\sin^2 x + 2 \\sin x - \\sin x \\right) \\\\
                                   & & \\color{Cyan} \\qquad \\tiny{(\\text{vì}  \\cos 2x = 1 - 2 \\sin^2 x )} \\\\
                                \\Rightarrow \\quad \\cos x - \\sqrt{3} \\sin x &=&  \\sin 2x + \\sqrt{3} \\cos 2x \\\\
                                \\\\
                                \\Rightarrow \\quad 2 \\cos \\left( x + \\dfrac{\\pi}{3} \\right) &=& 2 \\cos \\left( 2x - \\dfrac{\\pi}{6} \\right) \\\\
                                \\end{array}
                            """)
                            st.write("- Vậy, $\\left( x + \\dfrac{\\pi}{3} \\right) = \\pm \\left( 2x - \\dfrac{\\pi}{6} \\right) + 2k\\pi, k \\in \\mathbb{Z}$")

                    with cright:
                        st.write("- TH1: $x + \\dfrac{\\pi}{3} = 2x - \\dfrac{\\pi}{6} + 2k\\pi$")
                        st.latex("\\Rightarrow x = \\dfrac{\\pi}{2} + 2k\\pi, k \\in \\mathbb{Z} \\quad (\\color{Red} \\text{loại})")
                        st.write("- TH2: $x + \\dfrac{\\pi}{3} = -2x + \\dfrac{\\pi}{6} + 2k\\pi$")
                        st.latex("\\Rightarrow \\boxed{x = -\\dfrac{\\pi}{6} + \\dfrac{2k\\pi}{3}, k \\in \\mathbb{Z}}")

    elif level == "Khó":
        with st.expander("**:green[Câu 1]**"):
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write("Tính: $\\sin \\dfrac{\\pi}{10}$")
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                if see_answer:
                    st.write("Các công thức mà ta sẽ cần ở bài này gồm:")
                    st.success("$\\boxed{ \\begin{cases} \\sin x \\, = \\, \\cos \\left( \\dfrac{\\pi}{2} - x \\right) \\\\ \\sin 2 \\alpha \\, = \\, 2 \\sin \\alpha \\cdot \\cos \\alpha \\\\ \\cos 3 \\beta = 4 \\cos^3 \\beta - 3 \\cos \\beta \\end{cases} }$")
                    with c2:
                        cleft, cright = st.columns([2, 3], gap='small')
                        with cleft:
                            st.write("- Đầu tiên, ta có")
                            st.write("$\\qquad \\qquad \\qquad \\sin \\dfrac{2\\pi}{10} \\, = \\, \\cos \\dfrac{3\\pi}{10}$")
                            st.write('- Áp dụng các công thức nhân đôi và nhân ba,')
                            st.write("$\\qquad \\quad 2 \\sin \\dfrac{\\pi}{10} \\cos \\dfrac{\\pi}{10} \\, = \\, 4 \\cos^3 \\dfrac{\\pi}{10} - 3 \\cos \\dfrac{\\pi}{10}$")
                            st.write("dẫn đến")
                            st.latex("\\cos \\dfrac{\\pi}{10} \\cdot \\left[ 4 \\cos^2 \\dfrac{\\pi}{10} - 2 \\sin \\dfrac{\\pi}{10} - 3 \\right] = 0")
                        with cright:
                            st.write("- Dễ thấy $\\cos \\dfrac{\\pi}{10} \\neq 0$ nên $\\boxed{4 \\cos^2 \\dfrac{\\pi}{10} - 2 \\sin \\dfrac{\\pi}{10} - 3 = 0}$ ")
                            st.write("- Đặt $t = \\sin \\dfrac{\\pi}{10}$ với điều kiện $t \\in (0, 1)$ vì $\\dfrac{\\pi}{10} \\in \\left(0, \\dfrac{\\pi}{2} \\right),$ phương trình trên trở thành")
                            st.latex('4 (1 - t^2) - 2 t - 3 = 0 \\; \\Rightarrow \\; \\boxed{4t^2 + 2t - 1 = 0}')
                            st.write("- Lúc này, ta được $t = \\dfrac{-1 \\pm \\sqrt{5} }{4}$, và kết hợp với điều kiện $t \\in (0, 1)$, ta được")
                            st.latex(" \\boxed{\\sin \\dfrac{\\pi}{10} \\, := t \\, = \\dfrac{-1 + \\sqrt{5}}{4}}")

        with st.expander("**:green[Câu 2]**"):
            c1, c2 = st.columns([1, 2.5], gap='medium')
            with c1:
                st.write("Tính: $\\cos \\dfrac{\\pi}{15}$")
                trig_key = f"see_answer_{level}_{topic}_2"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                if see_answer:
                    st.write('- Dễ thấy rằng $\\dfrac{\\pi}{15} \\; = \\; \\dfrac{\\pi}{6} \\, - \\, \\dfrac{\\pi}{10}$, ngoài ra')
                    st.success('$\\sin \\left( \\alpha - \\beta \\right) = \\sin \\alpha \\cdot \\cos \\beta \\; - \\; \\cos \\alpha \\cdot \\sin \\beta$')
                    with c2:
                        cleft, cright = st.columns([5, 8])
                        with cleft:
                            st.write('- Ở câu 1 trên, ta đã tính được')
                            st.write('$\\qquad \\boxed{ \\sin \\dfrac{\\pi}{10} \\, = \\, \\dfrac{-1 + \\sqrt{5}}{4}}$')
                            st.write('do đó')
                            st.write('$\\qquad \\begin{array}{rcl} \\cos \\dfrac{\\pi}{10} &=& \\sqrt{1 - \\sin^2 \\dfrac{\\pi}{10}} \\\\ \\\\ &=& \\sqrt{1 - \\left( \\dfrac{\\sqrt{5} - 1}{4} \\right)^2} \\\\ \\\\ &=& \\sqrt{ \\dfrac{5 + \\sqrt{5}}{8} } \\end{array}$')
                        with cright:
                            st.write('- Các giá trị lượng giác của $\\dfrac{\\pi}{6}$ đều đã được biết đến với $\\cos \\dfrac{\\pi}{6} = \\dfrac{\\sqrt{3}}{2}$ và $\\sin \\dfrac{\\pi}{6} = \\dfrac{1}{2}$, do đó')
                            st.latex('''
                                \\begin{array}{rcl}
                                \\sin \\dfrac{\\pi}{15} &=& \\sin \\dfrac{\\pi}{6} \\cdot \\cos \\dfrac{\\pi}{10} - \\cos \\dfrac{\\pi}{6} \\cdot \\sin \\dfrac{\\pi}{10} 
                                \\\\ \\\\
                                    & = & \\dfrac{1}{2} \\cdot \\dfrac{ \\sqrt{5 + \\sqrt{5}} }{2 \\sqrt{2} } \\; - \\; \\dfrac{\\sqrt{3}}{2} \\cdot \\left( \\dfrac{\\sqrt{5} - 1}{4} \\right)
                                \\\\ \\\\
                                    & = & \\dfrac{1}{4} \\left[ \\sqrt{ \\dfrac{5 + \\sqrt{5}}{2} } \\; - \\; \\dfrac{\\sqrt{3} \\left( \\sqrt{5} - 1 \\right) }{2} \\right]
                                \\\\ \\\\
                                    & = & \\boxed{ \\dfrac{1}{4} \\sqrt{ 7 - 5 \\sqrt{5} - \\sqrt{6 (5 - \\sqrt{5} ) } } }
                                \\end{array}
                                ''')

    elif level == "Cực khó":
        with st.expander("**:green[Câu 1]**"):
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write("Tính: $\\sin^{-1} \\left( \\cos \\dfrac{43 \\pi}{30} \\right)$")
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                if see_answer:
                    st.write('Lưu ý rằng')
                    st.success("$\\boxed{ \\begin{array}{ccl} \\cos (\\pi + \\alpha) &=& - \\cos \\alpha \\\\ \\cos \\beta &=& \\sin \\left( \\dfrac{\\pi}{2} - \\beta \\right) \\end{array} }$")
                    with c2:
                        cleft, cright = st.columns(2)
                        with cleft:
                            st.write('Ta có')
                            st.write('$\\sin $')