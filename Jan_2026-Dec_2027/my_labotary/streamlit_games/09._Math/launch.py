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
    indicated_topic = ['Lượng giác', 'Hình học', 'XSTK', 'Ma trận',
                        'Game theory', 'Set theory', 'Dãy & chuỗi số',
                        'ODE & PDE', 'Tích phân',
                        'Giới hạn', 'Brainteaster', 'zMixed multipletopics'
                        ]
    topic = st.radio('**:blue[Topic]**', sorted(indicated_topic))
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

if topic == "Lượng giác":
    if level == "Dễ":
        with st.expander("**:green[Câu 1]**"):
            c1, c2 = st.columns([1, 2.5], gap='large')
            with c1:
                st.write("Tính $\\sin \\dfrac{\\pi}{12}$")
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))
                if see_answer:
                    st.write('Dễ thấy rằng $\\dfrac{\\pi}{12} = \\dfrac{\\pi}{3} - \\dfrac{\\pi}{4}$ và')
                    st.success('$\\; \\boxed{\\sin (\\alpha \\pm \\beta) \\, = \\, \\sin \\alpha \\cdot \\cos \\beta \\; \\pm \\; \\cos \\alpha \\cdot \\sin \\beta}$')

                    with c2:
                        cleft, cright = st.columns([2, 3], gap='medium')
                        with cleft:
                            st.write('Ngoài ra, ta đã biết')
                            st.latex('''
                                    \\begin{array}{rcl}
                                    \\sin \\dfrac{\\pi}{3} = \\dfrac{\\sqrt{3}}{2} 
                                    & , & \\cos \\dfrac{\\pi}{3} = \\dfrac{1}{2} 
                                    \\\\ 
                                    \\\\
                                    \\sin \\dfrac{\\pi}{4} = \\dfrac{\\sqrt{2}}{2} 
                                    & , & \\cos \\dfrac{\\pi}{4} = \\dfrac{\\sqrt{2}}{2}
                                    \\end{array}
                                    ''')
                        with cright:
                            st.write('Do đó,')
                            st.write('''$\\qquad \\begin{array}{rcl} \\sin \\dfrac{\\pi}{12} \\; = \\; \\sin \\left( \\dfrac{\\pi}{3} - \\dfrac{\\pi}{4} \\right) &=& \\sin \\dfrac{\\pi}{3} \\cdot \\cos \\dfrac{\\pi}{4} \\; - \\; \\cos \\dfrac{\\pi}{3} \\cdot \\sin \\dfrac{\\pi}{4} \\\\ \\\\ &=& \\quad \\dfrac{3}{2} \\cdot \\dfrac{\\sqrt{2}}{2} \\; - \\; \\dfrac{1}{2} \\cdot \\dfrac{\\sqrt{2}}{2} \\\\ \\\\ &=& \\color{Green} \\boxed{ \\dfrac{ \\sqrt{6} - \\sqrt{2} }{4} } \\end{array}$''')

        with st.expander("**:green[Câu 2]**"):
            c1, c2 = st.columns([0.9, 3], gap='medium')
            with c1:
                st.write("Rút gọn biểu thức lượng giác sau:")
                st.latex("\\cos \\dfrac{\\pi}{7} \\; - \\; \\cos \\dfrac{2\\pi}{7} \\; + \\; \\cos \\dfrac{3\\pi}{7}")
                trig_key = f"see_answer_{level}_{topic}_2"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                st.write("Đặt $\\color{Blue} \\boxed{A = \\cos \\dfrac{\\pi}{7} \\; - \\; \\cos \\dfrac{2\\pi}{7} \\; + \\; \\cos \\dfrac{3\\pi}{7}}$")
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
                                    2 A \\cdot \\sin \\dfrac{2\\pi}{7} & = & \\sin \\dfrac{5\\pi}{7} + \\left[ \\sin \\dfrac{3 \\pi}{7} - \\sin \\dfrac{4 \\pi}{7} \\right] - \\left[ \\sin \\dfrac{\\pi}{7} + \\sin \\dfrac{-\\pi}{7} \\right]
                                    \\\\ \\\\
                                            & = & \\quad \\sin \\dfrac{2\\pi}{7}
                                    \\end{array}
                                    """)
                                st.write("- Mà $\\sin \\dfrac{2\\pi}{7} \\neq 0,$ chia 2 về cho $\\sin \\dfrac{2\\pi}{7}$,")
                            else:
                                st.latex("""
                                    \\begin{array}{rcl}
                                    2 A \\cdot \\sin \\dfrac{3\\pi}{7} & = & \\sin \\dfrac{4\\pi}{7} + \\left[ \\sin \\dfrac{2 \\pi}{7} - \\sin \\dfrac{5 \\pi}{7} \\right] - \\left[ \\sin \\dfrac{6\\pi}{7} + \\sin \\dfrac{-\\pi}{7} \\right]
                                    \\\\ \\\\
                                            & = & \\quad \\sin \\dfrac{3\\pi}{7}
                                    \\end{array}
                                    """)
                                st.write("- Mà $\\sin \\dfrac{3\\pi}{7} \\neq 0,$ chia 2 về cho $\\sin \\dfrac{3\\pi}{7}$,")

                            st.success("$\\qquad \\qquad \\qquad \\qquad \\boxed{ A \\; = \\; \\dfrac{1}{2} }$")

        with st.expander("**:green[Câu 3]**"):
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write("Tính: $\\sin^{-1} \\left( \\cos \\dfrac{43 \\pi}{30} \\right)$")
                trig_key = f"see_answer_{level}_{topic}_3"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                if see_answer:
                    st.write('Lưu ý rằng')
                    st.success("$\\boxed{ \\begin{array}{ccl} \\cos (\\pi + \\alpha) &=& - \\cos \\alpha \\\\ \\cos \\beta &=& \\sin \\left( \\dfrac{\\pi}{2} - \\beta \\right) \\end{array} }$")
                    with c2:
                        cleft, cright = st.columns([2, 3], gap='medium')
                        with cleft:
                            st.write('Ta có')
                            st.write('$\\qquad \\begin{array}{rcl} \\cos \\dfrac{43 \\pi}{30} &=& \\cos \\left( \\pi + \\dfrac{13 \\pi}{30} \\right)  \\\\ \\\\ &=& - \\cos \\dfrac{13 \\pi}{30} \\end{array}$')
                            st.write('và,')
                            st.write('$\\qquad \\begin{array}{rcl} \\cos \\dfrac{13 \\pi}{30} &=& \\sin \\left( \\dfrac{\\pi}{2} - \\dfrac{13 \\pi}{30} \\right)  \\\\ \\\\ &=& \\quad \\sin \\dfrac{\\pi}{15} \\end{array}$')
                        with cright:
                            st.write('Suy ra')
                            st.latex('''
                                \\begin{array}{rcl} 
                                    \\sin^{-1} \\left( \\cos \\dfrac{43 \\pi}{30} \\right) &=& \\sin^{-1} \\left( - \\cos \\dfrac{13 \\pi}{30} \\right) 
                                    \\\\ \\\\
                                    & = & - \\sin^{-1} \\left( \\cos \\dfrac{13 \\pi}{30} \\right) \\\\ \\\\
                                    & = & - \\sin^{-1} \\left( \\sin \\dfrac{\\pi}{15} \\right) \\\\ \\\\
                                    & = & \\color{Green} \\boxed{- \\dfrac{\\pi}{15}}
                                \\end{array}
                            ''')

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
                                    \\qquad \\color{Blue} \\tiny{(\\text{vì}  \\sin 2x = 2 \\sin x \\cos x )} \\\\
                                \\Rightarrow \\quad \\cos x - \\sin 2x &=& \\sqrt{3} \\left( 1 - 2 \\sin^2 x + 2 \\sin x - \\sin x \\right) \\\\
                                   & & \\color{Blue} \\qquad \\tiny{(\\text{vì}  \\cos 2x = 1 - 2 \\sin^2 x )} \\\\
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
                        st.latex("\\Rightarrow \\color{Green} \\boxed{x = -\\dfrac{\\pi}{6} + \\dfrac{2k\\pi}{3}, k \\in \\mathbb{Z}}")

        with st.expander("**:green[Câu 3]**"):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write('Tìm $x \\in [0, 2 \\pi]$ biết rằng')
                st.write("$\\qquad \\quad \\arccos ( \\cos x ) = \\dfrac{x}{2}$")
                _, click, _ = st.columns([2, 4, 1])
                with click:
                    trig_key = f"see_answer_{level}_{topic}_3"
                    see_answer = st.button("Xem lời giải", key=hash(trig_key))

                if see_answer:
                    with c2:
                        cleft, cmid, cright = st.columns([6,5,6], gap='medium')
                        with cleft:
                            st.write('Nhắc lại định lý hàm ngược của $\\cos$ với $\\arccos$,')
                            st.success('$\\arccos ( \\cos \\alpha ) = \\begin{cases} \\alpha \\qquad \\text{ if } \\alpha \\in [0, \\pi) \\\\ \\\\ 2\\pi - \\alpha \\quad \\alpha \\in [\\pi, 2\\pi] \\end{cases}$')
                        with cmid:
                            st.write('- TH 1: $x \\in [0, \\pi],$ ta có')
                            st.write('$\\qquad x \\, = \\, \\dfrac{x}{2} \\; \\Rightarrow \\; x \\, = \\, 0$')
                            st.write('- TH 2: $x \\in [\\pi, 2\\pi],$ ta có')
                            st.write('$\\qquad 2\\pi - x \\, = \\, \\dfrac{x}{2} \\; \\Rightarrow \\; x \\, = \\, \\dfrac{4\\pi}{3}$')
                        with cright:
                            st.write('Cả 2 nghiệm đều thỏa điều kiện xác định, do đó')
                            st.success('$\\qquad \\boxed{x \\in \\left \\lbrace 0, \\dfrac{4\\pi}{3} \\right \\rbrace}$')

    elif level == "Khó":
        with st.expander("**:green[Câu 1]**"):
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write("Tính: $\\sin \\dfrac{\\pi}{10}$")
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

                if see_answer:
                    st.write("Các công thức mà ta sẽ cần ở bài này gồm:")
                    st.success("$\\boxed{ \\begin{array}{rcl} \\sin x &=& \\cos \\left( \\dfrac{\\pi}{2} - x \\right) \\\\ \\\\ \\sin 2 \\alpha &=& 2 \\sin \\alpha \\cdot \\cos \\alpha \\\\ \\\\ \\cos 3 \\beta &=& 4 \\cos^3 \\beta - 3 \\cos \\beta \\end{array} }$")
                    with c2:
                        cleft, cright = st.columns([2, 3], gap='small')
                        with cleft:
                            st.write("- Đầu tiên, ta có")
                            st.write("$\\qquad \\qquad \\qquad \\sin \\dfrac{2\\pi}{10} \\, = \\, \\cos \\dfrac{3\\pi}{10}$")
                            st.write('- Áp dụng lần lượt các công thức :red[nhân đôi] và :blue[nhân ba] cho vế trái và vế phải,')
                            st.write("$\\qquad \\quad \\begin{array}{rcl} \\color{Red}{2 \\sin \\dfrac{\\pi}{10} \\cos \\dfrac{\\pi}{10}} & = & \\color{Cyan}{4 \\cos^3 \\dfrac{\\pi}{10} - 3 \\cos \\dfrac{\\pi}{10}} \\end{array}$")
                            st.write("dẫn đến")
                            st.latex("\\cos \\dfrac{\\pi}{10} \\cdot \\left[ 4 \\cos^2 \\dfrac{\\pi}{10} - 2 \\sin \\dfrac{\\pi}{10} - 3 \\right] = 0")
                        with cright:
                            st.write("- Dễ thấy $\\cos \\dfrac{\\pi}{10} \\neq 0$ nên $\\boxed{4 \\cos^2 \\dfrac{\\pi}{10} - 2 \\sin \\dfrac{\\pi}{10} - 3 = 0}$ ")
                            st.write("- Đặt $t = \\sin \\dfrac{\\pi}{10}$ với điều kiện $t \\in (0, 1)$ vì $\\dfrac{\\pi}{10} \\in \\left(0, \\dfrac{\\pi}{2} \\right),$ phương trình trên trở thành")
                            st.latex('4 (1 - t^2) - 2 t - 3 = 0 \\; \\Rightarrow \\; \\boxed{4t^2 + 2t - 1 = 0}')
                            st.write("- Lúc này, ta được $t = \\dfrac{-1 \\pm \\sqrt{5} }{4}$, và kết hợp với điều kiện $t \\in (0, 1)$, ta được")
                            st.latex("\\color{Green} \\boxed{\\sin \\dfrac{\\pi}{10} \\, := t \\, = \\dfrac{-1 + \\sqrt{5}}{4}}")

        with st.expander("**:green[Câu 2]**"):
            c1, c2 = st.columns([1, 2.5], gap='medium')
            with c1:
                st.write("Tính: $\\cos \\dfrac{\\pi}{15}$")
                trig_key_2 = f"see_answer_{level}_{topic}_2"
                see_answer_2 = st.button("Xem lời giải", key=hash(trig_key_2))

                if see_answer_2:
                    st.write('- Dễ thấy rằng $\\dfrac{\\pi}{15} \\; = \\; \\dfrac{\\pi}{6} \\, - \\, \\dfrac{\\pi}{10}$, ngoài ra')
                    st.success('$\\sin \\left( \\alpha - \\beta \\right) = \\sin \\alpha \\cdot \\cos \\beta \\; - \\; \\cos \\alpha \\cdot \\sin \\beta$')
                    with c2:
                        cleft, cright = st.columns([5, 9])
                        with cleft:
                            st.write('- Ở câu 1 trên, ta đã tính được')
                            st.write('$\\qquad \\boxed{ \\sin \\dfrac{\\pi}{10} \\, = \\, \\dfrac{-1 + \\sqrt{5}}{4}}$')
                            st.write('do đó')
                            st.write('$\\qquad \\begin{array}{rcl} \\cos \\dfrac{\\pi}{10} &=& \\sqrt{1 - \\sin^2 \\dfrac{\\pi}{10}} \\\\ \\\\ &=& \\sqrt{1 - \\left( \\dfrac{\\sqrt{5} - 1}{4} \\right)^2} \\\\ \\\\ &=& \\sqrt{ \\dfrac{5 + \\sqrt{5}}{8} } \\end{array}$')
                        with cright:
                            st.write('- Các giá trị lượng giác của $\\dfrac{\\pi}{6}$ đều đã được biết đến với $\\color{Red} \\cos \\dfrac{\\pi}{6} = \\dfrac{\\sqrt{3}}{2}$ và $\\color{Red} \\sin \\dfrac{\\pi}{6} = \\dfrac{1}{2}$, do đó')
                            st.latex('''
                                \\begin{array}{rcl}
                                \\sin \\dfrac{\\pi}{15} &=& \\sin \\dfrac{\\pi}{6} \\cdot \\cos \\dfrac{\\pi}{10} - \\cos \\dfrac{\\pi}{6} \\cdot \\sin \\dfrac{\\pi}{10} 
                                \\\\ \\\\
                                    & = & \\dfrac{1}{2} \\cdot \\dfrac{ \\sqrt{5 + \\sqrt{5}} }{2 \\sqrt{2} } \\; - \\; \\dfrac{\\sqrt{3}}{2} \\cdot \\left( \\dfrac{\\sqrt{5} - 1}{4} \\right)
                                \\\\ \\\\
                                    & = & \\dfrac{1}{4} \\left[ \\sqrt{ \\dfrac{5 + \\sqrt{5}}{2} } \\; - \\; \\dfrac{\\sqrt{3} \\left( \\sqrt{5} - 1 \\right) }{2} \\right]
                                \\\\ \\\\
                                    & = & \\color{Green} \\boxed{ \\dfrac{1}{4} \\sqrt{ 7 - 5 \\sqrt{5} - \\sqrt{6 (5 - \\sqrt{5} ) } } }
                                \\end{array}
                                ''')

    elif level == "Cực khó":
       with st.expander("**:green[Câu 1]**"):
            c1, c2 = st.columns([1, 3], gap='medium')
            with c1:
                st.write("CMR:")
                st.write("$\\qquad \\displaystyle \\sin \\left( \\sum_{i=1}^n w_i \\alpha_i \\right) \\; \\geq \\; \\sum_{i=1}^n w_i \\sin \\alpha_i$")
                st.write('trong đó $\\alpha_i \\in (0, \\pi) \\; \\forall i$ và $w_i \\in (0, 1)$ thỏa')
                st.write('$\\qquad \\qquad \\qquad \\displaystyle \\sum_{i=1}^n \\alpha_i \\, = \\, 1$')
                st.write('Từ đó, xác định GTLN của')
                st.write('$\\qquad \\qquad \\sin A + \\sin B + \\sin C$')
                st.write('với $A, B, C$ là $3$ góc của một tam giác bất kỳ.')
                _, click, _ = st.columns([2, 4, 1])
                with click:
                    trig_key = f"see_answer_{level}_{topic}_1"
                    see_answer = st.button("Xem lời giải", key=hash(trig_key))
            with c2:
                cleft, cright = st.columns(2)
                if see_answer:
                    with cleft:
                        st.write("**:green[Solution]**")
                        st.write('- Đặt $f(x) = \\sin x$ và ta sẽ chứng minh $f$ là **:blue[concave]** trên miền $\\color{Blue} (0, \\pi)$')
                        st.write('$\\qquad$ Thật vậy')
                        st.write('$\\qquad \\qquad \\dfrac{\\partial^2 f}{\\partial x^2} \\, = \\, - \\sin x \\; < \\; 0 \\quad \\forall x \\in (0, \\pi)$')
                        st.write('- Áp dụng BĐT Jensen cho :blue[hàm concave] trên $(0, \\pi)$, ta được')
                        st.write("$\\qquad \\qquad  \\qquad \\boxed{\\displaystyle \\sin \\left( \\sum_{i=1}^n w_i \\alpha_i \\right) \\; \\geq \\; \\sum_{i=1}^n w_i \\sin \\alpha_i}$")
                        st.write('trong đó $\\alpha_i \\in (0, \\pi) , \\quad w_i \\in (0, 1) \\qquad \\forall i = \\overline{1, \\ldots, n}\\quad$ và')
                        st.write('$\\qquad \\qquad \\qquad \\qquad \\qquad \\displaystyle \\sum_{i=1}^n \\alpha_i \\, = \\, 1$')
                    with cright:
                        st.write('- Thay $n = 3$ và $w_1 = w_2 = w_3 = \\tfrac{1}{3}$, khi đó áp dụng BĐT Jensen và tổng 3 góc trong một tam giác, ta được')
                        st.latex('''
                                    \\begin{array}{rcl}
                                        \\dfrac{1}{3} \\left( \\sin A + \\sin B + \\sin C \\right) & \\leq & \\sin \\left( \\dfrac{A + B + C}{3} \\right)
                                        \\\\ \\\\
                                        & = & \\quad \\sin \\dfrac{\\pi}{3} \\quad = \\quad \\dfrac{\\sqrt{3}}{2}
                                    \\end{array}
                                ''')
                        st.write('do đó, ')
                        st.write('$\\qquad \\qquad \\qquad \\color{Green} \\boxed{\\sin A + \\sin B + \\sin C \\; \\leq \\; \\dfrac{3 \\sqrt{3}}{2}}$')

elif topic == "Giới hạn":
    if level == 'Dễ':
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns(2)
            with c1:
                st.write('Tính $\\displaystyle \\lim_{n \\to \\infty} \\left( \\sqrt{n^2 + 1} - n \\right)$')
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))


        with st.expander('**:green[Câu 2]**'):
            st.write('Cho số nguyên $k > 2$, hỏi chuỗi nào trong các chuỗi sau đây sẽ triệt tiêu về $0$ khi $k \\to \\infty$')
            col_ans = st.columns(4)
            options = ["$a_k = ((2k + 1) / (2k + 2))^k$", 
                       "$b_k = (2k - 1)^k$", 
                       "$c_k = (2k + 1)^k$", 
                       "$d_k = ((2k - 1) / (2k - 2))^k$"]
            for idx, option in enumerate(col_ans):
                col = col_ans[idx]
                with col:
                    st.write(options[idx])

            trig_key = f"see_answer_{level}_{topic}_2"
            see_answer = st.button("Xem lời giải", key=hash(trig_key))


    elif level == "Trung bình":
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns(2)
            with c1:
                st.write('Tính $\\displaystyle \\lim_{n \\to \\infty} \\left( \\dfrac{n-3}{n+3} \\right)^n$')
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

        with st.expander('**:green[Câu 2]**'):
            c1, c2 = st.columns(2)
            with c1:
                st.write('Tính $\\displaystyle \\lim_{x \\to 0} \\dfrac{\\exp({x^2}) - \\cos x}{x^2}$')
                trig_key = f"see_answer_{level}_{topic}_2"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

        with st.expander('**:green[Câu 3]**'):
            c1, c2 = st.columns(2)
            with c1:
                st.write('Tính $\\displaystyle \\lim_{x \\to \\infty} \\dfrac{\\sin (\\log x)}{\\log x}$')
                trig_key = f"see_answer_{level}_{topic}_3"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

        with st.expander('**:green[Câu 4]**'):
            c1, c2 = st.columns(2)
            with c1:
                st.write('Tính $\\displaystyle \\lim_{x \\to 0} \\dfrac{\\sin x - \\tan x}{x^3}$')
                trig_key = f"see_answer_{level}_{topic}_4"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

elif topic == "XSTK":
    if level == "Dễ":
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write('The three distinct entries of a `2 x 2 symmetric matrix` are drawn from the `uniform distribution` [-60, 60]. What is the `expected determinant` of the matrix?')
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))
            
        with st.expander('**:green[Câu 2]**'):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write('Two points are chosen, uniformly and independently distributed, on a line of length 60. What is average (expected) distance between them?')
                trig_key = f"see_answer_{level}_{topic}_2"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

    elif level == 'Trung bình':
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write("Triangle $ABC$ has sides of length 45, 60, and 75. Place a point $D$ randomly and uniformly inside the triangle. What is the expected value of the sum of perpendicular distances from point D to the triangle's three sides?")
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))                

elif topic == 'Hình học':
    if level == "Dễ":
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write('Let $\\Omega$ be a square-base prymarid. Its height is 20m and rge side of base is 24m. Consider the centroids of the 4 triangular faces and the center of the base, which create another prymarid $\\Lambda$ (up-side down) What is the volumn of B?')

    elif level == 'Trung bình':
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns([1, 2.5])
            with c1:
                st.write(''' 
                    - Let $F$ be a transformation of a `plane` where any point $A$ (except the origin) with **polar coordinate** 
                    $A(r, \\phi)$ is transformed into a point $B = F(A)$ with polar coordinates $B (1/r, \\phi)$. 
                    - Let $C$ be the `boundary of a circle` with `radius` 0.5 and its `center` being 0.5 away from the origin. 
                     
                    What is the shape of $F(C)$?
                ''')

    elif level == 'Khó':
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write('There is a unit cube with `internal mirror faces`. A ray is emitted into the cube from one vertex, reflects off four faces (without touching vertices or edges), and stops at the opposite vertex from which it started. <br> What is the minimum possible distance the ray travels? [The answer should be decimal format approximated to 4 digits of precision, i.e. format a.bcd as in 1.234. ]')

        with st.expander('**:green[Câu 2]**'):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write(' We cut the zero-one interval into 4 parts by selecting 3 seperator points, uniformly and independently from the zero-one interval. What is the probability that the 4 segments can formed a quadrilateral')

elif topic == "Brainteaster":
    if level == "Dễ":
        with st.expander('**:green[Câu 1]**'):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write('How many times a day do the :green[hour] and :red[minute hands] of an :blue[analog clock] form a :orange[right angle]?')
                trig_key = f"see_answer_{level}_{topic}_1"
                see_answer = st.button("Xem lời giải", key=hash(trig_key))

        with st.expander('**:green[Câu 2]**'):
            st.markdown("There are three containers of pens. One container has red pens, one has blue pens, and one contains both red and blue pens. Each container is labeled. One is labeled `red`, one `blue`, and one `red and blue`. However, none of the containers' labels match the contents, and you want to fix this. If you are only allowed to sample just one pen from just one of the containers, what would you do so you can fix the labels on the containers?")
            options = ["A.  Sample a pen from the container labeled `red`", "B. Sample a pen from the container labeled `red and blue`",
                       "C. Sample a pen from the container labeled `blue`", "D. Sample a pen from any of the three containers", "E. None of the above is sufficient"]
            columns = st.columns(3)
            for idx, option in enumerate(options):
                col = columns[idx % 3]
                with col:
                    st.write(option)

            trig_key = f"see_answer_{level}_{topic}_2"
            see_answer = st.button("Xem lời giải", key=hash(trig_key))

            if see_answer:
                st.markdown("""
                <div style="font-size:16px; line-height:1.6">
                    We denoted them be <span style="color:purple; font-weight:bold">RR</span>, 
                    <span style="color:purple; font-weight:bold">BB</span> and 
                    <span style="color:purple; font-weight:bold">RB</span> be 3 labled containers, then<br>
                    - <span style="color:purple; font-weight:bold">RR</span> may be 
                    <span style="color:red; font-weight:bold">RB</span> | 
                    <span style="color:red; font-weight:bold">BB</span><br>
                    - <span style="color:purple; font-weight:bold">BB</span> may be 
                    <span style="color:red; font-weight:bold">RB</span> | 
                    <span style="color:red; font-weight:bold">RR</span><br>
                    - <span style="color:purple; font-weight:bold">RB</span> may be 
                    <span style="color:red; font-weight:bold">BB</span> | 
                    <span style="color:red; font-weight:bold">RR</span><br><br> 
                    From <span style="color:purple; font-weight:bold">RB</span>; if we get 
                    <span style="color:red; font-weight:bold">B</span> then this must be 
                    <span style="color:red; font-weight:bold">BB</span><br>
                    =&gt; the <span style="color:purple; font-weight:bold">RR</span> must be 
                    <span style="color:red; font-weight:bold">RB</span>;<br>
                    =&gt; <span style="color:purple; font-weight:bold">BB</span> will be 
                    <span style="color:red; font-weight:bold">RR</span><br>
                    otherwise <span style="color:red; font-weight:bold">RR</span><br>
                    =&gt; the <span style="color:purple; font-weight:bold">BB</span> must be 
                    <span style="color:red; font-weight:bold">RB</span>;<br>
                    =&gt; <span style="color:purple; font-weight:bold">RR</span> will be 
                    <span style="color:red; font-weight:bold">BB</span><br><br>
                    If we based on <span style="color:purple; font-weight:bold">RR</span> or 
                    <span style="color:purple; font-weight:bold">BB</span> first, we do not have enough information to conclude
                </div>
                """, unsafe_allow_html=True)