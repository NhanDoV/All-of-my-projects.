import random
import html
import streamlit as st

# FunnyMath — single-file Streamlit app
# Run with: streamlit run FunnyMath.py

st.set_page_config(
    page_title="FunnyMath | Math Practice",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Theme ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --fm-bg: #0b1020;
        --fm-panel: #121a2d;
        --fm-panel-2: #17223a;
        --fm-border: #293650;
        --fm-text: #eef4ff;
        --fm-muted: #9eacc5;
        --fm-cyan: #67e8f9;
        --fm-violet: #a5b4fc;
        --fm-green: #86efac;
        --fm-orange: #fdba74;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(67, 97, 238, .18), transparent 32%),
          radial-gradient(ellipse at 92% 8%, rgba(34, 211, 238, .10), transparent 28%),
          var(--fm-bg);
        color: var(--fm-text);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(11,16,32,.78); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--fm-text) !important; }
    p, label, span, div { font-family: 'DM Sans', sans-serif; }
    .hero {
        position: relative; overflow: hidden; padding: 30px 32px; border-radius: 24px;
        border: 1px solid rgba(148,163,184,.22);
        background: linear-gradient(125deg, rgba(28,42,72,.97), rgba(17,28,49,.94) 58%, rgba(24,43,66,.94));
        box-shadow: 0 18px 55px rgba(0,0,0,.20); margin-bottom: 24px;
    }
    .hero:after {
        content: "∑"; position: absolute; right: 34px; top: -34px; font-size: 190px;
        line-height: 1; color: rgba(103,232,249,.08); font-family: 'Space Grotesk', sans-serif;
        transform: rotate(12deg); pointer-events: none;
    }
    .hero-kicker { color: var(--fm-cyan); text-transform: uppercase; letter-spacing: .16em; font-size: .76rem; font-weight: 700; }
    .hero-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.2rem, 5vw, 3.5rem); font-weight: 700; letter-spacing: -.055em; line-height: 1.05; margin: 8px 0 10px; color: #f8fbff; }
    .hero-copy { max-width: 650px; color: #b9c7df; font-size: 1rem; line-height: 1.65; }
    .hero-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
    .hero-tag { padding: 6px 11px; border: 1px solid #3a4c6c; border-radius: 999px; color: #dbeafe; background: rgba(15,23,42,.35); font-size: .78rem; }
    .section-kicker { color: var(--fm-cyan); font-size: .75rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; margin-bottom: 5px; }
    .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700; color: var(--fm-text); margin-bottom: 4px; }
    .section-copy { color: var(--fm-muted); font-size: .92rem; margin-bottom: 18px; }
    .stSelectbox label, .stRadio label, .stSlider label, .stNumberInput label { color: #dce7fa !important; font-weight: 600 !important; }
    [data-baseweb="select"] > div { background: #121a2d; border-color: #33435f; border-radius: 12px; color: #eef4ff; }
    [data-testid="stRadio"] > div { gap: 12px; }
    [data-testid="stRadio"] label { background: #121a2d; border: 1px solid #2b3954; border-radius: 12px; padding: 9px 13px; }
    [data-testid="stRadio"] label:hover { border-color: #67e8f9; }
    .stButton > button {
        border: 1px solid #67e8f9; border-radius: 12px; min-height: 2.8rem;
        background: linear-gradient(110deg, #67e8f9, #a5b4fc); color: #0b1020;
        font-weight: 700; transition: transform .16s ease, box-shadow .16s ease;
    }
    .stButton > button:hover { border-color: #a5f3fc; color: #060b16; transform: translateY(-1px); box-shadow: 0 8px 22px rgba(103,232,249,.15); }
    .stButton > button:focus { box-shadow: 0 0 0 .2rem rgba(103,232,249,.25); }
    .question-card {
        border: 1px solid var(--fm-border); background: linear-gradient(145deg, rgba(18,26,45,.98), rgba(17,27,46,.96));
        border-radius: 18px; padding: 20px 22px; margin: 12px 0 7px; box-shadow: 0 9px 26px rgba(0,0,0,.10);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--fm-border) !important;
        background: linear-gradient(145deg, rgba(18,26,45,.98), rgba(17,27,46,.96)) !important;
        border-radius: 18px !important;
        padding: 18px 22px !important;
        margin: 12px 0 7px !important;
        box-shadow: 0 9px 26px rgba(0,0,0,.10);
    }
    .question-top { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom: 13px; }
    .question-number { color: var(--fm-cyan); font-size:.76rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
    .question-level { color:#c7d2fe; background:rgba(129,140,248,.12); border:1px solid rgba(165,180,252,.25); padding:4px 9px; border-radius:999px; font-size:.74rem; white-space:nowrap; }
    .question-text { color:#f0f5ff; font-size:1.04rem; line-height:1.75; }
    .solution-card { border:1px solid rgba(134,239,172,.28); border-radius:14px; padding:17px 19px; margin: 8px 0 20px; background:linear-gradient(135deg,rgba(20,57,51,.35),rgba(18,26,45,.92)); }
    .solution-title { color:var(--fm-green); font-weight:700; margin-bottom:9px; }
    .solution-body { color:#d9e8e5; line-height:1.8; }
    .empty-state { text-align:center; padding:35px 20px; border:1px dashed #3a4964; border-radius:18px; color:var(--fm-muted); background:rgba(18,26,45,.55); }
    .footer { text-align:center; color:#71819d; font-size:.8rem; padding-top:28px; }
    [data-testid="stMarkdownContainer"] hr { border-color:#27334b; }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 23px 20px; }
        .hero:after { right: 5px; font-size: 120px; top: 2px; }
        .question-card { padding: 16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Fixed question bank ----------
# Each solution contains a short explanation and a final answer.
QUESTION_BANK = {
    5: {
        "Hình học": [
            {
                "id": "g5-01",
                "question": "Một hình chữ nhật có chiều dài $12\\,\\text{cm}$, chiều rộng $7\\,\\text{cm}$. Tính chu vi hình chữ nhật.",
                "steps": [
                    "Chu vi hình chữ nhật bằng tổng chiều dài và chiều rộng nhân với 2.",
                    "$P=(12+7)\\times 2=38\\,\\text{cm}$.",
                ],
                "answer": "$\\boxed{38\\,\\text{cm}}$",
            },
            {
                "id": "g5-02",
                "question": "Một hình vuông có cạnh $9\\,\\text{cm}$. Tính diện tích hình vuông.",
                "steps": [
                    "Diện tích hình vuông bằng cạnh nhân với chính nó.",
                    "$S=9\\times 9=81\\,\\text{cm}^2$.",
                ],
                "answer": "$\\boxed{81\\,\\text{cm}^2}$",
            },
            {
                "id": "g5-03",
                "question": "Một tam giác có độ dài ba cạnh lần lượt là $5\\,\\text{cm}$, $6\\,\\text{cm}$ và $8\\,\\text{cm}$. Tính chu vi tam giác.",
                "steps": [
                    "Chu vi tam giác bằng tổng độ dài ba cạnh.",
                    "$P=5+6+8=19\\,\\text{cm}$.",
                ],
                "answer": "$\\boxed{19\\,\\text{cm}}$",
            },
            {
                "id": "g5-04",
                "question": "Một hình chữ nhật có diện tích $48\\,\\text{cm}^2$ và chiều rộng $6\\,\\text{cm}$. Tính chiều dài.",
                "steps": [
                    "Chiều dài bằng diện tích chia cho chiều rộng.",
                    "$48:6=8\\,\\text{cm}$.",
                ],
                "answer": "$\\boxed{8\\,\\text{cm}}$",
            },
            {
                "id": "g5-05",
                "question": "Một hình vuông có chu vi $36\\,\\text{cm}$. Tính độ dài cạnh và diện tích của hình vuông.",
                "steps": [
                    "Độ dài một cạnh bằng chu vi chia cho 4: $36:4=9\\,\\text{cm}$.",
                    "Diện tích là $9\\times9=81\\,\\text{cm}^2$.",
                ],
                "answer": "$\\boxed{\\text{Cạnh }9\\,\\text{cm};\\ S=81\\,\\text{cm}^2}$",
            },
        ],
        "Số học": [
            {
                "id": "a5-01",
                "question": "Tính: $245+378$.",
                "steps": ["Cộng theo hàng đơn vị, chục rồi đến trăm.", "$245+378=623$."],
                "answer": "$\\boxed{623}$",
            },
            {
                "id": "a5-02",
                "question": "Tính: $864:8$.",
                "steps": ["Thực hiện phép chia.", "$864:8=108$."],
                "answer": "$\\boxed{108}$",
            },
            {
                "id": "a5-03",
                "question": "Tính giá trị biểu thức: $125\\times 4+360:6$.",
                "steps": [
                    "Thực hiện phép nhân và phép chia trước: $125\\times4=500$, $360:6=60$.",
                    "Cộng hai kết quả: $500+60=560$.",
                ],
                "answer": "$\\boxed{560}$",
            },
            {
                "id": "a5-04",
                "question": "Tìm $x$, biết $x+275=640$.",
                "steps": ["Muốn tìm số hạng chưa biết, lấy tổng trừ số hạng đã biết.", "$x=640-275=365$."],
                "answer": "$\\boxed{x=365}$",
            },
            {
                "id": "a5-05",
                "question": "Một cửa hàng có $240$ quyển vở, đã bán $\\frac{3}{8}$ số vở đó. Hỏi cửa hàng đã bán bao nhiêu quyển vở?",
                "steps": [
                    "Số vở đã bán bằng tổng số vở nhân với phân số chỉ phần đã bán.",
                    "$240\\times\\frac{3}{8}=30\\times3=90$ quyển.",
                ],
                "answer": "$\\boxed{90\\text{ quyển vở}}$",
            },
        ],
    },
    9: {
        "Hình học": [
            {
                "id": "g9-01",
                "question": "Một đường tròn có bán kính $R=5\\,\\text{cm}$. Tính chu vi và diện tích đường tròn theo $\\pi$.",
                "steps": [
                    "Chu vi: $C=2\\pi R=2\\pi\\times5=10\\pi\\,\\text{cm}$.",
                    "Diện tích: $S=\\pi R^2=\\pi\\times5^2=25\\pi\\,\\text{cm}^2$.",
                ],
                "answer": "$\\boxed{C=10\\pi\\,\\text{cm};\\ S=25\\pi\\,\\text{cm}^2}$",
            },
            {
                "id": "g9-02",
                "question": "Tam giác vuông $ABC$ vuông tại $A$, có $AB=6\\,\\text{cm}$, $AC=8\\,\\text{cm}$. Tính $BC$.",
                "steps": [
                    "Áp dụng định lý Pythagore: $BC^2=AB^2+AC^2$.",
                    "$BC^2=6^2+8^2=100$, nên $BC=10\\,\\text{cm}$.",
                ],
                "answer": "$\\boxed{BC=10\\,\\text{cm}}$",
            },
            {
                "id": "g9-03",
                "question": "Một hình trụ có bán kính đáy $r=3\\,\\text{cm}$, chiều cao $h=7\\,\\text{cm}$. Tính thể tích hình trụ.",
                "steps": ["Công thức thể tích hình trụ: $V=\\pi r^2h$.", "$V=\\pi\\times3^2\\times7=63\\pi\\,\\text{cm}^3$."],
                "answer": "$\\boxed{63\\pi\\,\\text{cm}^3}$",
            },
            {
                "id": "g9-04",
                "question": "Một tam giác đồng dạng với tam giác khác theo tỉ số đồng dạng $k=2$. Nếu một cạnh của tam giác nhỏ dài $4\\,\\text{cm}$, cạnh tương ứng của tam giác lớn dài bao nhiêu?",
                "steps": ["Tỉ số đồng dạng bằng tỉ số hai cạnh tương ứng.", "$\\text{Cạnh lớn}=2\\times4=8\\,\\text{cm}$."],
                "answer": "$\\boxed{8\\,\\text{cm}}$",
            },
            {
                "id": "g9-05",
                "question": "Một hình nón có bán kính đáy $r=4\\,\\text{cm}$, chiều cao $h=3\\,\\text{cm}$. Tính thể tích hình nón.",
                "steps": ["Công thức thể tích hình nón: $V=\\frac{1}{3}\\pi r^2h$.", "$V=\\frac13\\pi\\times4^2\\times3=16\\pi\\,\\text{cm}^3$."],
                "answer": "$\\boxed{16\\pi\\,\\text{cm}^3}$",
            },
        ],
        "Số học": [
            {
                "id": "a9-01",
                "question": "Rút gọn phân thức $\\frac{x^2-9}{x+3}$, với điều kiện $x\\ne-3$.",
                "steps": ["Phân tích tử thành hiệu hai bình phương: $x^2-9=(x-3)(x+3)$.", "Vì $x\\ne-3$, rút gọn được $\\frac{(x-3)(x+3)}{x+3}=x-3$."],
                "answer": "$\\boxed{x-3\\quad (x\\ne-3)}$",
            },
            {
                "id": "a9-02",
                "question": "Giải phương trình: $3x-7=11$.",
                "steps": ["Chuyển $-7$ sang vế phải: $3x=11+7=18$.", "Chia hai vế cho 3: $x=6$."],
                "answer": "$\\boxed{x=6}$",
            },
            {
                "id": "a9-03",
                "question": "Phân tích đa thức thành nhân tử: $x^2-5x+6$.",
                "steps": ["Tìm hai số có tích bằng 6 và tổng bằng $-5$: đó là $-2$ và $-3$.", "Suy ra $x^2-5x+6=(x-2)(x-3)$."],
                "answer": "$\\boxed{(x-2)(x-3)}$",
            },
            {
                "id": "a9-04",
                "question": "Giải hệ phương trình: $\\begin{cases}x+y=7\\\\x-y=1\\end{cases}$.",
                "steps": ["Cộng hai phương trình: $2x=8\\Rightarrow x=4$.", "Thay vào $x+y=7$: $y=3$."],
                "answer": "$\\boxed{(x,y)=(4,3)}$",
            },
            {
                "id": "a9-05",
                "question": "Một hộp có 3 viên bi đỏ và 2 viên bi xanh. Lấy ngẫu nhiên một viên. Tính xác suất lấy được viên bi xanh.",
                "steps": ["Có tổng cộng $3+2=5$ viên bi, trong đó có 2 viên xanh.", "Xác suất bằng số kết quả thuận lợi chia tổng số kết quả đồng khả năng: $P=\\frac25$."],
                "answer": "$\\boxed{\\frac25}$",
            },
        ],
    },
    12: {
        "Hình học": [
            {
                "id": "g12-01",
                "question": "Trong không gian $Oxyz$, cho hai điểm $A(1,2,3)$, $B(4,6,3)$. Tính độ dài đoạn thẳng $AB$.",
                "steps": ["Dùng công thức khoảng cách trong không gian.", "$AB=\\sqrt{(4-1)^2+(6-2)^2+(3-3)^2}=\\sqrt{9+16}=5$."],
                "answer": "$\\boxed{AB=5}$",
            },
            {
                "id": "g12-02",
                "question": "Cho mặt cầu tâm $I(0,0,0)$, bán kính $R=3$. Viết phương trình mặt cầu.",
                "steps": ["Phương trình mặt cầu tâm $I(a,b,c)$, bán kính $R$ là $(x-a)^2+(y-b)^2+(z-c)^2=R^2$.", "Thay $a=b=c=0$, $R=3$, được $x^2+y^2+z^2=9$."],
                "answer": "$\\boxed{x^2+y^2+z^2=9}$",
            },
            {
                "id": "g12-03",
                "question": "Một khối lập phương có cạnh $a=4\\,\\text{cm}$. Tính thể tích khối lập phương.",
                "steps": ["Thể tích khối lập phương: $V=a^3$.", "$V=4^3=64\\,\\text{cm}^3$."],
                "answer": "$\\boxed{64\\,\\text{cm}^3}$",
            },
            {
                "id": "g12-04",
                "question": "Trong không gian, cho vectơ $\\vec{u}=(2,-1,2)$. Tính độ dài của vectơ $\\vec{u}$.",
                "steps": ["Độ dài vectơ $(a,b,c)$ bằng $\\sqrt{a^2+b^2+c^2}$.", "$\\|\\vec{u}\\|=\\sqrt{2^2+(-1)^2+2^2}=\\sqrt9=3$."],
                "answer": "$\\boxed{3}$",
            },
            {
                "id": "g12-05",
                "question": "Một hình nón tròn xoay có bán kính đáy $r=3\\,\\text{cm}$, chiều cao $h=4\\,\\text{cm}$. Tính thể tích.",
                "steps": ["Dùng công thức $V=\\frac13\\pi r^2h$.", "$V=\\frac13\\pi\\times9\\times4=12\\pi\\,\\text{cm}^3$."],
                "answer": "$\\boxed{12\\pi\\,\\text{cm}^3}$",
            },
        ],
        "Số học": [
            {
                "id": "a12-01",
                "question": "Tính đạo hàm của hàm số $f(x)=3x^4-2x^2+5$.",
                "steps": ["Áp dụng quy tắc $(x^n)'=nx^{n-1}$ và đạo hàm của hằng số bằng 0.", "$f'(x)=12x^3-4x$."],
                "answer": "$\\boxed{f'(x)=12x^3-4x}$",
            },
            {
                "id": "a12-02",
                "question": "Tính tích phân $\\int_0^2 3x^2\\,dx$.",
                "steps": ["Nguyên hàm của $3x^2$ là $x^3$.", "$\\int_0^2 3x^2\\,dx=[x^3]_0^2=8-0=8$."],
                "answer": "$\\boxed{8}$",
            },
            {
                "id": "a12-03",
                "question": "Giải phương trình $\\log_2(x)=3$.",
                "steps": ["Đổi sang dạng lũy thừa: $x=2^3$.", "Suy ra $x=8$, thỏa điều kiện $x>0$."],
                "answer": "$\\boxed{x=8}$",
            },
            {
                "id": "a12-04",
                "question": "Tìm số phức liên hợp của $z=3-4i$ và mô-đun của $z$.",
                "steps": ["Số phức liên hợp đổi dấu phần ảo: $\\overline z=3+4i$.", "Mô-đun: $|z|=\\sqrt{3^2+(-4)^2}=5$."],
                "answer": "$\\boxed{\\overline z=3+4i,\\quad |z|=5}$",
            },
            {
                "id": "a12-05",
                "question": "Tìm giá trị lớn nhất của hàm số $f(x)=-x^2+4x+1$ trên $\\mathbb{R}$.",
                "steps": ["Hoàn thành bình phương: $f(x)=-(x-2)^2+5$.", "Vì $(x-2)^2\\ge0$, nên $f(x)\\le5$, đạt được khi $x=2$."],
                "answer": "$\\boxed{\\max f(x)=5\\text{ tại }x=2}$",
            },
        ],
    },
}

# ---------- Header ----------
st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Learn · Practice · Discover</div>
      <div class="hero-title">FunnyMath <span style="color:#67e8f9">∑</span></div>
      <div class="hero-copy">
        Một góc luyện Toán nhẹ nhàng mà hiệu quả. Chọn khối lớp, chọn chủ đề, thử sức với từng câu hỏi và chỉ mở lời giải khi bạn đã sẵn sàng.
        (Sẽ còn update trong tương lai)
      </div>
      <div class="hero-tags">
        <span class="hero-tag">✦ Lớp 5 · 9 · 12</span>
        <span class="hero-tag">⌘ Hình học &amp; Số học</span>
        <span class="hero-tag">∑ Lời giải từng bước</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "quiz_config" not in st.session_state:
    st.session_state.quiz_config = None
if "shown_solutions" not in st.session_state:
    st.session_state.shown_solutions = set()

st.markdown(
    '<div class="section-kicker">01 / Setup</div>'
    '<div class="section-title">Thiết lập bài ôn tập</div>'
    '<div class="section-copy">Chọn cấu hình rồi nhấn Generate quiz để tạo một bộ câu hỏi.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    col_grade, col_topic, col_count = st.columns([1, 1.4, 1])
    with col_grade:
        grade = st.selectbox("Khối lớp", options=[5, 9, 12], format_func=lambda x: f"Lớp {x}", key="grade")
    with col_topic:
        topic = st.selectbox("Chủ đề", options=["Hình học", "Số học"], key="topic")
    with col_count:
        num_questions = st.selectbox("Số câu hỏi", options=[3, 4, 5], index=0, format_func=lambda x: f"{x} câu", key="num_questions")

    st.markdown("")
    generate_col, note_col = st.columns([1, 2.5], vertical_alignment="center")
    with generate_col:
        generate = st.button("✦  Generate quiz", use_container_width=True, type="primary")
    with note_col:
        st.caption("Question bank cố định · Mỗi câu có lời giải riêng · Công thức hỗ trợ LaTeX")

if generate:
    available = QUESTION_BANK[grade][topic]
    # Sample without replacement; current bank has five questions per grade/topic.
    seed = random.SystemRandom().randint(0, 2**32 - 1)
    rng = random.Random(seed)
    st.session_state.quiz = rng.sample(available, k=min(num_questions, len(available)))
    st.session_state.quiz_config = (grade, topic, num_questions)
    st.session_state.shown_solutions = set()

# Keep the current generated quiz stable when Streamlit reruns due to solution buttons.
if st.session_state.quiz and st.session_state.quiz_config:
    quiz_grade, quiz_topic, quiz_count = st.session_state.quiz_config
    st.markdown("---")
    st.markdown(
        f'<div class="section-kicker">02 / Practice</div>'
        f'<div class="section-title">Bài ôn tập của bạn</div>'
        f'<div class="section-copy">Lớp {quiz_grade} &nbsp;·&nbsp; {html.escape(quiz_topic)} &nbsp;·&nbsp; {len(st.session_state.quiz)} câu hỏi</div>',
        unsafe_allow_html=True,
    )

    for index, item in enumerate(st.session_state.quiz, start=1):
        question_id = item["id"]
        # Use a native Streamlit container for the card, then let Streamlit's
        # Markdown renderer handle LaTeX in the question text.
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="question-top">
                  <span class="question-number">Question {index:02d}</span>
                  <span class="question-level">Lớp {quiz_grade} · {html.escape(quiz_topic)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(item["question"])
        # A unique key per question ensures independent reveal/hide behavior.
        if question_id in st.session_state.shown_solutions:
            if st.button("🙈  Hide solution", key=f"hide_{question_id}", use_container_width=False):
                st.session_state.shown_solutions.remove(question_id)
                st.rerun()
            st.markdown(
                '<div class="solution-card"><div class="solution-title">✓ Solution · Lời giải</div><div class="solution-body">',
                unsafe_allow_html=True,
            )
            for step_no, step in enumerate(item["steps"], start=1):
                st.markdown(f"**Bước {step_no}.** {step}")
            st.markdown(f"**Đáp án:** {item['answer']}")
            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            if st.button("💡  Show solution", key=f"show_{question_id}", use_container_width=False):
                st.session_state.shown_solutions.add(question_id)
                st.rerun()

    st.markdown(
        '<div class="footer">Bạn làm được rồi — cứ đi từng bước một. ✦ FunnyMath</div>',
        unsafe_allow_html=True,
    )
else:
    if not st.session_state.quiz:
        st.markdown(
            """
            <div class="empty-state">
              <div style="font-size:2rem;margin-bottom:8px">✦ ∑ ✦</div>
              <div style="font-weight:700;color:#dce7fa;margin-bottom:5px">Sẵn sàng thử thách bản thân?</div>
              <div>Chọn khối lớp và chủ đề, sau đó nhấn <b>Generate quiz</b> để bắt đầu.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Existing quiz remains visible until a new quiz is generated.
        pass
