import random
import html
import streamlit as st
from pathlib import Path

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
    .block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 3rem; }
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
    .section-copy { color: var(--fm-muted); font-size: .83rem; margin-bottom: 18px; }
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
    .question-number { color: var(--fm-cyan); font-size:.71rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
    .question-level { color:#c7d2fe; background:rgba(129,140,248,.12); border:1px solid rgba(165,180,252,.25); padding:4px 9px; border-radius:999px; font-size:.72rem; white-space:nowrap; }
    .question-text { color:#f0f5ff; font-size:15px; }
    .solution-card { border:1px solid rgba(134,239,172,.28); border-radius:14px; padding:17px 19px; margin: 8px 0 20px; background:linear-gradient(135deg,rgba(20,57,51,.35),rgba(18,26,45,.92)); }
    .solution-title { color:var(--fm-green); font-weight:700; margin-bottom:9px; }
    .solution-body { color:#d9e8e5; line-height:1.8; }
    .empty-state { text-align:center; padding:35px 20px; border:1px dashed #3a4964; border-radius:18px; color:var(--fm-muted); background:rgba(18,26,45,.55); }
    .footer { text-align:center; color:#71819d; font-size:.72rem; padding-top:28px; }
    [data-testid="stMarkdownContainer"] hr { border-color:#27334b; }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 23px 20px; }
        .hero:after { right: 5px; font-size: 110px; top: 2px; }
        .question-card { padding: 16px; }
    }
    /* Font size cho LaTeX (KaTeX) */
    .katex {
        font-size: 1.05em !important;   /* hoặc 0.95em, 14px, ... tùy bạn muốn nhỏ hơn */
    }

    /* Nếu muốn nhỏ hơn nữa cho công thức trong solution */
    .solution-body .katex {
        font-size: 0.95em !important;
    }

    /* Công thức display ($$...$$) nếu sau này dùng */
    .katex-display {
        font-size: 1.1em !important;
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
            {
                "id": "g5-06",
                "question": "Một mảnh đất hình chữ nhật với chiều dài $52$ m và chiều rộng bằng $\\frac{3}{5}$ chiều dài. Tính chu vi và diện tích của mảnh đất.",
                "steps": [
                    "Chiều rộng là $52 \\times \\frac{3}{5} = 31.2$ m.",
                    "Chu vi là $(52 + 31.2) \\times 2 = 166.4$ m.",
                    "Diện tích là $52 \\times 31.2 = 1622.4$ m$^2$.",
                ],
                "answer": "$\\boxed{\\text{Chu vi }166.4\\,\\text{m};\\ \\text{Diện tích }1622.4\\,\\text{m}^2}$",
            },
            {
                "id": "g5-07",
                "question": "Một tấm bìa hình chữ nhật có chiều dài $23$ cm và chiều rộng kém chiều dài $5$ cm. Tính diện tích tấm bìa hình vuông có chu vi bằng chu vi của tấm bìa hình chữ nhật kể trên.",
                "steps": [
                    "Chiều rộng là $23 - 5 = 18$ cm.",
                    "Chu vi của tấm bìa hình chữ nhật là $(23 + 18) \\times 2 = 82$ cm.",
                    "Độ dài cạnh của hình vuông là $82:4 = 20.5$ cm.",
                    "Diện tích của hình vuông là $20.5 \\times 20.5 = 420.25$ cm$^2$.",
                ],
                "answer": "$\\boxed{\\text{Chu vi }82\\,\\text{cm};\\ \\text{Diện tích }420.25\\,\\text{cm}^2}$",
            },
            {
                "id": "g5-08",
                "question": "Một hình thang có diện tích bằng $60$ m$^2$, hiệu hai đáy là $4$ m. Tính độ dài mỗi đáy, biết rằng nếu đáy lớn được tăng thêm $2$ m thì diện tích hình thang cũng tăng thêm $6$ m$^2$.",
                "steps": [
                    "Gọi đáy lớn là $a$ và đáy nhỏ là $b$. Ta có $a - b = 4$.",
                    "Diện tích hình thang là $S = \\frac{(a + b)h}{2} = 60$ m$^2$.",
                    "Khi đáy lớn tăng thêm $2$ m, diện tích mới là $S' = \\frac{(a + 2 + b)h}{2} = 66$ m$^2$.",
                    "Từ hai phương trình diện tích, ta có hệ phương trình: $\\begin{cases} \\frac{(a + b)h}{2} = 60 \\\\ \\frac{(a + 2 + b)h}{2} = 66 \\end{cases}$",
                    "Suy ra $\\dfrac{a+b+2}{a+b} = \\dfrac{66}{60} = \\dfrac{11}{10}$ hay $\\dfrac{2}{a+b} = \\dfrac{1}{10}$, tức là $a + b = 20$.",
                    "Kết hợp với $a - b = 4$, ta tìm được $a$ và $b$."
                ],
                "answer": "$\\boxed{\\text{Đáy lớn } a = 12\\,\\text{m};\\ \\text{Đáy nhỏ } b = 8\\,\\text{m}}$",
            },
            {
                "id": "g5-09",
                "question": "Một thửa ruộng hình chữ nhật có chu vi $192$ m; nếu chiều dài giảm đi $3$ m và chiều rộng tăng thêm $3$ m thì thửa ruộng đó trở thành hình vuông. Tính diện tích của thửa ruộng hình chữ nhật.",
                "steps": [
                    "Gọi chiều dài là $a$ và chiều rộng là $b$. Ta có hệ phương trình: $\\begin{cases} 2(a + b) = 192 \\\\ a - 3 = b + 3 \\end{cases}$",
                    "Từ phương trình thứ hai: $a - b = 6$.",
                    "Cộng hai phương trình: $2a + 2b + a - b = 192 + 6$, suy ra $3a + b = 198$.",
                    "Thay $b = a - 6$ vào phương trình trên: $3a + (a - 6) = 198$, giải được $a = 51$.",
                    "Suy ra $b = 45$.",
                    "Diện tích thửa ruộng là $a \\times b = 51 \\times 45 = 2295$ m$^2$."
                ],
                "answer": "$\\boxed{\\text{Diện tích } 2295\\,\\text{m}^2}$",
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
            {
                "id": "a5-06",
                "question": "Trong vườn có $45$ cây vừa cam, vừa chanh, vừa bưởi. Số cây cam bằng $\\frac{3}{4}$ số cây chanh và số cây chanh chỉ bằng một nửa số cây bưởi. Hỏi trong vườn có bao nhiêu cây cam, cây chanh và cây bưởi?",
                "steps": [
                    "Gọi số cây bưởi là $x$.",
                    "Số cây chanh là $\\frac{1}{2}x$ và số cây cam là $\\frac{3}{4}\\cdot\\frac{1}{2}x=\\frac{3}{8}x$.",
                    "Tổng số cây: $x + \\frac{1}{2}x + \\frac{3}{8}x = 45$.",
                    "Giải phương trình: $\\frac{15}{8}x = 45 \\Rightarrow x = 24$."
                ],
                "answer": "$\\boxed{\\text{Cam: } 9\\text{ cây};\\ \\text{Chanh: } 12\\text{ cây};\\ \\text{Bưởi: } 24\\text{ cây}}$"
            },
            {
                "id": "a5-07",
                "question": "Tổng số tuổi của hai anh em An và Bình năm nay là $18$ tuổi. Biết rằng sáu năm nữa thì tuổi của An bằng $\\frac{2}{3}$ tuổi của Bình. Hỏi hiện nay mỗi người bao nhiêu tuổi?",
                "steps": [
                    "Gọi tuổi của An hiện nay là $x$ và tuổi của Bình hiện nay là $y$.",
                    "Từ đề bài, ta có hệ phương trình: $\\begin{cases} x + y = 18 \\\\ x + 6 = \\frac{2}{3}(y + 6) \\end{cases}$",
                    "Từ phương trình thứ hai: $3(x + 6) = 2(y + 6)$, suy ra $3x + 18 = 2y + 12$, hay $3x - 2y = -6$.",
                    "Giải hệ phương trình: $\\begin{cases} x + y = 18 \\\\ 3x - 2y = -6 \\end{cases}$",
                    "Nhân phương trình thứ nhất với 2: $2x + 2y = 36$.",
                    "Cộng với phương trình thứ hai: $5x = 30 \\Rightarrow x = 6$.",
                    "Suy ra $y = 12$."
                ],
                "answer": "$\\boxed{\\text{An: } 6\\text{ tuổi};\\ \\text{Bình: } 12\\text{ tuổi}}$"
            },
            {
                "id": "a5-08",
                "question": "Thực hiện phép tính sau: $6$ năm $7$ tháng $+$ $4$ năm $8$ tháng",
                "steps": [
                    "Cộng năm với năm và tháng với tháng: $6 + 4 = 10$ năm, $7 + 8 = 15$ tháng.",
                    "Chuyển $15$ tháng thành $1$ năm $3$ tháng.",
                    "Tổng cộng: $10 + 1 = 11$ năm và $3$ tháng.",
                ],
                "answer": "$\\boxed{11\\text{ năm } 3\\text{ tháng}}$"
            },
            {
                "id": "a5-09",
                "question": "Một chiếc BMW chạy trên cao tốc, trong $3$ giờ đầu tiên mỗi giờ đi được $100$ km. Sau đó vào thành phố, mỗi giờ đi được $40$ km và xe dừng chân sau $2$ giờ. Hỏi trung bình mỗi giờ chiếc BMW đi được bao nhiêu km?",
                "steps": [
                    "Quãng đường đi trong $3$ giờ đầu: $3 \\times 100 = 300$ km.",
                    "Quãng đường đi trong $2$ giờ sau: $2 \\times 40 = 80$ km.",
                    "Tổng quãng đường: $300 + 80 = 380$ km.",
                    "Tổng thời gian: $3 + 2 = 5$ giờ.",
                    "Vận tốc trung bình: $\\frac{380}{5} = 76$ km/h.",
                ],
                "answer": "$\\boxed{76\\text{ km/h}}$"
            }
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
                "question": "Cho tam giác $ABC$ cân tại $A$, còn $I$ là tâm đường tròn nội tiếp, $K$ là tâm đường tròn bàng tiếp góc $A$; và $O$ là trung điểm của IK. \n 1) CMR 4 điểm $B, C, I, K$ cùng một đường tròn \n 2) CMR $AC$ là tiếp tuyến của đường tròn $(O)$ \n 3) Tính bán kính đường tròn $(O)$ nếu biết $AB = AC = 20$ cm và $BC = 24$ cm",
                "steps": [
                    "Vì $I$ là tâm đường tròn nội tiếp và $K$ là tâm đường tròn bàng tiếp góc $A$, nên $BI, BK$ là các tia phân giác của các góc kề bù đỉnh $B$, do đó $\\widehat{IBK} = 90^\\circ$. \n Tương tự, $\\widehat{ICK} = 90^\\circ$. \n $\\Rightarrow$ Suy ra 4 điểm $B, C, I, K$ $\\boxed{\\text{cùng nằm trên đường tròn đường kính}}$ $IK$.",
                    "Vì $CI$ là tia phân giác của góc $C$, nên $\\widehat{ACI} = \\widehat{ICB}$ (i).",
                    "Gọi $H$ là giao điểm của $AI$ và $BC$, khi đó $\\widehat{IHC} = 90^\\circ \\Rightarrow \\widehat{ICB} + \\widehat{HIC} = 90^\\circ$",
                    "Ngoài ra, $\\Delta OIC$ cân tại $O$ nên $\\widehat{OIC} = \\widehat{ICO}$ (iii);",
                    "Từ 3 dữ kiện (i), (ii), (iii) trên, ta được $\\widehat{ACI} + \\widehat{ICO} = 90^\\circ \\Rightarrow AC \\bot OC$",
                    "Vậy $AC$ là $\\boxed{\\text{tiếp tuyến của đường tròn}}$ tâm $O$ đường kính $IK$.",
                    "Từ các giả thiết về độ dài đã cho, $CH = 12$ cm (do $H$ đồng thời là trung điểm $BC$)",
                    "Áp dụng định lý Pythagore trong tam giác $ACH$, ta được $AH = \\sqrt{AC^2 - HC^2} = 16$ cm",
                    "Xét tam giác $AOC$ vuông tại $C$ với đường cao $CH$, ta có $CH^2 = AH \\cdot OH \\Rightarrow OH = \\frac{CH^2}{AH} = 9$ cm",
                    "Cuối cùng, xét tam giác $OHC$ vuông tại $H$, nên $OC = \\sqrt{OH^2 + HC^2} = 15$ cm",
                ],
                "answer": "$\\boxed{15\\, \\text{cm}}$",
            },
            {
                "id": "g9-06",
                "question": "Cho tam giác $ABC$ vuông tại $A$, có $AB=6\\,\\text{cm}$, $AC=8\\,\\text{cm}$. Gọi $H$ là đường cao kẻ từ $A$ và $M$ là trung điểm của $BC$. Tính độ dài đoạn $HM$.",
                "steps": [
                    "Tính $BC$ bằng định lý Pythagore: $BC^2 = AB^2 + AC^2 = 6^2 + 8^2 = 36 + 64 = 100 \\Rightarrow BC = 10$ cm.",
                    "Ngoài ra, $AH$ là đường cao nên $AH = \\frac{AB \\cdot AC}{BC} = \\frac{6 \\cdot 8}{10} = 4.8$ cm.",
                    "Trung điểm $M$ của $BC$ có tọa độ trung bình của $B$ và $C$, nên $BM = MC = \\frac{BC}{2} = 5$ cm.",
                    "Xét tam giác vuông $BAH$ có $BH = \\sqrt{AB^2 - AH^2} = \\sqrt{6^2 - 4.8^2} = \\sqrt{36 - 23.04} = \\sqrt{12.96} = 3.6$ cm.",
                    "Suy ra, $HM = BM - BH = 5 - 3.6 = 1.4$ cm.",
                ],
                "answer": "$\\boxed{1.4\\, \\text{cm}}$",
            },
            {
                "id": "g9-07",
                "question": "Cho tam giác $ABC$ cân tại $A$, các đường cao $AD$ và $BE$ cắt nhau tại $H$. Gọi $O$ là tâm đường tròn ngoại tiếp tam giác $AHE$. \n 1) CMR tứ giác $CEHD$ nội tiếp \n 2) CMR 4 điểm $A, E, D, B$ cùng thuộc một đường tròn \n 3) CMR $DE = \\dfrac{1}{2} BC$ và $DE$ là tiếp tuyến của $(O)$ \n 4) Tính độ dài $DE$ nếu biết $AH = 6$ cm và $DH = 2$ cm",
                "steps": [
                    "Trong tứ giác $CEHD$, ta có $\\widehat{CEH} = 90^\\circ$ ($BE$ là đường cao) và $\\widehat{CDH} = 90^\\circ$ (do $AD$ là đường cao).",
                    "Như vậy, $\\widehat{CEH} + \\widehat{CDH} = 180^\\circ$ và tứ giác $CEHD$ nội tiếp trong một đường tròn đường kính $CH$",
                    "Vì $AD$ và $BE$ là các đường cao nên $\\widehat{AEB} = \\widehat{ADB} = 90^\\circ$; dẫn đến các đỉnh $E, D$ cùng nhìn $AB$ dưới góc vuông.",
                    "Do đó, tứ giác $AEDB$ nội tiếp trong một đường tròn đường kính $AB$.",
                    "Do $ABC$ cân tại $A$ với $AD$ là đường cao nên cũng đồng thời là đường trung tuyến, suy ra $D$ là trung điểm của $BC$.",
                    "Hơn nữa, $\\widehat{BEC} = 90^\\circ$, nên tam giác $BEC$ vuông tại $E$ và $DE$ là trung tuyến nên $DE = \\frac{1}{2} BC$.",
                    "Vì $O$ là tâm đường tròn ngoại tiếp tam giác $AHE$, nên $O$ là trung điểm $AH$; dẫn đến $OAE$ là tam giác cân (tại $O$) nên $\\widehat{OAE} = \\widehat{OEA}$.",
                    "Ngoài ra, tam giác $DBE$ cân tại $D$ nên $\\widehat{DEB} = \\widehat{DBE}$.",
                    "Mà $\\widehat{DBE} = \\widehat{OAE}$ (cùng phụ với $\\widehat{ACB}$) do đó $$\\widehat{DEB} = \\widehat{OEA}$$",                    
                    "Cộng cả 2 vế cho góc $\\widehat{OEB}$, suy ra $\\widehat{DEB} + \\widehat{OEB} = \\widehat{OEA} + \\widehat{OEB} = 90^\\circ \\Rightarrow DE \\perp OE$; do đó $DE$ là tiếp tuyến của đường tròn $(O)$.",
                    "Từ các giả thiết về độ dài đã cho, ta được $OH = OE = 3$ cm; suy ra $OD = 5$ cm (do $DH = 2$ cm).",
                    "Xét tam giác vuông $ODE$, ta có $ED = \\sqrt{OD^2 - OE^2} = \\sqrt{25 - 9} = \\sqrt{16} = 4$ cm.",
                ],
                "answer": "$\\boxed{4 \\, \\text{cm}}$"
            }
         ],
        "Số học": [
            {
                "id": "a9-01",
                "question": "Rút gọn phân thức $\\frac{x^2-9}{x+3}$, với điều kiện $x \\neq -3$.",
                "steps": ["Phân tích tử thành hiệu hai bình phương: $x^2-9=(x-3)(x+3)$.", "Vì $x \\neq -3$, rút gọn được $\\frac{(x-3)(x+3)}{x+3}=x-3$."],
                "answer": "$\\boxed{x-3\\quad (x\\neq-3)}$",
            },
            {
                "id": "a9-02",
                "question": "Cho phương trình: $3x^2 - 5x + 3 = 0$ có 2 nghiệm $x_1, x_2$. Hãy tính $x_1^3 + x_2^3$.",
                "steps": [
                            "Tổng quát với tam thức bậc hai $ax^2 + bx + c = 0$ có hai nghiệm $x_1, x_2$, ta có: $x_1 + x_2 = -\\frac{b}{a}$ và $x_1 x_2 = \\frac{c}{a}$.", 
                            "Chú ý rằng $x_1^3 + x_2^3 = (x_1 + x_2)^3 - 3x_1x_2(x_1 + x_2)$.",
                            "Thay các giá trị vào và ta được $x_1 + x_2 = \\frac{5}{3}$ và $x_1 x_2 = 1$.",
                        ],
                "answer": "$\\boxed{\\frac{95}{27}}$",
            },
            {
                "id": "a9-03",
                "question": "Phân tích đa thức thành nhân tử: $2 x^3 - 9 x^2 + 7 x + 6$.",
                "steps": [
                            "Ta nhận thấy rằng $2 x^3 - 9 x^2 + 7x + 6 = (2x^3 - 4x^2) - (5x^2 - 10x) - (3x - 6)$",
                            "Tiếp theo, lấy $(x-2)$ làm nhân tử chung; ta được $(x-2)(2x^2 - 5x - 3)$",
                            "Cuối cùng, phân tích tiếp $2x^2 - 5x - 3 = (x-3)(2x+1)$",
                            "Vậy $2 x^3 - 9 x^2 + 7 x + 6 = (x-2)(x-3)(2x+1)$"
                        ],
                "answer": "$\\boxed{(x-2)(x-3)(2x+1)}$",
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
            {
                "id": "a9-06",
                "question": "Rút gọn căn thức $\\dfrac{1}{2 - \\sqrt{3}} +  \\sqrt{7 - 4\\sqrt{3}}$",
                "steps": [
                    "Rút gọn phân thức: $\\dfrac{1}{2 - \\sqrt{3}} = \\dfrac{2 + \\sqrt{3}}{(2 - \\sqrt{3})(2 + \\sqrt{3})} = 2 + \\sqrt{3}$.",
                    "Rút gọn căn thức: $\\sqrt{7 - 4\\sqrt{3}} = \\sqrt{(2 - \\sqrt{3})^2} = 2 - \\sqrt{3}$.",
                    "Cộng hai kết quả: $(2 + \\sqrt{3}) + (2 - \\sqrt{3}) = 4$.",
                ],
                "answer": "$\\boxed{4}$",
            },
            {
                "id": "a9-07",
                "question": "Rút gọn biểu thức: $\\sqrt{21 - 8\\sqrt{5}} + 2 \\sqrt{6 - 2\\sqrt{5}}$",
                "steps": [
                    "Đầu tiên, $\\sqrt{21 - 8\\sqrt{5}} = |4 - 2\\sqrt{5}| = 4 - 2\\sqrt{5}$.",
                    "Tiếp theo, $\\sqrt{6 - 2\\sqrt{5}} = |\\sqrt{5} - 1| = \\sqrt{5} - 1$.",
                    "Vậy $\\sqrt{21 - 8\\sqrt{5}} + 2\\sqrt{6 - 2\\sqrt{5}} = (4 - 2\\sqrt{5}) + 2(\\sqrt{5} - 1) = 2$."
                ],
                "answer": "$\\boxed{2}$"
            },
            {
                "id": "a9-08",
                "question": "CMR $A = 1 + 2^3 + 3^3 + \\ldots + 2026^3$ chia hết cho $2027$.",
                "steps": [
                        "Nhận thấy rằng $A$ có $1013$ cặp số, và ta có thể nhóm chúng lại như sau: $(1^3 + 2026^3) + (2^3 + 2025^3) + \\ldots + (1013^3 + 1014^3)$.",
                        "Áp dụng hằng đẳng thức $(a^3 + b^3) = (a + b)(a^2 - ab + b^2)$.",
                        "Với cặp số đầu tiên, $1^3 + 2026^3 = (1 + 2026)(1^2 - 1*2026 + 2026^2) = 2027(1 - 2026 + 2026^2)$, nên chia hết cho $2027$.",
                        "Tương tự, với cặp số thứ hai, $2^3 + 2025^3 = (2 + 2025)(2^2 - 2*2025 + 2025^2) = 2027(4 - 4050 + 2025^2)$, cũng chia hết cho $2027$.",
                        "Tiếp tục như vậy, tất cả các cặp số đều chia hết cho $2027$.",
                        "Vậy $A$ là tổng của $1013$ cặp số và mọi cặp số đều chia hết cho $2027$."
                    ], 
                "answer": "$\\boxed{A \\text{ chia hết cho } 2027}$"
            }
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
            {
                "id": "g12-06",
                "question": "Cho tứ diện $ABCD$, ở mỗi mặt, ta lấy các trọng tâm $A'$ ở $(BCD)$; $B'$ ở $(ACD)$; $C'$ ở $(ABD)$ và $D'$ ở $(ABC)$. Hãy tính tỷ lệ thể tích giữa 2 khối $A'B'C'D'$ và $ABCD$",
                "steps": [
                    "Chọn điểm $O$ làm gốc tọa độ. Gọi $\\vec{a}=\\overrightarrow{OA}$, $\\vec{b}=\\overrightarrow{OB}$, $\\vec{c}=\\overrightarrow{OC}$, $\\vec{d}=\\overrightarrow{OD}$ lần lượt là các vectơ vị trí của bốn đỉnh $A,B,C,D$.",
                    "Vì $A',B',C',D'$ lần lượt là trọng tâm các tam giác $BCD,ACD,ABD,ABC$, nên ta có: $\\vec{a'}=\\dfrac{\\vec{b}+\\vec{c}+\\vec{d}}{3}$, $\\vec{b'}=\\dfrac{\\vec{a}+\\vec{c}+\\vec{d}}{3}$, $\\vec{c'}=\\dfrac{\\vec{a}+\\vec{b}+\\vec{d}}{3}$ và $\\vec{d'}=\\dfrac{\\vec{a}+\\vec{b}+\\vec{c}}{3}$.",
                    "Ta có: $\\overrightarrow{A'B'}=\\overrightarrow{OB'} - \\overrightarrow{OA'} = \\vec{b'}-\\vec{a'} = \\dfrac{\\vec{a}-\\vec{b}}{3} = -\\dfrac{1}{3}\\overrightarrow{AB}$.",
                    "Tương tự, $\\overrightarrow{A'C'}=-\\dfrac{1}{3}\\overrightarrow{AC}$ và $\\overrightarrow{A'D'}=-\\dfrac{1}{3}\\overrightarrow{AD}$.",
                    "Như vậy, ba vectơ cạnh xuất phát từ $A'$ lần lượt bằng $-\\dfrac{1}{3}$ các vectơ cạnh tương ứng xuất phát từ $A$. Do đó, các cạnh tương ứng song song, ngược hướng và có độ dài bằng $\\dfrac{1}{3}$ cạnh ban đầu.",
                    "Suy ra tứ diện $A'B'C'D'$ đồng dạng với tứ diện $ABCD$ theo tỉ số đồng dạng $k=\\dfrac{1}{3}$.",
                    "Tỉ số thể tích bằng lập phương tỉ số đồng dạng: $\\dfrac{V_{A'B'C'D'}}{V_{ABCD}}=k^3=\\left(\\dfrac{1}{3}\\right)^3=\\dfrac{1}{27}$."
                ],
                "answer": "$\\boxed{\\dfrac{1}{27}}$"
            }
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
                "steps": ["Hoàn thành bình phương: $f(x)=-(x-2)^2+5$.", "Vì $(x-2)^2\\ge 0$, nên $f(x)\\le5$, đạt được khi $x=2$."],
                "answer": "$\\boxed{\\max f(x)=5\\text{ tại }x=2}$",
            },
            {
                "id": "a12-06",
                "question": "Cho số nguyên $k > 2$, trong các dãy sau thì dãy nào có giới hạn về 0 \n $\\qquad A. \\left( \\frac{2k+1}{2k+2} \\right)^k \\qquad B. (2k - 1)^k \\qquad C. (2k + 1)^k \\qquad D. \\left( \\frac{2k-1}{2k-2} \\right)^k$",
                "steps": [
                    "Đầu tiên, để ngắn gọn ta sẽ gán $a = \\frac{2k+1}{2k+2}; b = 2k-1; c = 2k+1$ và $d = \\frac{2k-1}{2k-2}$",
                    "Với $k > 2$ thì $b, c, d > 1$ chỉ riêng $a < 1$, một dãy $x^k$ hội tụ về $0$ nếu $x < 1$"
                ],
                "answer": "$\\boxed{\\text{A}\\, \\left( \\frac{2k+1}{2k+2} \\right)^k}$"
            },
            {
                "id": "a12-07",
                "question": "Cho $f(x)$ là một đa thức, biết rằng $f(x) - f'(x) = (x+1)^2$, hãy tìm giá trị của $f(2)$",
                "steps": [
                    "Vì $f$ là một đa thức và $f(x) - f'(x) = x^2 + 2x + 1$ nên bậc tối đa của đa thức là 2, ta gọi $f(x) = ax^2 + bx + c$",
                    "Khi đó, $f - f' = ax^2 + (b - 2a)x + (c - b)$",
                    "Đồng nhất thức, ta được $\\begin{cases}a = 1\\\\b-2a=2\\\\c-b=1 \\end{cases}$, suy ra $\\begin{cases}a=1\\\\b=4\\\\c=5 \\end{cases}$",
                    "Thay các giá trị tìm được, ta có $f(x) = x^2 + 4x + 5$; suy ra $f(2) = 17$", 
                ],
                "answer": "$\\boxed{17}$"
            }
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
            st.markdown('<div class="question-text">', unsafe_allow_html=True)
            st.markdown(item["question"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        # A unique key per question ensures independent reveal/hide behavior.
        if question_id in st.session_state.shown_solutions:
            if st.button("🙈  Hide solution", key=f"hide_{question_id}", use_container_width=False):
                st.session_state.shown_solutions.remove(question_id)
                st.rerun()
            st.markdown(
                '<div class="solution-card"><div class="solution-title">✓ Solution · Lời giải</div><div class="solution-body">',
                unsafe_allow_html=True,
            )
            if question_id.startswith("g"):
                img_col, step_cols = st.columns([2, 5])
                with step_cols:
                    for step_no, step in enumerate(item["steps"], start=1):
                        st.markdown(f"**Bước {step_no}.** {step}", unsafe_allow_html=True)
                    st.markdown(f"**Đáp án:** {item['answer']}")
                    st.markdown("</div></div>", unsafe_allow_html=True)
                with img_col:
                    class_level, lv_id = question_id.split("-")
                    img_path = Path(__file__).parent / "assets" / f"{class_level}" / f"{lv_id}.png"
                    if img_path.exists():
                        st.image(img_path)
                        st.caption('Hình minh họa')

            else:
                for step_no, step in enumerate(item["steps"], start=1):
                    st.markdown(f"**Bước {step_no}.** {step}", unsafe_allow_html=True)
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
