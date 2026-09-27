import streamlit as st
import random
import numpy as np
import time
from typing import Set

st.set_page_config(
    page_title="Funny Math",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

config_col, play_col = st.columns([3, 4], gap = 'large')

with config_col:
    game_name = st.selectbox("Chọn game", ['PRIME HUNTER', 'ROOT DUEL'])
    if game_name == "ROOT DUEL":
        # ================== SESSION STATE ==================
        if "root_game" not in st.session_state:
            st.session_state.root_game = {
                "generated": False,
                "A": None,
                "B": None,
                "n": None,
                "correct": None,
                "root_level": 3,
                "finished": False,
                "is_correct": None,
                "game_id": 0,
            }

        # ================== HÀM PHÁT SINH GAME MỚI ==================
        def generate_new_game(level: int):
            n = int(np.random.randint(3, 25))
            diff = int(np.random.randint(4, 15))

            up = int(np.random.randint(1, diff))
            down = int(np.random.randint(1, diff))
            while up == down:
                down = int(np.random.randint(1, diff))

            n_pow = n ** level
            A = n_pow + up
            B = n_pow - down

            root_A = A ** (1 / level)
            root_B = B ** (1 / level)

            left  = root_A - n
            right = n - root_B

            if left > right:
                correct = ">"
            elif left < right:
                correct = "<"
            else:
                correct = "="

            st.session_state.root_game.update({
                "generated": True,
                "A": A,
                "B": B,
                "n": n,
                "correct": correct,
                "root_level": level,
                "finished": False,
                "is_correct": None,
                "game_id": st.session_state.root_game["game_id"] + 1,
            })

        # ---------- CONFIG COLUMN ----------
        with config_col:
            param_col, but_col = st.columns(2, gap='large')
            with param_col:
                root_level = st.selectbox("Chọn căn bậc", [2, 3], index=1, key="root_level_select")
            with but_col:
                st.write(" ")
                if st.button("🔄 New Game", type="primary", use_container_width=True):
                    generate_new_game(root_level)
                    st.rerun()

            # Game description
            st.markdown("---")
            st.markdown("### 📖 Mô tả game")
            st.markdown("""
            So sánh hai khoảng cách tới số nguyên **n**:
            
            - Bên trái:  $\\sqrt[k]{A} - n$  
            - Bên phải: $n - \\sqrt[k]{B}$
            
            Chọn dấu **>** hoặc **<** cho đúng.
            
            *Mẹo: Không cần tính căn chính xác – dùng ước lượng hoặc tính chất căn là đủ!*
            """)

        # ---------- PLAY COLUMN ----------
        with play_col:
            # Tạo game lần đầu
            if not st.session_state.root_game["generated"]:
                generate_new_game(root_level)
                st.rerun()

            game = st.session_state.root_game

            st.markdown(f"### So sánh hai khoảng cách tới **{game['n']}**")

            # CSS
            st.markdown("""
                <style>
                .katex { font-size: 2.0em !important; }

                .latex-box {
                    background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #B6D7A8 100%);
                    border: 1px solid rgba(148, 163, 184, 0.35);
                    border-radius: 14px;
                    padding: 18px 10px;
                    text-align: center;
                    margin-bottom: 8px;
                }

                /* Tăng font-size chữ trong button cột compare */
                div[data-testid="stHorizontalBlock"] button p {
                    font-size: 1.5rem !important;
                    font-weight: 700 !important;
                }

                /* Gradient background cho st.button */
                div.stButton > button {
                    background: linear-gradient(135deg, #064e3b 0%, #0f766e 50%, #14b8a6 100%) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 10px !important;
                    font-weight: 600 !important;
                    transition: all 0.25s ease !important;
                }

                div.stButton > button:hover {
                    background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 50%, #5eead4 100%) !important;
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(20, 184, 166, 0.45);
                }

                div.stButton > button:active {
                    transform: translateY(0);
                }
                </style>
                """, unsafe_allow_html=True)

            numA, compare, numB = st.columns([2.3, 1.9, 2.3], gap="medium")

            with numA:
                st.markdown('<div class="latex-box"> Num_A', unsafe_allow_html=True)
                if game["root_level"] == 2:
                    st.latex(rf"\sqrt{{{game['A']}}} - {game['n']}")
                else:
                    st.latex(rf"\sqrt[3]{{{game['A']}}} - {game['n']}")
                st.markdown('</div>', unsafe_allow_html=True)

            with compare:
                st.markdown('<div class="latex-box"> Select one of these buttons', unsafe_allow_html=True)
                st.write("")
                if not game["finished"]:
                    _, col_gt, _, col_lt, _ = st.columns([0.1, 4, 0.5, 4, 0.1])
                    with col_gt:
                        if st.button(
                            ">",
                            key=f"btn_gt_{game['game_id']}",
                            use_container_width=True,
                            disabled=game["finished"]       
                        ):
                            st.session_state.root_game["finished"] = True
                            st.session_state.root_game["is_correct"] = (">" == game["correct"])
                            st.rerun()

                    with col_lt:
                        if st.button(
                            "<",
                            key=f"btn_lt_{game['game_id']}",
                            use_container_width=True,
                            disabled=game["finished"]          # ← disable khi đã trả lời
                        ):
                            st.session_state.root_game["finished"] = True
                            st.session_state.root_game["is_correct"] = ("<" == game["correct"])
                            st.rerun()
                else:
                    chosen = ">" if game.get("is_correct") and game["correct"] == ">" else \
                            "<" if game.get("is_correct") and game["correct"] == "<" else game["correct"]
                    st.markdown(
                        f"<div style='text-align:center; font-size:2.4rem; font-weight:800; padding:12px 0;'>"
                        f"{chosen}</div>",
                        unsafe_allow_html=True
                    )

            with numB:
                st.markdown('<div class="latex-box"> Num_B', unsafe_allow_html=True)
                if game["root_level"] == 2:
                    st.latex(rf"{game['n']} - \sqrt{{{game['B']}}}")
                else:
                    st.latex(rf"{game['n']} - \sqrt[3]{{{game['B']}}}")
                st.markdown('</div>', unsafe_allow_html=True)

            # ---------- MESSAGE + NEW GAME (cuối trang) ----------
            if game["finished"]:
                st.markdown("---")
                mes_col, new_col = st.columns([2.5, 1], gap="medium")

                with mes_col:
                    if game["is_correct"]:
                        st.success("Chính xác! 🎉")
                        st.balloons()
                    else:
                        st.error(f"Sai rồi! Đáp án đúng là **`{game['correct']}`**")

                with new_col:
                    st.write("")  # spacer cho nút căn giữa hơn
                    if st.button("🎮 Chơi ván mới", use_container_width=True, type="primary"):
                        generate_new_game(root_level)
                        st.rerun()

    if game_name == 'PRIME HUNTER':
        # ==================== HÀM KIỂM TRA SỐ NGUYÊN TỐ ====================
        def is_prime(num: int) -> bool:
            if num < 2:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            for factor in range(3, int(num**0.5) + 1, 2):
                if num % factor == 0:
                    return False
            return True

        def get_primes_in_range(start: int, end: int) -> Set[int]:
            return {n for n in range(start, end + 1) if is_prime(n)}


        # ==================== SESSION STATE ====================
        def init_session():
            if "level" not in st.session_state:
                st.session_state.level = 1
            if "game_state" not in st.session_state:          # ready | playing | won | lost
                st.session_state.game_state = "ready"
            if "selected" not in st.session_state:             # các số đã click
                st.session_state.selected = set()
            if "correct_selected" not in st.session_state:     # các số nguyên tố đã chọn đúng
                st.session_state.correct_selected = set()
            if "start_time" not in st.session_state:
                st.session_state.start_time = None
            if "time_limit" not in st.session_state:
                st.session_state.time_limit = 0
            if "turns_left" not in st.session_state:
                st.session_state.turns_left = 0
            if "primes" not in st.session_state:
                st.session_state.primes = set()

        def reset_level(level: int = None):
            if level is None:
                level = st.session_state.level
            st.session_state.level = level
            start = (level - 1) * 100
            end = level * 100 - 1
            primes = get_primes_in_range(start, end)
            st.session_state.primes = primes
            st.session_state.time_limit = len(primes) * 3
            st.session_state.turns_left = len(primes) + 5
            st.session_state.selected = set()
            st.session_state.correct_selected = set()
            st.session_state.game_state = "ready"
            st.session_state.start_time = None

        init_session()
        if "primes" not in st.session_state or not st.session_state.primes:
            reset_level(1)

        # ==================== LOGIC CLICK ====================
        def handle_click(num: int):
            if st.session_state.game_state != "playing":
                return
            if num in st.session_state.selected:
                return  # đã chọn rồi

            st.session_state.selected.add(num)
            st.session_state.turns_left -= 1

            if num in st.session_state.primes:
                st.session_state.correct_selected.add(num)

            # Kiểm tra thắng / thua ngay sau mỗi click
            check_game_status()

        def check_game_status():
            if st.session_state.game_state != "playing":
                return

            # Hết giờ?
            elapsed = time.time() - st.session_state.start_time
            if elapsed >= st.session_state.time_limit:
                st.session_state.game_state = "lost"
                return

            # Hết lượt?
            if st.session_state.turns_left <= 0:
                # Nếu đã tìm đủ số nguyên tố thì vẫn thắng
                if len(st.session_state.correct_selected) == len(st.session_state.primes):
                    st.session_state.game_state = "won"
                else:
                    st.session_state.game_state = "lost"
                return

            # Đã tìm đủ số nguyên tố?
            if len(st.session_state.correct_selected) == len(st.session_state.primes):
                st.session_state.game_state = "won"

        # ==================== CSS ====================
        st.markdown("""
        <style>
        /* Ép style cho tất cả button trong board */
        div[data-testid="stHorizontalBlock"] div.stButton > button {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #B6D7A8 100%) !important;
            color: #042f2e !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 900 !important;
            font-size: 49px !important;
            height: 58px !important;
            transition: all 0.15s ease !important;
        }
        div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
            transform: scale(1.06) !important;
            box-shadow: 0 0 18px rgba(20, 184, 166, 0.75) !important;
        }

        /* Khi đã chọn (disabled) → giữ opacity */
        div[data-testid="stHorizontalBlock"] div.stButton > button:disabled {
            opacity: 1 !important;
            transform: none !important;
        }

        /* Màu cho ô đúng (có dấu ✓) – xanh dương tươi sáng */
        div[data-testid="stHorizontalBlock"] div.stButton > button[kind="primary"] {
            background: radial-gradient(circle at center, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 0 16px rgba(59, 130, 246, 0.85) !important;
        }

        /* Màu cho ô sai (có dấu ✗) – đỏ rực */
        div[data-testid="stHorizontalBlock"] div.stButton > button[kind="secondary"]:disabled {
            background: linear-gradient(135deg, #7f1d1d 0%, #ef4444 55%, #fca5a5 100%) !important;
            color: #fef2f2 !important;
            opacity: 1 !important;
        }
        /* ========== STYLE CHO st.metric ========== */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 55%, #B6D7A8 100%) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            box-shadow: 0 4px 12px rgba(15, 118, 110, 0.35) !important;
            text-align: center !important;
        }

        /* Label của metric – canh giữa */
        div[data-testid="stMetricLabel"] {
            text-align: center !important;
            justify-content: center !important;
            color: #042f2e !important;
            font-weight: 700 !important;
        }

        /* Value của metric – canh giữa */
        div[data-testid="stMetricValue"] {
            text-align: center !important;
            justify-content: center !important;
            color: #042f2e !important;
            font-weight: 900 !important;
            -webkit-text-stroke: 1.2px #ffffff !important;
            paint-order: stroke fill !important;
        }

        /* Delta (nếu có) cũng canh giữa */
        div[data-testid="stMetricDelta"] {
            text-align: center !important;
            justify-content: center !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ---------- PHẦN MÔ TẢ ----------
        st.write("##### 📖 Nhiệm vụ")
        st.markdown("""
        Tìm **tất cả số nguyên tố** trong bảng 10×10 bằng cách click vào ô.
        
        - Mỗi level gồm 100 số liên tiếp  
        - Thời gian & số lượt được tính theo số lượng số nguyên tố của level đó  
        - Click **Play** để bắt đầu đếm giờ  
        """)

        level = st.session_state.level
        start = (level - 1) * 100
        end = level * 100 - 1
        total_primes = len(st.session_state.primes)

        st.markdown(f"**Level {level}**: số từ `{start}` → `{end}`")
        st.markdown(f"**Số nguyên tố cần tìm**: `{total_primes}`")

        # Metric thời gian & lượt
        if st.session_state.game_state == "playing" and st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            time_left = max(0, int(st.session_state.time_limit - elapsed))
        else:
            time_left = st.session_state.time_limit

        m1, m2, s3 = st.columns(3)
        m1.metric("⏱ Thời gian còn", f"{time_left}s", delta=None)
        m2.metric("🎯 Lượt còn", st.session_state.turns_left)
        s3.metric("Đã tìm thấy", f"{len(st.session_state.correct_selected)} / {total_primes}")

        if st.session_state.game_state == "ready":
            st.info("Nhấn **Play** ở bên phải để bắt đầu!")
        elif st.session_state.game_state == "playing":
            st.success("Đang chơi… Hãy tìm hết số nguyên tố!")
        elif st.session_state.game_state == "won":
            st.success("🎉 Xuất sắc! Bạn đã vượt qua level này.")
        else:
            st.error("😢 Hết giờ hoặc hết lượt rồi.")


        # ---------- CỘT CHƠI ----------
        with play_col:
            level_col, _, button_col = st.columns([2, 1, 2], gap='medium')
            with level_col:
                st.header(f"🎮 Level {st.session_state.level}")

            # Nút Play
            with button_col:
                if st.session_state.game_state == "ready":
                    if st.button("▶  Play", type="primary", use_container_width=True):
                        st.session_state.game_state = "playing"
                        st.session_state.start_time = time.time()
                        st.rerun()

            if st.session_state.game_state == "playing":
                check_game_status()

            # ===== BOARD CLICK TRỰC TIẾP =====
            start_num = (st.session_state.level - 1) * 100

            for row in range(10):
                cols = st.columns(10)
                for col_idx, col in enumerate(cols):
                    num = start_num + row * 10 + col_idx

                    if num in st.session_state.correct_selected:
                        label = f"✓ {num}"
                        disabled = True
                    elif num in st.session_state.selected:
                        label = f"✗ {num}"
                        disabled = True
                    else:
                        label = str(num)
                        disabled = st.session_state.game_state != "playing"

                    with col:
                        if num in st.session_state.correct_selected:
                            label = f"✓ {num}"
                            btn_type = "primary"          # ← dùng primary cho đúng
                            disabled = True
                        elif num in st.session_state.selected:
                            label = f"✗ {num}"
                            btn_type = "secondary"        # ← secondary cho sai
                            disabled = True
                        else:
                            label = str(num)
                            btn_type = "secondary"
                            disabled = st.session_state.game_state != "playing"

                        if st.button(
                            label,
                            key=f"btn_{st.session_state.level}_{num}",
                            type=btn_type,                # ← quan trọng
                            disabled=disabled,
                            use_container_width=True
                        ):
                            handle_click(num)
                            st.rerun()

            # ---------- KẾT THÚC GAME ----------
            if st.session_state.game_state == "won":
                st.balloons()
                st.success("🎉 Xuất sắc! Bạn đã vượt qua level này.")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.session_state.level < 10:
                        if st.button("➡️ Next Level", type="primary", use_container_width=True):
                            reset_level(st.session_state.level + 1)
                            st.rerun()
                    else:
                        st.success("🏆 Bạn đã hoàn thành cả 10 level!")
                with c2:
                    if st.button("🔄 Chơi lại Level này", use_container_width=True):
                        reset_level(st.session_state.level)
                        st.rerun()

            elif st.session_state.game_state == "lost":
                st.error("😢 Hết giờ hoặc hết lượt rồi. Đừng nản lòng nhé!")
                if st.button("🔄 Try Again", type="primary", use_container_width=True):
                    reset_level(st.session_state.level)
                    st.rerun()