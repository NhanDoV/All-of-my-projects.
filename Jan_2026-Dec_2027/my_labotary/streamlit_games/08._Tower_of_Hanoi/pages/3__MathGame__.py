import streamlit as st
import random
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
        st.warning("Sẽ sớm update")

    elif game_name == 'PRIME HUNTER':
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
            background: linear-gradient(135deg, #166534 0%, #4ade80 55%, #fef08a 100%) !important;
            color: #14532d !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 900 !important;
            font-size: 29px !important;
            height: 58px !important;
            transition: all 0.15s ease !important;
        }
        div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
            transform: scale(1.06) !important;
            box-shadow: 0 0 14px rgba(74, 222, 128, 0.55) !important;
        }

        /* Khi đã chọn (disabled) → chuyển sang xanh dương hoặc đỏ tùy label */
        div[data-testid="stHorizontalBlock"] div.stButton > button:disabled {
            opacity: 1 !important;
            transform: none !important;
        }

        /* Màu cho ô đúng (có dấu ✓) */
        div[data-testid="stHorizontalBlock"] div.stButton > button[kind="primary"] {
            background: radial-gradient(circle at center, #1e3a8a 0%, #3b82f6 55%, #93c5fd 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.7) !important;
        }

        /* Màu cho ô sai (có dấu ✗) */
        div[data-testid="stHorizontalBlock"] div.stButton > button[kind="secondary"]:disabled {
            background: linear-gradient(135deg, #7f1d1d, #ef4444) !important;
            color: #fecaca !important;
            opacity: 1 !important;
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