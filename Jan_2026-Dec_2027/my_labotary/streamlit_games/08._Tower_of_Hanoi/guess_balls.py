import streamlit as st
import random

def run():
    # ============================================================
    # STYLE
    # ============================================================
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }

            /* Main title */
            h1 {
                margin-bottom: 0.3rem;
            }

            /* Section titles */
            h2, h3 {
                margin-top: 0.8rem;
            }

            /* Metric cards */
            [data-testid="stMetric"] {
                background: #062b4f;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 12px 16px;
            }

            /* Divider */
            hr {
                margin: 1.2rem 0;
            }

            /* History rows */
            .history-row {
                background: #062b4f;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 10px 14px;
                margin-bottom: 8px;
            }

            /* Info box */
            .game-info {
                background: #062b4f;
                border-radius: 10px;
                padding: 12px 16px;
                margin: 10px 0 18px 0;
            }

            /* Secret code */
            .secret-code {
                font-size: 1.4rem;
                font-weight: 600;
                letter-spacing: 6px;
            }

            /* Small label */
            .small-label {
                color: #64748b;
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # CONSTANTS
    # ============================================================
    COLORS = {
        "🔴 Red": "red",
        "🟢 Green": "green",
        "🟡 Yellow": "yellow",
        "🔵 Blue": "blue",
        "🟣 Violet": "violet",
        "⚪ White": "white",
    }

    COLOR_OPTIONS = list(COLORS.keys())
    ALL_COLOR_OPTIONS = list(COLORS.keys())

    # ============================================================
    # GAME FUNCTIONS
    # ============================================================
    def new_game():
        code_len = st.session_state.code_length

        # 1. Random chọn đúng code_len màu từ 6 màu gốc
        selected_labels = random.sample(ALL_COLOR_OPTIONS, code_len)
        st.session_state.active_colors = {label: COLORS[label] for label in selected_labels}
        st.session_state.color_options = selected_labels

        # 2. Secret code là hoán vị của đúng các màu đã chọn
        st.session_state.secret_code = random.sample(
            list(st.session_state.active_colors.values()), code_len
        )

        st.session_state.history = []
        st.session_state.game_over = False
        st.session_state.won = False

    def evaluate_guess(guess, secret):
        exact = sum(g == s for g, s in zip(guess, secret))

        remaining_guess = [g for g, s in zip(guess, secret) if g != s]
        remaining_secret = [s for g, s in zip(guess, secret) if g != s]

        color_only = sum(
            min(remaining_guess.count(color), remaining_secret.count(color))
            for color in set(remaining_guess)
        )
        return exact, color_only

    def color_balls(code):
        """Convert color values back to emoji balls."""
        return " ".join(
            next(
                label.split()[0]
                for label, value in COLORS.items()
                if value == color
            )
            for color in code
        )

    # ============================================================
    # KHỞI TẠO SESSION STATE AN TOÀN
    # ============================================================
    if "code_length" not in st.session_state:
        st.session_state.code_length = 4
        st.session_state.max_tries = 10
        new_game()

    # ============================================================
    # MAIN LAYOUT
    # ============================================================
    config_col, play_col = st.columns([2, 3], gap="large")

    # ============================================================
    # LEFT COLUMN — CONFIG + GUIDE
    # ============================================================
    with config_col:
        st.title("🎨 Code Breaker")

        st.caption(
            "Break the secret color code by finding the correct colors "
            "and their exact positions."
        )

        st.subheader("⚙️ Cấu hình trò chơi")

        col1, col2 = st.columns(2)

        with col1:
            code_length = st.selectbox(
                "Số bóng",
                options=[4, 5, 6],
                index=0,
                key="code_length_selector",
            )

        with col2:
            default_max_tries = {
                4: 10,
                5: 12,
                6: 15,
            }[code_length]

            max_tries = st.number_input(
                "Số lượt tối đa",
                min_value=5,
                max_value=30,
                value=default_max_tries,
                step=1,
                key="max_tries_input",
            )

        # Initialize / reset when configuration changes
        if (
            "code_length" not in st.session_state
            or st.session_state.code_length != code_length
            or st.session_state.max_tries != max_tries
        ):
            st.session_state.code_length = code_length
            st.session_state.max_tries = max_tries
            new_game()

        st.markdown(
            f"""
            <div class="game-info">
                <div class="small-label">CURRENT GAME</div>
                <b>{code_length} màu</b>
                &nbsp; • &nbsp;
                <b>{max_tries} lượt</b>
                &nbsp; • &nbsp;
                Không lặp màu
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # HOW TO PLAY
        # --------------------------------------------------------
        hint_col, trick_col = st.columns([4, 5], gap="small")
        with hint_col:
            st.write("#### 📘 Hướng dẫn")
            st.markdown(
                """
                **Mục tiêu:** tìm chính xác dãy màu bí mật.

                - Máy tạo một dãy gồm **4 / 5 / 6 màu**.
                - Mỗi màu chỉ xuất hiện **một lần**.
                - Mỗi lượt, chọn màu cho từng vị trí.
                - Sau khi kiểm tra, hệ thống trả về:
                - **✅** — đúng màu, đúng vị trí.
                - **🔄** — đúng màu nhưng sai vị trí.
                - Thắng khi tất cả các bóng đều **✅**.
                """
            )
        with trick_col:
            st.write("#### 💡 Mẹo chơi")
            st.markdown(
                """
                - **Lượt 1–2:** ưu tiên các màu khác nhau để quét thông tin.
                - Dùng tổng **✅ + 🔄** để biết có bao nhiêu màu nằm trong code.
                - Khi đã có **✅**, nên giữ nguyên vị trí đó.
                - Với các màu đã xác định, thử **hoán vị vị trí**.
                - Tránh lặp lại các cấu hình đã bị loại.
                - Khi còn ít lượt, thay đổi ít vị trí để giữ lại thông tin đã biết.
                """
            )

    # ============================================================
    # RIGHT COLUMN — GAME PLAY
    # ============================================================
    with play_col:

        # --------------------------------------------------------
        # GAME STATUS
        # --------------------------------------------------------
        tries_used = len(st.session_state.history)
        tries_left = st.session_state.max_tries - tries_used

        st.write(" ")
        st.markdown(
            """
            <span style='color:#ffa07a; font-size:23px; font-family:Consolas, monospace; font-weight:800; letter-spacing:0.5px;'>
                <br> </span>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([2, 2, 3], gap="small")

        with col1:
            st.metric("Lượt đã dùng", f"{tries_used}/{st.session_state.max_tries}")

        with col2:
            st.metric("Lượt còn lại", tries_left)

        # --------------------------------------------------------
        # CURRENT GUESS
        # --------------------------------------------------------
        st.write("-------------")
        if not st.session_state.game_over:
            guess_col, button_col = st.columns([3, 2], gap="small")
            with guess_col:
                st.markdown(
                    """
                    🎯 <span style='font-size:29px; font-family:Consolas, monospace; font-weight:800; letter-spacing:0.5px;'>
                            Lượt đoán hiện tại </span>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                f"Chọn {st.session_state.code_length} màu "
                "theo thứ tự bạn muốn kiểm tra."
            )

            selected_colors = []

            columns = st.columns(
                st.session_state.code_length,
                gap="small",
            )

            for i, col in enumerate(columns):
                with col:
                    choice = st.selectbox(
                        label=f"Ô {i + 1}",
                        options=st.session_state.color_options,
                        key=f"guess_{i}",
                        label_visibility="visible",
                    )
                    selected_colors.append(COLORS[choice])

            with col3:
                st.markdown(
                    """
                    <div class="game-info">
                        <b>🧠 Feedback</b><br>
                        ✅ Đúng màu + đúng vị trí
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        🔄 Đúng màu + sai vị trí
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with button_col:
                if st.button(
                    "🔍 Kiểm tra đáp án",
                    type="primary",
                    use_container_width=True,
                ):
                    exact, color_only = evaluate_guess(
                        selected_colors,
                        st.session_state.secret_code,
                    )

                    st.session_state.history.append(
                        {
                            "guess": selected_colors,
                            "exact": exact,
                            "color_only": color_only,
                        }
                    )

                    if exact == st.session_state.code_length:
                        st.session_state.won = True
                        st.session_state.game_over = True

                    elif len(st.session_state.history) >= st.session_state.max_tries:
                        st.session_state.game_over = True

                    st.rerun()

        # --------------------------------------------------------
        # HISTORY
        # --------------------------------------------------------
        st.divider()
        st.subheader("📜 Lịch sử đoán")

        if not st.session_state.history:
            st.info("Chưa có lượt đoán nào.")
        else:
            for turn, item in enumerate(reversed(st.session_state.history), start=1):
                real_turn = len(st.session_state.history) - turn + 1
                balls = color_balls(item["guess"])

                c1, c2, c3 = st.columns([3, 1, 1], gap="small")

                with c1:
                    st.markdown(
                        f"""
                        <div class="history-row">
                            <b>Lượt {real_turn}</b>
                            &nbsp;&nbsp;
                            {balls}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown(
                        f"""
                        Đúng vị trí : {item["exact"]}
                        """,
                        unsafe_allow_html=True,
                    )

                with c3:
                    st.markdown(
                        f"""
                        Sai vị trí : {item["color_only"]}
                        """,
                        unsafe_allow_html=True,
                    )

        # --------------------------------------------------------
        # GAME RESULT
        # --------------------------------------------------------
        if st.session_state.game_over:
            st.divider()

            secret_balls = color_balls(st.session_state.secret_code)

            if st.session_state.won:
                st.success(
                    f"🎉 Chúc mừng! Bạn đã phá mã "
                    f"trong **{tries_used} lượt**."
                )
            else:
                st.error("💥 Hết lượt! Bạn chưa phá được mã.")

            st.markdown(
                f"""
                **🔐 Đáp án:**  
                <div class="secret-code">
                    {secret_balls}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🔄 Chơi ván mới", use_container_width=True):
                new_game()
                st.rerun()