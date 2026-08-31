import streamlit as st

st.markdown("""
            <style>
                .block-container {
                    max-width:100%;
                    padding-top: 1.5rem;
                    padding-bottom: 0.5rem;
                }
            </style>
            """, unsafe_allow_html=True)
st.set_page_config(page_title="Mini Flappy", layout="wide")

game_html = """
    <!DOCTYPE html>
    <html>
        <head>
        <style>
            body {
                margin: 0;
                overflow: hidden;
                background: #70c5ce;
                font-family: Arial;
            }

            #game {
                display: block;
                margin: auto;
                background: linear-gradient(#70c5ce, #dff6ff);
                border: 3px solid #222;
                width: min(500px, calc(100vw - 32px));
                height: auto;
                box-sizing: border-box;
            }

            #hint {
                text-align: center;
                font-size: 16px;
                margin-top: 6px;
                color: #222;
            }

            #stats {
                width: 500px;
                margin: 0 auto 12px;
                display: flex;
                gap: 12px;
            }

            .stat-box {
                flex: 1;
                padding: 12px 16px;
                background: white;
                border: 2px solid #222;
                border-radius: 10px;
                color: #222;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
                box-sizing: border-box;
            }

        </style>
        </head>

        <body>
        <br>
        <div id="stats">
            <div class="stat-box">🎯 Score: <span id="score">0</span></div>
            <div class="stat-box">🏆 Best: <span id="best">0</span></div>
        </div>

        <canvas id="game" width="500" height="600"></canvas>
        <div id="hint">Nhấn Space hoặc click để bay — nhấn R để chơi lại</div>

        <script>
        const canvas = document.getElementById("game");
        const ctx = canvas.getContext("2d");
        const scoreElement = document.getElementById("score");
        const bestElement = document.getElementById("best");

        let bestScore = Number(localStorage.getItem("flappyBest")) || 0;
        bestElement.textContent = bestScore;

        let bird;
        let pipes;
        let score;
        let gameOver;
        let frame;

        function resetGame() {
            bird = {
                x: 80,
                y: 250,
                radius: 15,
                velocity: 0
            };

            pipes = [];
            score = 0;
            scoreElement.textContent = score;

            gameOver = false;
            frame = 0;
        }

        function flap() {
            if (gameOver) {
                resetGame();
            }

            bird.velocity = -7;
        }

        function addPipe() {
            const gap = 155;
            const minHeight = 70;
            const maxHeight = canvas.height - gap - 100;
            const topHeight = Math.floor(
                Math.random() * (maxHeight - minHeight) + minHeight
            );

            pipes.push({
                x: canvas.width,
                width: 60,
                top: topHeight,
                gap: gap,
                passed: false
            });
        }

        function collision(pipe) {
            const hitPipe =
                bird.x + bird.radius > pipe.x &&
                bird.x - bird.radius < pipe.x + pipe.width &&
                (
                    bird.y - bird.radius < pipe.top ||
                    bird.y + bird.radius > pipe.top + pipe.gap
                );

            const hitFloor =
                bird.y + bird.radius > canvas.height ||
                bird.y - bird.radius < 0;

            return hitPipe || hitFloor;
        }

        function update() {
            if (gameOver) return;

            bird.velocity += 0.35;
            bird.y += bird.velocity;

            if (frame % 95 === 0) {
                addPipe();
            }

            for (const pipe of pipes) {
                pipe.x -= 2.8;

                if (!pipe.passed && pipe.x + pipe.width < bird.x) {
                    pipe.passed = true;
                        score++;

                        scoreElement.textContent = score;

                        if (score > bestScore) {
                            bestScore = score;
                            bestElement.textContent = bestScore;
                            localStorage.setItem("flappyBest", bestScore);
                        }
                }

                if (collision(pipe)) {
                    gameOver = true;
                }
            }

            pipes = pipes.filter(pipe => pipe.x + pipe.width > 0);
            frame++;
        }

        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Chim
            ctx.fillStyle = "#35a7ff";
            ctx.beginPath();
            ctx.arc(bird.x, bird.y, bird.radius, 0, Math.PI * 2);
            ctx.fill();

            // Mỏ
            ctx.fillStyle = "#ff9f1c";
            ctx.beginPath();
            ctx.moveTo(bird.x + 12, bird.y);
            ctx.lineTo(bird.x + 28, bird.y + 6);
            ctx.lineTo(bird.x + 12, bird.y + 10);
            ctx.fill();

            // Ống – kiểu góc cạnh hơn (thân + cap)
            const pipeColor = "#2ecc40";
            const pipeBorder = "#1a7a28";
            const lipHeight = 22;   // độ cao của “môi” ống
            const lipExtra = 8;     // phần nhô ra hai bên

            for (const pipe of pipes) {
                // ===== Ống trên =====
                // Thân ống
                ctx.fillStyle = pipeColor;
                ctx.fillRect(pipe.x, 0, pipe.width, pipe.top);

                // Cap (môi dưới của ống trên)
                ctx.fillRect(
                    pipe.x - lipExtra,
                    pipe.top - lipHeight,
                    pipe.width + lipExtra * 2,
                    lipHeight
                );

                // Viền
                ctx.strokeStyle = pipeBorder;
                ctx.lineWidth = 3;
                ctx.strokeRect(pipe.x, 0, pipe.width, pipe.top);
                ctx.strokeRect(
                    pipe.x - lipExtra,
                    pipe.top - lipHeight,
                    pipe.width + lipExtra * 2,
                    lipHeight
                );

                // ===== Ống dưới =====
                const bottomY = pipe.top + pipe.gap;
                const bottomH = canvas.height - bottomY;

                // Thân ống
                ctx.fillStyle = pipeColor;
                ctx.fillRect(pipe.x, bottomY, pipe.width, bottomH);

                // Cap (môi trên của ống dưới)
                ctx.fillRect(
                    pipe.x - lipExtra,
                    bottomY,
                    pipe.width + lipExtra * 2,
                    lipHeight
                );

                // Viền
                ctx.strokeStyle = pipeBorder;
                ctx.lineWidth = 3;
                ctx.strokeRect(pipe.x, bottomY, pipe.width, bottomH);
                ctx.strokeRect(
                    pipe.x - lipExtra,
                    bottomY,
                    pipe.width + lipExtra * 2,
                    lipHeight
                );
            }

            if (gameOver) {
                ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                ctx.fillStyle = "white";
                ctx.textAlign = "center";
                ctx.font = "bold 36px Arial";
                ctx.fillText("Game Over", canvas.width / 2, 270);

                ctx.font = "20px Arial";
                ctx.fillText(
                    "Click hoặc Space để chơi lại",
                    canvas.width / 2,
                    315
                );

                ctx.textAlign = "left";
            }
        }

        function loop() {
            update();
            draw();
            requestAnimationFrame(loop);
        }

        document.addEventListener("keydown", event => {
            if (event.code === "Space") {
                event.preventDefault();
                flap();
            }

            if (event.key.toLowerCase() === "r") {
                resetGame();
            }
        });

        canvas.addEventListener("click", flap);

        resetGame();
        loop();
        </script>
        </body>
    </html>
"""

st.title(":yellow[🐤 Mini Flappy Bird]")
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.components.v1.html(game_html, height=725, scrolling=False)