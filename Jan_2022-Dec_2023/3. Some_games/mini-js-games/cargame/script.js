const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
canvas.width = 400;
canvas.height = 600;

/* =====================
   GAME STATE
===================== */
let gameState = 'menu'; // menu | playing | paused | gameover
let deathCount = 0;
let highScore = 0;

const COLOR_MAP = {
    red:    { css: '#ff0000', label: '🔴 ĐỎ' },
    blue:   { css: '#0000ff', label: '🔵 XANH' },
    yellow: { css: '#ffff00', label: '🟡 VÀNG' },
    purple: { css: '#800080', label: '🟣 TÍM' }
};

let player = {
    x: 180,
    y: 500,
    width: 40,
    height: 70,
    colorKey: 'red'
};

let playerName = 'Player';
let score = 0;
let level = 1;
let gameSpeed = 3;
let roadOffset = 0;
let keys = {};
let enemies = [];
let power = 100;

/* =====================
   UI ELEMENTS
===================== */
const startScreen = document.getElementById('startScreen');
const gameUI = document.getElementById('gameUI');
const gameOverScreen = document.getElementById('gameOverScreen');
const playerInfo = document.getElementById('playerInfo');
const pauseBtn = document.getElementById('pauseBtn');
const playBtn = document.getElementById('playBtn');

/* =====================
   INPUT
===================== */
window.addEventListener('keydown', e => keys[e.key] = true);
window.addEventListener('keyup', e => keys[e.key] = false);
playBtn.addEventListener('click', startGame);
pauseBtn.addEventListener('click', togglePause);

/* =====================
   COLOR SELECT
===================== */
function selectColor(colorKey) {
    if (COLOR_MAP[colorKey]) {
        player.colorKey = colorKey;
        updatePlayerInfo();
    }
}

/* =====================
   GAME CONTROL
===================== */
function startGame() {
    playerName = document.getElementById('playerName').value || 'Player';

    score = 0;
    level = 1;
    gameSpeed = 3;
    roadOffset = 0;
    enemies = [];
    player.x = 180;

    power = 100 - 5 * deathCount;
    if (power < 0) power = 0;

    startScreen.classList.add('hidden');
    gameOverScreen.classList.add('hidden');
    gameUI.classList.remove('hidden');

    pauseBtn.textContent = '⏸️ PAUSE';
    gameState = 'playing';
    updatePlayerInfo();
}

function togglePause() {
    if (gameState === 'playing') {
        gameState = 'paused';
        pauseBtn.textContent = '▶️ RESUME';
    } else if (gameState === 'paused') {
        gameState = 'playing';
        pauseBtn.textContent = '⏸️ PAUSE';
    }
}

function restartGame() {
    score = 0;
    level = 1;
    gameSpeed = 3;
    roadOffset = 0;
    enemies = [];
    keys = {};
    player.x = 180;

    power = 100 - 5 * deathCount;
    if (power < 0) power = 0;

    gameOverScreen.classList.add('hidden');
    startScreen.classList.add('hidden');
    gameUI.classList.remove('hidden');

    pauseBtn.textContent = '⏸️ PAUSE';
    gameState = 'playing';
    updatePlayerInfo();
}

function backToMenu() {
    gameState = 'menu';
    enemies = [];
    gameOverScreen.classList.add('hidden');
    gameUI.classList.add('hidden');
    startScreen.classList.remove('hidden');
}

/* =====================
   UI
===================== */
function updatePlayerInfo() {
    const color = COLOR_MAP[player.colorKey];
    playerInfo.textContent =
        `${playerName} - ${color.label} | Power: ${power}% | High Score: ${highScore}`;
}

/* =====================
   DRAW
===================== */
function drawBackground() {
    ctx.fillStyle = '#87CEEB';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#228B22';
    ctx.fillRect(0, 0, 50, canvas.height);
    ctx.fillRect(350, 0, 50, canvas.height);

    ctx.fillStyle = '#666';
    ctx.fillRect(50, 0, 300, canvas.height);
}

function drawRoadLines() {
    ctx.fillStyle = '#fff';
    for (let i = 0; i < 20; i++) {
        const y = (i * 40 - roadOffset) % 800;
        ctx.fillRect(195, y, 10, 30);
    }
}

function drawCar(x, y, colorCss, isPlayer = false) {
    const gradient = ctx.createLinearGradient(x, y, x, y + player.height);
    gradient.addColorStop(0, '#fff');
    gradient.addColorStop(0.3, colorCss);
    gradient.addColorStop(1, '#000');

    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, player.width, player.height);

    ctx.fillStyle = '#000';
    ctx.fillRect(x + 5, y + 5, 8, 8);
    ctx.fillRect(x + 27, y + 5, 8, 8);
    ctx.fillRect(x + 5, y + 55, 8, 8);
    ctx.fillRect(x + 27, y + 55, 8, 8);

    if (isPlayer) {
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.fillText(playerName.slice(0, 6), x + 3, y + 25);
    }
}

/* =====================
   ENEMIES (NO OVERLAP)
===================== */
function spawnEnemy() {
    if (Math.random() < 0.015 * level) {
        const lanes = [70, 150, 230, 310];
        const availableLanes = lanes.filter(lane =>
            !enemies.some(e => e.x === lane && e.y > -200 && e.y < 100)
        );

        if (availableLanes.length === 0) return;

        const lane = availableLanes[Math.floor(Math.random() * availableLanes.length)];
        enemies.push({
            x: lane,
            y: -80,
            width: 40,
            height: 70,
            color: ['#00ff00', '#0000ff', '#ff00ff'][Math.floor(Math.random() * 3)],
            speed: gameSpeed + Math.random() * 1.5
        });
    }
}

function updateEnemies() {
    for (let i = enemies.length - 1; i >= 0; i--) {
        enemies[i].y += enemies[i].speed;

        for (let j = i - 1; j >= 0; j--) {
            if (
                enemies[i].x === enemies[j].x &&
                Math.abs(enemies[i].y - enemies[j].y) < 80
            ) {
                enemies[i].y += 20;
                enemies[j].y -= 10;
            }
        }

        if (enemies[i].y > canvas.height) {
            enemies.splice(i, 1);
            score += 10;
            if (score > highScore) highScore = score;
        }
    }
}

/* =====================
   COLLISION & POWER
===================== */
function checkCollisions() {
    let hit = false;

    if (player.x < 55 || player.x > 345 - player.width) {
        power -= 30;
        hit = true;
    }

    for (let e of enemies) {
        if (
            player.x < e.x + e.width &&
            player.x + player.width > e.x &&
            player.y < e.y + e.height &&
            player.y + player.height > e.y
        ) {
            power -= 40;
            hit = true;
        }
    }

    if (power <= 0) {
        deathCount++;
        gameOver();
        return true;
    }

    if (hit) updatePlayerInfo();
    return false;
}

/* =====================
   SCORE
===================== */
function updateScore() {
    score += level;
    if (score > level * 1000) {
        level++;
        gameSpeed += 0.5;
    }
    if (score > highScore) highScore = score;
}

/* =====================
   UI DRAW
===================== */
function drawUI() {
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(10, 10, 260, 70);

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 16px Arial';
    ctx.fillText(`Score: ${score}`, 20, 30);
    ctx.fillText(`Level: ${level}`, 20, 50);
    ctx.fillText(`High Score: ${highScore}`, 130, 30);

    ctx.fillStyle = '#555';
    ctx.fillRect(130, 40, 120, 15);

    const g = ctx.createLinearGradient(130, 40, 250, 40);
    if (power > 60) {
        g.addColorStop(0, '#0f0');
        g.addColorStop(1, '#7f0');
    } else if (power > 30) {
        g.addColorStop(0, '#ff0');
        g.addColorStop(1, '#fa0');
    } else {
        g.addColorStop(0, '#f00');
        g.addColorStop(1, '#800');
    }

    ctx.fillStyle = g;
    ctx.fillRect(130, 40, 120 * (power / 100), 15);
}

/* =====================
   GAME OVER
===================== */
function gameOver() {
    gameState = 'gameover';
    document.getElementById('finalScore').innerHTML =
        `🎉 ${playerName}<br>Score: ${score}<br>Level: ${level}<br>High Score: ${highScore}`;

    gameOverScreen.classList.remove('hidden');
    gameUI.classList.add('hidden');
    updatePlayerInfo();
}

/* =====================
   GAME LOOP
===================== */
function gameLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (gameState === 'playing') {
        roadOffset += gameSpeed;

        if (keys['ArrowLeft'] && player.x > 55) {
            player.x -= 5;
        }
        if (keys['ArrowRight'] && player.x < 345 - player.width) {
            player.x += 5;
        }

        spawnEnemy();
        updateEnemies();
        updateScore();
        checkCollisions();
    }

    drawBackground();
    drawRoadLines();
    enemies.forEach(e => drawCar(e.x, e.y, e.color));
    drawCar(player.x, player.y, COLOR_MAP[player.colorKey].css, true);
    drawUI();

    requestAnimationFrame(gameLoop);
}

gameLoop();