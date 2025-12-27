const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");

const ROW = 24;
const COL = 15;
const SQ = 24;
const VACANT = "#020617";

let board = [];
let score = 0;
let level = 1;
let dropInterval = 600;
let dropStart = Date.now();
let gameOver = false;
let isPaused = false;

function initBoard() {
  board = Array.from({ length: ROW }, () =>
    new Array(COL).fill(VACANT)
  );
}

function drawSquare(x, y, color) {
  // fill
  ctx.fillStyle = color;
  ctx.fillRect(x * SQ, y * SQ, SQ, SQ);

  // neon grid
  ctx.strokeStyle = "rgba(34, 211, 238, 0.25)"; // cyan neon blur
  ctx.lineWidth = 1;

  ctx.shadowColor = "rgba(34, 211, 238, 0.35)";
  ctx.shadowBlur = 4;

  ctx.strokeRect(
    x * SQ + 0.5,
    y * SQ + 0.5,
    SQ - 1,
    SQ - 1
  );

  // reset shadow (QUAN TRỌNG)
  ctx.shadowBlur = 0;
}

function drawBoard() {
  board.forEach((row, r) =>
    row.forEach((color, c) => drawSquare(c, r, color))
  );
}

const PIECES = [
  [[[0,1,0],[1,1,1],[0,0,0]], "#22d3ee"],
  [[[1,1],[1,1]], "#facc15"],
  [[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], "#4ade80"],
  [[[0,0,1],[1,1,1],[0,0,0]], "#6366f1"],
  [[[1,0,0],[1,1,1],[0,0,0]], "#38bdf8"],
  [[[0,1,1],[1,1,0],[0,0,0]], "#a855f7"],
  [[[1,1,0],[0,1,1],[0,0,0]], "#ec4899"],
];

function randomPiece() {
  const r = Math.floor(Math.random() * PIECES.length);
  return new Piece(PIECES[r][0], PIECES[r][1]);
}

function Piece(matrix, color) {
  this.matrix = matrix;
  this.color = color;
  this.x = 6;
  this.y = -2;
}

Piece.prototype.draw = function () {
  this.matrix.forEach((row, r) =>
    row.forEach((v, c) => {
      if (v) drawSquare(this.x + c, this.y + r, this.color);
    })
  );
};

Piece.prototype.unDraw = function () {
  this.matrix.forEach((row, r) =>
    row.forEach((v, c) => {
      if (v) drawSquare(this.x + c, this.y + r, VACANT);
    })
  );
};

Piece.prototype.collision = function (dx, dy, m = this.matrix) {
  return m.some((row, r) =>
    row.some((v, c) => {
      if (!v) return false;
      let x = this.x + c + dx;
      let y = this.y + r + dy;
      return x < 0 || x >= COL || y >= ROW || (y >= 0 && board[y][x] !== VACANT);
    })
  );
};

Piece.prototype.lock = function () {
  this.matrix.forEach((row, r) =>
    row.forEach((v, c) => {
      if (v) board[this.y + r][this.x + c] = this.color;
    })
  );

  let lines = 0;
  for (let r = 0; r < ROW; r++) {
    if (board[r].every(cell => cell !== VACANT)) {
      board.splice(r, 1);
      board.unshift(new Array(COL).fill(VACANT));
      lines++;
    }
  }

  if (lines) {
    score += lines * 10;
    level = Math.floor(score / 50) + 1;
    dropInterval = Math.max(150, 600 - level * 60);
    updateUI();
  }

  drawBoard();
  currentPiece = randomPiece();
};

Piece.prototype.moveDown = function () {
  if (!this.collision(0, 1)) {
    this.unDraw();
    this.y++;
    this.draw();
  } else {
    this.lock();
  }
};

Piece.prototype.moveLeft = function () {
  if (!this.collision(-1, 0)) {
    this.unDraw();
    this.x--;
    this.draw();
  }
};

Piece.prototype.moveRight = function () {
  if (!this.collision(1, 0)) {
    this.unDraw();
    this.x++;
    this.draw();
  }
};

Piece.prototype.rotate = function () {
  const N = this.matrix.length;
  const rotated = this.matrix.map((_, i) =>
    this.matrix.map(row => row[i]).reverse()
  );
  if (!this.collision(0, 0, rotated)) {
    this.unDraw();
    this.matrix = rotated;
    this.draw();
  }
};

let currentPiece = randomPiece();

function updateUI() {
  document.getElementById("score").textContent = score;
  document.getElementById("level").textContent = level;
}

function drop() {
  if (!isPaused && Date.now() - dropStart > dropInterval) {
    currentPiece.moveDown();
    dropStart = Date.now();
  }
  if (!gameOver) requestAnimationFrame(drop);
}

document.getElementById("startBtn").onclick = () => {
  initBoard();
  drawBoard();
  score = 0;
  level = 1;
  dropInterval = 600;
  updateUI();
  gameOver = false;
  isPaused = false;
  currentPiece = randomPiece();
  dropStart = Date.now();
  drop();
};

document.getElementById("pauseBtn").onclick = () => {
  isPaused = !isPaused;
};

document.addEventListener("keydown", e => {
  if (isPaused) return;
  if (e.key === "ArrowLeft") currentPiece.moveLeft();
  if (e.key === "ArrowRight") currentPiece.moveRight();
  if (e.key === "ArrowDown") currentPiece.moveDown();
  if (e.key === "ArrowUp") currentPiece.rotate();
});

document.querySelectorAll(".controls button").forEach(btn => {
  btn.onclick = () => {
    if (isPaused) return;
    const a = btn.dataset.action;
    if (a === "left") currentPiece.moveLeft();
    if (a === "right") currentPiece.moveRight();
    if (a === "down") currentPiece.moveDown();
    if (a === "rotate") currentPiece.rotate();
  };
});

initBoard();
drawBoard();