const SIZE = 9;
let board = [];
let fullSolution = [];
let puzzle = [];
let selectedCell = null;

// =====================
// Tạo Sudoku solution mới
// =====================
function generateFullSolution(grid = null) {
    let g = grid ? grid.map(row => [...row]) : Array.from({ length: SIZE }, () => Array(SIZE).fill(0));

    function solve(grid) {
        for (let row = 0; row < SIZE; row++) {
            for (let col = 0; col < SIZE; col++) {
                if (grid[row][col] === 0) {
                    let nums = shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9]);
                    for (let num of nums) {
                        if (isValid(grid, row, col, num)) {
                            grid[row][col] = num;
                            if (solve(grid)) return true;
                            grid[row][col] = 0;
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    solve(g);
    return g;
}

// =====================
// Shuffle array
// =====================
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// =====================
// Check số hợp lệ
// =====================
function isValid(grid, row, col, num) {
    for (let i = 0; i < SIZE; i++) {
        if (grid[row][i] === num) return false;
        if (grid[i][col] === num) return false;
    }
    const startRow = Math.floor(row / 3) * 3;
    const startCol = Math.floor(col / 3) * 3;
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            if (grid[startRow + i][startCol + j] === num) return false;
        }
    }
    return true;
}

// =====================
// Generate puzzle playable
// =====================
function generatePuzzle() {
    fullSolution = generateFullSolution();
    puzzle = fullSolution.map(row => [...row]);

    // List tất cả ô
    let cells = [];
    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            cells.push([i, j]);
        }
    }

    cells = shuffle(cells);

    let removed = 0;
    const maxRemove = 40;

    for (let [row, col] of cells) {
        if (removed >= maxRemove) break;

        let rowCount = puzzle[row].filter(v => v !== 0).length;
        let colCount = puzzle.map(r => r[col]).filter(v => v !== 0).length;

        // Nếu row và col còn ít nhất 1 số thì remove
        if (rowCount > 1 && colCount > 1) {
            puzzle[row][col] = 0;
            removed++;
        }
    }

    board = puzzle.map(row => [...row]);
    return board;
}

// =====================
// Tạo board HTML
// =====================
function createBoard() {
    const boardElement = document.getElementById('sudoku-board');
    boardElement.innerHTML = '';

    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = i;
            cell.dataset.col = j;

            if (puzzle[i][j] !== 0) {
                cell.textContent = puzzle[i][j];
                cell.classList.add('filled');
            }

            cell.addEventListener('click', () => selectCell(i, j, cell));
            boardElement.appendChild(cell);
        }
    }
}

function selectCell(row, col, cell) {
    if (selectedCell) selectedCell.classList.remove('selected');
    selectedCell = cell;
    cell.classList.add('selected');
}

// =====================
// Number pad input
// =====================
function createNumberPad() {
    const pad = document.getElementById('number-pad');
    pad.innerHTML = '';
    for (let i = 1; i <= 9; i++) {
        const btn = document.createElement('button');
        btn.className = 'number-btn';
        btn.textContent = i;
        btn.addEventListener('click', () => inputNumber(i));
        pad.appendChild(btn);
    }
}

function inputNumber(num) {
    if (!selectedCell || selectedCell.classList.contains('filled')) return;

    const row = parseInt(selectedCell.dataset.row);
    const col = parseInt(selectedCell.dataset.col);

    // Xóa số cũ tạm thời để check valid
    const old = board[row][col];
    board[row][col] = 0;

    if (!isValid(board, row, col, num)) {
        selectedCell.classList.add('error');
        board[row][col] = old;
        setTimeout(() => selectedCell.classList.remove('error'), 500);
        selectedCell.textContent = '';
        return;
    }

    // Số hợp lệ → ghi vào board
    board[row][col] = num;
    selectedCell.textContent = num;

    checkWin();
}

// =====================
// Check win
// =====================
function checkWin() {
    const isComplete = board.every(row => row.every(cell => cell !== 0));
    if (!isComplete) return;

    const isCorrect = board.every((row, i) =>
        row.every((cell, j) => cell === fullSolution[i][j])
    );

    const status = document.getElementById('status');
    if (isCorrect) {
        status.textContent = '🎉 CHÚC MỪNG! Bạn đã hoàn thành Sudoku! 🎉';
        status.style.color = '#43e97b';
        setTimeout(generateNewPuzzle, 2000);
    } else {
        status.textContent = '❌ GAME OVER! Sudoku không đúng. Thử lại!';
        status.style.color = '#ff4757';
        setTimeout(generateNewPuzzle, 2000);
    }
}

// =====================
// Generate new game
// =====================
function generateNewPuzzle() {
    board = generatePuzzle();
    createBoard();
    createNumberPad();
    document.getElementById('status').textContent = 'Click ô trống để chọn, sau đó click số 1-9!';
    document.getElementById('status').style.color = 'white';
    selectedCell = null;
}

// =====================
// Load lần đầu
// =====================
window.onload = function () {
    generateNewPuzzle();
};