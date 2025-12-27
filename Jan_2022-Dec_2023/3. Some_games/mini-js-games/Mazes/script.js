class MazeGame {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.level = 1;
        this.cols = 5;   // khởi đầu số lẻ
        this.rows = 5;  // khởi đầu số lẻ
        this.cellSize = 30;
        this.player = { x: 1, y: 1, size: 0.6 };
        this.finish = { x: this.cols - 2, y: this.rows - 2 };
        this.keys = {};
        this.maze = [];
        this.gameRunning = true;

        // Popup winScreen
        this.winScreen = document.getElementById('winScreen');
        this.finalLevelSpan = document.getElementById('finalLevel');

        // Ẩn popup lúc bắt đầu
        this.winScreen.classList.remove('show');
        this.winScreen.style.display = 'none';

        this.init();
        this.generateMazeWithCheck();
        this.setupEventListeners();
        this.gameLoop();

        // Resize window tự động scale canvas
        window.addEventListener('resize', () => this.updateCanvasSize());
    }

    init() {
        this.updateCanvasSize();
        document.getElementById('level').textContent = this.level;
        document.getElementById('size').textContent = `${this.cols}x${this.rows}`;
    }

    updateCanvasSize() {
        this.canvas.width = this.cols * this.cellSize;
        this.canvas.height = this.rows * this.cellSize;

        // Tự động scale canvas vừa container
        const containerWidth = this.canvas.parentElement.clientWidth - 20;
        const containerHeight = window.innerHeight * 0.7;
        const scaleX = containerWidth / this.canvas.width;
        const scaleY = containerHeight / this.canvas.height;
        this.scale = Math.min(scaleX, scaleY, 1);
        this.canvas.style.transform = `scale(${this.scale})`;
        this.canvas.style.transformOrigin = 'top left';
    }

    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            this.keys[e.key.toLowerCase()] = true;
        });

        document.addEventListener('keyup', (e) => {
            this.keys[e.key.toLowerCase()] = false;
        });

        document.getElementById('nextLevel').addEventListener('click', () => {
            this.nextLevel();
        });
    }

    generateMaze() {
        this.maze = Array(this.rows).fill().map(() => Array(this.cols).fill(1));
        const stack = [];
        let current = { x: 1, y: 1 };
        this.maze[1][1] = 0;

        while (true) {
            const neighbors = this.getUnvisitedNeighbors(current);
            if (neighbors.length > 0) {
                const next = neighbors[Math.floor(Math.random() * neighbors.length)];
                this.removeWall(current, next);
                stack.push(current);
                current = next;
                this.maze[current.y][current.x] = 0;
            } else if (stack.length > 0) {
                current = stack.pop();
            } else {
                break;
            }
        }

        this.maze[1][1] = 0;
        this.maze[this.finish.y][this.finish.x] = 0;
    }

    getUnvisitedNeighbors(cell) {
        const neighbors = [];
        const dirs = [
            { x: 0, y: -2 },
            { x: 2, y: 0 },
            { x: 0, y: 2 },
            { x: -2, y: 0 }
        ];
        for (let dir of dirs) {
            const nx = cell.x + dir.x;
            const ny = cell.y + dir.y;
            if (nx > 0 && nx < this.cols - 1 && ny > 0 && ny < this.rows - 1 && this.maze[ny][nx] === 1) {
                neighbors.push({ x: nx, y: ny });
            }
        }
        return neighbors;
    }

    removeWall(a, b) {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        if (dx === 2) this.maze[a.y][a.x + 1] = 0;
        if (dx === -2) this.maze[a.y][a.x - 1] = 0;
        if (dy === 2) this.maze[a.y + 1][a.x] = 0;
        if (dy === -2) this.maze[a.y - 1][a.x] = 0;
    }

    isReachable() {
        const visited = Array(this.rows).fill().map(() => Array(this.cols).fill(false));
        const queue = [{ x: 1, y: 1 }];
        visited[1][1] = true;

        const directions = [
            { x: 0, y: -1 },
            { x: 1, y: 0 },
            { x: 0, y: 1 },
            { x: -1, y: 0 }
        ];

        while (queue.length > 0) {
            const current = queue.shift();
            if (current.x === this.finish.x && current.y === this.finish.y) return true;
            for (let dir of directions) {
                const nx = current.x + dir.x;
                const ny = current.y + dir.y;
                if (nx >= 0 && nx < this.cols && ny >= 0 && ny < this.rows &&
                    !visited[ny][nx] && this.maze[ny][nx] === 0) {
                    visited[ny][nx] = true;
                    queue.push({ x: nx, y: ny });
                }
            }
        }
        return false;
    }

    generateMazeWithCheck() {
        do {
            this.generateMaze();
        } while (!this.isReachable());
    }

    movePlayer() {
        if (!this.gameRunning) return;

        const oldX = this.player.x;
        const oldY = this.player.y;

        if (this.keys['w'] || this.keys['arrowup']) this.player.y -= 1;
        if (this.keys['s'] || this.keys['arrowdown']) this.player.y += 1;
        if (this.keys['a'] || this.keys['arrowleft']) this.player.x -= 1;
        if (this.keys['d'] || this.keys['arrowright']) this.player.x += 1;

        if (this.player.x < 1) this.player.x = 1;
        if (this.player.y < 1) this.player.y = 1;
        if (this.player.x > this.cols - 2) this.player.x = this.cols - 2;
        if (this.player.y > this.rows - 2) this.player.y = this.rows - 2;

        if (this.maze[this.player.y][this.player.x] === 1) {
            this.player.x = oldX;
            this.player.y = oldY;
        }

        if (this.player.x === this.finish.x && this.player.y === this.finish.y) {
            this.win();
        }
    }

    win() {
        this.gameRunning = false;
        this.finalLevelSpan.textContent = this.level;
        this.winScreen.style.display = 'block';
        this.winScreen.classList.add('show');
    }

    nextLevel() {
        // nếu đã max size thì dừng game, hiện MAX LEVEL
        if (this.cols >= 13 && this.rows >= 15) {
            this.finalLevelSpan.textContent = `MAX LEVEL ${this.level}`;
            this.winScreen.style.display = 'block';
            this.winScreen.classList.add('show');
            this.gameRunning = false;
            return;
        }

        this.level++;
        if (this.level % 2 === 0) this.cols += 2;
        else this.rows += 2;

        this.cols = Math.min(this.cols, 13);  // max limit
        this.rows = Math.min(this.rows, 15);  // max limit

        this.player = { x: 1, y: 1, size: 0.6 };
        this.finish = { x: this.cols - 2, y: this.rows - 2 };
        this.gameRunning = true;

        this.winScreen.classList.remove('show');
        this.winScreen.style.display = 'none';

        this.init();
        this.generateMazeWithCheck();
    }

    draw() {
        this.ctx.fillStyle = '#111';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const px = x * this.cellSize;
                const py = y * this.cellSize;

                if (this.maze[y][x] === 1) {
                    const wallGradient = this.ctx.createLinearGradient(px, py, px + this.cellSize, py + this.cellSize);
                    wallGradient.addColorStop(0, '#000000');
                    wallGradient.addColorStop(1, '#222222');
                    this.ctx.fillStyle = wallGradient;
                    this.ctx.fillRect(px, py, this.cellSize, this.cellSize);
                    this.ctx.strokeStyle = '#444';
                    this.ctx.strokeRect(px, py, this.cellSize, this.cellSize);
                } else {
                    const pathGradient = this.ctx.createLinearGradient(px, py, px + this.cellSize, py + this.cellSize);
                    pathGradient.addColorStop(0, '#4fcf4f'); // dịu mắt hơn
                    pathGradient.addColorStop(1, '#2fae2f');
                    this.ctx.fillStyle = pathGradient;
                    this.ctx.fillRect(px, py, this.cellSize, this.cellSize);
                }
            }
        }

        const px = this.player.x * this.cellSize + this.cellSize * 0.2;
        const py = this.player.y * this.cellSize + this.cellSize * 0.2;
        const psize = this.cellSize * this.player.size;

        const gradient = this.ctx.createRadialGradient(
                                                        px + psize / 2, py + psize / 2, 0, 
                                                        px + psize / 2, py + psize / 2, psize / 2
                                                    );
        gradient.addColorStop(0, '#4facfe');
        gradient.addColorStop(1, '#00f2fe');

        this.ctx.fillStyle = gradient;
        this.ctx.shadowColor = '#00f2fe';
        this.ctx.shadowBlur = 15;
        this.ctx.fillRect(px, py, psize, psize);
        this.ctx.shadowBlur = 0;

        const fx = this.finish.x * this.cellSize + this.cellSize * 0.1;
        const fy = this.finish.y * this.cellSize + this.cellSize * 0.1;
        const fsize = this.cellSize * 0.8;

        const finishGradient = this.ctx.createRadialGradient(
                                                                fx + fsize / 2, fy + fsize / 2, 0, 
                                                                fx + fsize / 2, fy + fsize / 2, fsize / 2
                                                            );
        finishGradient.addColorStop(0, '#ffd700');
        finishGradient.addColorStop(0.7, '#ffed4a');
        finishGradient.addColorStop(1, '#ff6b35');

        this.ctx.fillStyle = finishGradient;
        this.ctx.shadowColor = '#ffd700';
        this.ctx.shadowBlur = 20;
        this.ctx.fillRect(fx, fy, fsize, fsize);
        this.ctx.shadowBlur = 0;
    }

    gameLoop() {
        this.movePlayer();
        this.draw();
        requestAnimationFrame(() => this.gameLoop());
    }
}

window.addEventListener('load', () => {
    new MazeGame();
});