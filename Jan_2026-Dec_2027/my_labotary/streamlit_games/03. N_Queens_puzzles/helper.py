import string

def pos_to_label(pos: tuple[int, int]) -> str:
    """
        (row=0, col=0) -> a1
        (row=3, col=7) -> d8
    """
    row, col = pos
    return f"{string.ascii_lowercase[row]}{col + 1}"

def is_attacking(pos_1: tuple[int, int], pos_2: tuple[int, int]) -> bool:
    """
        2 queens bất kỳ xung đột nhau nếu:
        - cùng hàng
        - cùng cột
        - cùng đường chéo
    """
    r1, c1 = pos_1
    r2, c2 = pos_2

    same_row = r1 == r2
    same_col = c1 == c2
    same_diagonal = abs(r1 - r2) == abs(c1 - c2)

    return same_row or same_col or same_diagonal

def get_conflict_queens(queens: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Trả về tập các queens đang xung đột."""
    conflicts = set()
    queen_list = list(queens)

    for i in range(len(queen_list)):
        for j in range(i + 1, len(queen_list)):
            q1 = queen_list[i]
            q2 = queen_list[j]

            if is_attacking(q1, q2):
                conflicts.add(q1)
                conflicts.add(q2)

    return conflicts

def get_attacked_cells(queens: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    """
        Trả về các ô trống bị ít nhất một queen tấn công.
        Không dùng cho logic thắng/thua, chỉ để render gợi ý trực quan.
    """
    attacked = set()

    for row in range(n):
        for col in range(n):
            current_pos = (row, col)

            if current_pos in queens:
                continue

            for queen_pos in queens:
                if is_attacking(current_pos, queen_pos):
                    attacked.add(current_pos)
                    break

    return attacked

def get_valid_moves(queens: set[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Các ô trống có thể đặt queen mà không xung đột với queens hiện tại."""
    valid_moves = []

    for row in range(n):
        for col in range(n):
            pos = (row, col)

            if pos in queens:
                continue

            if not any(is_attacking(pos, queen) for queen in queens):
                valid_moves.append(pos)

    return valid_moves


def find_one_solution(n: int) -> list[tuple[int, int]] | None:
    """
        Backtracking để tìm một nghiệm hợp lệ.
        Mỗi lần thử đặt một queen trên một hàng.
    """
    solution = []

    def backtrack(row: int) -> bool:
        if row == n:
            return True

        for col in range(n):
            candidate = (row, col)

            if all(
                not is_attacking(candidate, placed_queen)
                for placed_queen in solution
            ):
                solution.append(candidate)

                if backtrack(row + 1):
                    return True

                solution.pop()

        return False

    return solution.copy() if backtrack(0) else None