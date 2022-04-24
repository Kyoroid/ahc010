import sys
import random
import math
from typing import NamedTuple
import logging

input = sys.stdin.readline
logger = logging.getLogger()

TERM = -1
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
ROTATION = [
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (4, 5, 4, 5),
    (5, 4, 5, 4),
    (6, 7, 6, 7),
    (7, 6, 7, 6),
]


def read():
    N = 30
    T = []
    for i in range(N):
        row = list(map(int, list(input().strip())))
        T.append(row)
    return N, T


def rotate_cell(cell, dr: int):
    t = ROTATION[cell.t][dr % 4]
    return create_cell(t)


def diff_rotation(before, after) -> int:
    return ROTATION[before.t].index(after.t)


def create_cell(t: int):
    if t == 0:
        return Cell(north=WEST, west=NORTH, south=TERM, east=TERM, t=0)
    elif t == 1:
        return Cell(west=SOUTH, south=WEST, north=TERM, east=TERM, t=1)
    elif t == 2:
        return Cell(south=EAST, east=SOUTH, north=TERM, west=TERM, t=2)
    elif t == 3:
        return Cell(east=NORTH, north=EAST, south=TERM, west=TERM, t=3)
    elif t == 4:
        return Cell(north=WEST, west=NORTH, south=EAST, east=SOUTH, t=4)
    elif t == 5:
        return Cell(west=SOUTH, south=WEST, east=NORTH, north=EAST, t=5)
    elif t == 6:
        return Cell(east=WEST, west=EAST, north=TERM, south=TERM, t=6)
    elif t == 7:
        return Cell(north=SOUTH, south=NORTH, east=TERM, west=TERM, t=7)
    return Cell(TERM, TERM, TERM, TERM, -1)


class Cell(NamedTuple):
    north: int
    east: int
    south: int
    west: int
    t: int


def create_cells(N, T):
    cells = []
    for i in range(N):
        row_cells = []
        for j in range(N):
            row_cells.append(create_cell(T[i][j]))
        cells.append(row_cells)
    return cells


def copy_cells(N, cells):
    copied_cells = []
    for i in range(N):
        row_cells = []
        for j in range(N):
            row_cells.append(cells[i][j])
        copied_cells.append(row_cells)
    return copied_cells


def write_cells(N, before, after):
    outline = []
    for i in range(N):
        for j in range(N):
            a = before[i][j]
            b = after[i][j]
            r = diff_rotation(a, b)
            outline.append(f"{r}")
    print("".join(outline))


def calc_loop_line(
    N: int,
    cells,
    visited,
    si: int,
    sj: int,
    start_direction: int,
) -> int:
    u = cells[si][sj]
    i, j = si, sj
    direction = start_direction
    # 未訪問のルートだけを調べる。重複カウントになるので、訪問済みなら再計算しない。
    if visited[i][j][direction]:
        return 0
    # 閉路長を計算
    loop_length = 0
    is_closed_loop = False
    while direction != TERM:
        if visited[i][j][direction]:
            is_closed_loop = True
            break
        visited[i][j][direction] = True
        if direction == NORTH:
            ni, nj = i - 1, j
            if ni < 0:
                break
            visited[ni][nj][SOUTH] = True
            direction = cells[ni][nj][SOUTH]
        elif direction == SOUTH:
            ni, nj = i + 1, j
            if N <= ni:
                break
            visited[ni][nj][NORTH] = True
            direction = cells[ni][nj][NORTH]
        elif direction == EAST:
            ni, nj = i, j + 1
            if N <= nj:
                break
            visited[ni][nj][WEST] = True
            direction = cells[ni][nj][WEST]
        else:
            ni, nj = i, j - 1
            if nj < 0:
                break
            visited[ni][nj][EAST] = True
            direction = cells[ni][nj][EAST]

        loop_length += 1
        i, j = ni, nj
    return loop_length if is_closed_loop else 0


def list_loop_length(N: int, cells):
    visited = [[[False for k in range(4)] for j in range(N)] for i in range(N)]
    loop_lengths = []
    for si in range(N):
        for sj in range(N):
            for start_direction in range(4):
                ll = calc_loop_line(N, cells, visited, si, sj, start_direction)
                if ll > 0:
                    loop_lengths.append(ll)
    loop_lengths.sort(reverse=True)
    return loop_lengths


def get_score(ll):
    if len(ll) < 2:
        return 0
    return ll[0] * ll[1]


def get_random_point(N: int, n_iter: int, i: int):
    linear_0 = 0.75 * (1.0 - i / n_iter)
    linear_1 = 0.5 * (1.0 - i / n_iter) + 0.5
    x0 = max(0, int(linear_0 * N // 2))
    x1 = min(N, N - x0)
    x2 = max(0, int(linear_1 * N // 2))
    x3 = min(N, N - x2)
    ni = random.randint(x0, x1-1)
    nj = random.randint(x0, x1-1)
    for i in range(5):
        if not (x2 <= ni < x3 and x2 <= nj < x3):
            break
        ni = random.randint(x0, x1-1)
        nj = random.randint(x0, x1-1)
    return ni, nj


def simulated_annealing(N: int, cells, n_iter: int, T_start=300, T_end=10):
    base = copy_cells(N, cells)
    before = cells
    score_before = get_score(list_loop_length(N, before))
    best = cells
    score_best = score_before
    for i in range(n_iter):
        Ti = pow(T_start, 1.0 - i / n_iter) * pow(T_end, i / n_iter)
        after = copy_cells(N, before)
        # next state
        for _ in range(20):
            ni, nj = get_random_point(N, n_iter, i)
            dr = random.randint(0, 3)
            cell = after[ni][nj]
            after[ni][nj] = rotate_cell(cell, dr)
        # eval
        score_after = get_score(list_loop_length(N, after))
        score_delta = score_after - score_before
        update_proba = 1.0 if score_delta >= 0 else pow(math.e, score_delta / Ti)
        if random.random() < update_proba:
            logger.debug(f"[{i}/{n_iter}]Update local: {score_before} -> {score_after}")
            before = after
            score_before = score_after
            if score_after > score_best:
                logger.info(f"[{i}/{n_iter}]Update global: {score_best} -> {score_after}")
                best = after
                score_best = score_after
                # write_cells(N, base, best)
    write_cells(N, base, best)


def climbing(N: int, cells, n_iter: int):
    base = copy_cells(N, cells)
    before = cells
    score_before = get_score(list_loop_length(N, before))
    best = copy_cells(N, cells)
    score_best = score_before
    for i in range(n_iter):
        after = copy_cells(N, before)
        # next state
        for _ in range(20):
            ni, nj = get_random_point(N, n_iter, i)
            dr = random.randint(0, 3)
            cell = after[ni][nj]
            after[ni][nj] = rotate_cell(cell, dr)
        # eval
        score_after = get_score(list_loop_length(N, after))
        score_delta = score_after - score_before
        if score_delta >= 0:
            logger.debug(f"[{i}/{n_iter}]Update local: {score_before} -> {score_after}")
            before = after
            score_before = score_after
            if score_after > score_best:
                logger.info(f"[{i}/{n_iter}]Update global: {score_best} -> {score_after}")
                best = after
                score_best = score_after
                write_cells(N, base, best)
    write_cells(N, base, best)


def solve(N: int, T):
    before = create_cells(N, T)
    climbing(N, before, n_iter=3000)


if __name__ == "__main__":
    random.seed(2022)
    logging.basicConfig(level=logging.INFO)
    inputs = read()
    outputs = solve(*inputs)
    if outputs is not None:
        print("%s" % str(outputs))
