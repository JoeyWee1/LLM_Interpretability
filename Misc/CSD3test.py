#!/usr/bin/env python3
"""
Perfect Tic-Tac-Toe self-play (minimax) with full step-by-step printing.

WARNING: Printing every step for 1000 games will produce a *lot* of output.
If you want less, set GAMES smaller or set SHOW_STEPS=False.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import random

GAMES = 1000
SHOW_STEPS = True  # set False if output is too big
SEED = 0           # change/None for different randomness among equally-optimal moves

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

EMPTY = " "
X, O = "X", "O"


def render(board: tuple[str, ...]) -> str:
    def cell(i: int) -> str:
        return board[i] if board[i] != EMPTY else str(i + 1)

    rows = []
    for r in range(0, 9, 3):
        rows.append(f" {cell(r)} | {cell(r+1)} | {cell(r+2)} ")
    return "\n---+---+---\n".join(rows)


def winner(board: tuple[str, ...]) -> str | None:
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board: tuple[str, ...]) -> bool:
    return winner(board) is None and all(v != EMPTY for v in board)


def moves(board: tuple[str, ...]) -> list[int]:
    return [i for i, v in enumerate(board) if v == EMPTY]


def other(p: str) -> str:
    return O if p == X else X


@lru_cache(maxsize=None)
def minimax_value(board: tuple[str, ...], player_to_move: str) -> int:
    """
    Returns the game-theoretic value from the perspective of X:
      +1 : X can force a win
       0 : perfect play leads to draw
      -1 : O can force a win
    """
    w = winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    if is_draw(board):
        return 0

    # If it's X to move, choose max; if it's O to move, choose min.
    if player_to_move == X:
        best = -2
        for m in moves(board):
            nxt = list(board)
            nxt[m] = X
            best = max(best, minimax_value(tuple(nxt), O))
            if best == 1:
                break
        return best
    else:
        best = 2
        for m in moves(board):
            nxt = list(board)
            nxt[m] = O
            best = min(best, minimax_value(tuple(nxt), X))
            if best == -1:
                break
        return best


def best_moves(board: tuple[str, ...], player_to_move: str) -> list[int]:
    """All optimal moves for the current player (perfect play)."""
    vals = []
    for m in moves(board):
        nxt = list(board)
        nxt[m] = player_to_move
        v = minimax_value(tuple(nxt), other(player_to_move))
        vals.append((v, m))

    # Current player wants to maximize X's outcome if X, minimize if O.
    if player_to_move == X:
        target = max(v for v, _ in vals)
    else:
        target = min(v for v, _ in vals)
    return [m for v, m in vals if v == target]


@dataclass
class GameResult:
    winner: str | None
    moves: int


def play_one(game_index: int, rng: random.Random, show: bool) -> GameResult:
    board = tuple([EMPTY] * 9)
    player = X
    ply = 0

    if show:
        print(f"\n========== GAME {game_index+1} ==========")
        print("Start position:")
        print(render(board))

    while True:
        ply += 1
        opts = best_moves(board, player)
        m = rng.choice(opts)  # random among equally-optimal moves

        nxt = list(board)
        nxt[m] = player
        board = tuple(nxt)

        if show:
            print(f"\nMove {ply}: {player} plays {m+1} (optimal choices: {[i+1 for i in opts]})")
            print(render(board))

        w = winner(board)
        if w is not None:
            if show:
                print(f"\nResult: {w} wins in {ply} moves.")
            return GameResult(winner=w, moves=ply)
        if is_draw(board):
            if show:
                print(f"\nResult: draw in {ply} moves.")
            return GameResult(winner=None, moves=ply)

        player = other(player)


def main():
    rng = random.Random(SEED)

    x_wins = 0
    o_wins = 0
    draws = 0
    total_moves = 0

    for g in range(GAMES):
        res = play_one(g, rng, SHOW_STEPS)
        total_moves += res.moves
        if res.winner == X:
            x_wins += 1
        elif res.winner == O:
            o_wins += 1
        else:
            draws += 1

    print("\n========== SUMMARY ==========")
    print(f"Games: {GAMES}")
    print(f"X wins: {x_wins}")
    print(f"O wins: {o_wins}")
    print(f"Draws : {draws}")
    print(f"Average moves/game: {total_moves / GAMES:.2f}")
    print("Note: With perfect play, Tic-Tac-Toe is a draw; any wins would indicate a bug.")


if __name__ == "__main__":
    main()
