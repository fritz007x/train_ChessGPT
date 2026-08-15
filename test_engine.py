"""Correctness checks for engine.py's move scoring.

The packed scorer builds a hand-written attention mask, which is easy to get subtly
wrong, so every check compares it against the naive scorer (one full row per candidate)
and against permutations of the candidate order - cross-candidate attention leakage
shows up as scores that change when candidates are reordered.

Usage:  python test_engine.py [--out_dir=out-chess-16layer] [--device=cpu]
"""
import argparse
import random
import time

import chess
import torch

import engine
from play import load_model, load_encoding

TOL = 1e-4


def transcript(sans):
    s = ";"
    for i, san in enumerate(sans):
        if i % 2 == 0:
            s += f"{i // 2 + 1}."
        s += san + " "
    return s


def position(moves):
    """Play a SAN line and return (board, prompt) in the model's training format."""
    board = chess.Board()
    sans = []
    for mv in moves.split():
        san = board.san(board.parse_san(mv))  # canonicalise, e.g. add check suffixes
        board.push_san(mv)
        sans.append(san)
    prompt = transcript(sans)
    if board.turn == chess.WHITE:
        prompt += f"{board.fullmove_number}."
    return board, prompt


CASES = [
    ("start position", *position("")),
    ("open middlegame", *position(
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 "
        "Nbd2 Bb7 Bc2 Re8 Nf1 Bf8 Ng3 g6 a4 c5 d5 c4 Bg5 h6 Be3 Nc5 Qd2 h5")),
    ("castling available", *position("e4 e5 Nf3 Nc6 Bc4 Bc5 d3 d6 Nc3 Nf6")),
    ("mate in 1 (Qxf7#)", *position("e4 e5 Bc4 Nc6 Qh5 Nf6")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out-chess-16layer")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, ckpt = load_model(args.out_dir, args.device)
    encode, _ = load_encoding(args.out_dir, ckpt)
    dev = args.device
    failures = []

    # promotion: reached by FEN rather than a long game. The prefix is off-distribution,
    # which is fine - this case exists to exercise the longest SAN ("e8=Q+") and the
    # padding path, not to check the model's chess taste.
    promo = chess.Board("8/4P1k1/8/8/8/8/6K1/8 w - - 0 60")
    cases = CASES + [("promotion (longest SAN)", promo, ";1.e4 e5 2.Nf3 Nc6 3.")]

    for name, board, prompt in cases:
        n = board.legal_moves.count()
        packed = engine.score_legal_moves(model, encode, dev, board, prompt, method="packed")
        naive = engine.score_legal_moves(model, encode, dev, board, prompt, method="naive", chunk=16)

        by_san_naive = {d["san"]: d["logprob"] for d in naive}
        worst = max(abs(d["logprob"] - by_san_naive[d["san"]]) for d in packed)
        ok = worst < TOL
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {n} moves, packed vs naive max diff {worst:.2e}")
        if not ok:
            failures.append(name)

        # permutation invariance: reorder the candidates fed to the packed scorer and the
        # per-move scores must be identical. This is the check that isolates leakage.
        _, sans, prefix_ids, suffix_ids, _ = engine._prepare(board, prompt, encode, model.config.block_size)
        order = list(range(len(suffix_ids)))
        random.Random(0).shuffle(order)
        straight = engine._score_packed(model, prefix_ids, suffix_ids, dev)
        shuffled = engine._score_packed(model, prefix_ids, [suffix_ids[i] for i in order], dev)
        pworst = max(abs(shuffled[k] - straight[i]) for k, i in enumerate(order))
        pok = pworst < TOL
        print(f"[{'PASS' if pok else 'FAIL'}] {name}: permutation invariance, max diff {pworst:.2e}")
        if not pok:
            failures.append(name + " (permutation)")

        best = packed[0]
        runners = ", ".join("{} {:.3f}".format(d["san"], d["prob"]) for d in packed[1:4])
        print("        best move: {}  p={:.3f}   runners-up: {}".format(best["san"], best["prob"], runners))

    # Mating moves must be scored under the "+" spelling the model actually uses. Without
    # this, python-chess's "#" costs a mate ~10-16 nats and the engine will not deliver
    # checkmate. (Whether the model then *chooses* the mate is a question about the model,
    # not about this code - it misses both mates below, which is what search would fix.)
    for line in ("e4 e5 Bc4 Nc6 Qh5 Nf6", "f3 e5 g4"):
        board, prompt = position(line)
        scored = engine.score_legal_moves(model, encode, dev, board, prompt)
        mate = next(d for d in scored if d["san"].endswith("#"))
        hash_only = engine._score_packed(model, encode(prompt), [encode(mate["san"] + " ")], dev)[0]
        gain = mate["logprob"] - hash_only
        ok = gain > 5.0
        print(f"[{'PASS' if ok else 'FAIL'}] mate spelling {mate['san']}: "
              f"{hash_only:.2f} -> {mate['logprob']:.2f} (+{gain:.2f} nats)")
        if not ok:
            failures.append(f"mate spelling ({line})")

    # whatever it picks must be legal, in every case
    for name, board, prompt in cases:
        move, san, _ = engine.pick_move(model, encode, dev, board, prompt)
        if move not in board.legal_moves:
            print(f"[FAIL] illegal move returned for {name}: {san}")
            failures.append(f"legality ({name})")
    print(f"[PASS] every returned move is legal" if not any("legality" in f for f in failures) else "")

    # timing at a late-game prefix, where CPU cost actually bites
    board, prompt = CASES[1][1], CASES[1][2]
    for label, kw in (("packed", dict(method="packed")), ("naive", dict(method="naive", chunk=32))):
        t0 = time.perf_counter()
        engine.score_legal_moves(model, encode, dev, board, prompt, **kw)
        print(f"        timing {label}: {time.perf_counter() - t0:.2f}s "
              f"({board.legal_moves.count()} moves, {len(prompt)}-char prefix)")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
