"""
Play a real game of chess against a trained ChessGPT checkpoint.

The model was trained on transcripts shaped like:
    ;1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 ...
so we keep the game as that exact running string and hand it to engine.py, which scores
every legal move under the model and plays the most likely one.

Usage:
    python play.py --out_dir=out-chess-16layer                  # model plays black
    python play.py --out_dir=out-chess-16layer --human_color=b  # model plays white
    python play.py --hf_repo_id=fritz007x/chess-gpt-16layer-ckpt # pull ckpt.pt from HF Hub first

Enter moves in SAN, e.g. e4, Nf3, exd5, O-O, Qxe7+, e8=Q. Type "quit" to stop.
"""
import argparse
import os
import pickle
import sys

import torch

try:
    import chess
except ImportError:
    sys.exit("Missing dependency: run `pip install chess` (python-chess) first.")

import engine
from model import GPT, GPTConfig


def load_model(out_dir, device):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, checkpoint


def load_encoding(out_dir, checkpoint):
    meta_path = None
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        candidate = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        if os.path.exists(candidate):
            meta_path = candidate
    if meta_path is None:
        candidate = os.path.join("data", "lichess_hf_dataset", "meta.pkl")
        if os.path.exists(candidate):
            meta_path = candidate
    if meta_path is None:
        sys.exit("Could not find meta.pkl for the char-level chess vocab.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return encode, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-chess-16layer")
    parser.add_argument("--hf_repo_id", default="", help="if set, download ckpt.pt from this HF repo first")
    parser.add_argument("--human_color", default="w", choices=["w", "b"])
    parser.add_argument("--move_temperature", type=float, default=0.0,
                        help="0 plays the highest-scoring move (strongest); higher samples over whole moves")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.hf_repo_id:
        from huggingface_hub import hf_hub_download
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Downloading ckpt.pt from {args.hf_repo_id} ...")
        hf_hub_download(repo_id=args.hf_repo_id, filename="ckpt.pt",
                         local_dir=args.out_dir, local_dir_use_symlinks=False)

    print(f"Loading model from {args.out_dir} on {args.device} ...")
    model, checkpoint = load_model(args.out_dir, args.device)
    encode, decode = load_encoding(args.out_dir, checkpoint)

    board = chess.Board()
    game_str = ";"
    human_is_white = args.human_color == "w"

    print(board)
    print("You are", "White" if human_is_white else "Black", "- enter moves in SAN (e.g. Nf3, exd5, O-O). 'quit' to exit.\n")

    while not board.is_game_over():
        white_to_move = board.turn == chess.WHITE
        if white_to_move:
            game_str += f"{board.fullmove_number}."

        humans_turn = white_to_move == human_is_white

        if humans_turn:
            move = None
            while move is None:
                text = input(f"Your move ({'White' if white_to_move else 'Black'}): ").strip()
                if text.lower() in ("quit", "exit"):
                    return
                try:
                    move = board.parse_san(text)
                except ValueError:
                    print("Not a legal/parseable move, try again.")
            san = board.san(move)
        else:
            move, san, scored = engine.pick_move(model, encode, args.device, board, game_str,
                                                 move_temperature=args.move_temperature)
            others = ", ".join(f"{d['san']} {d['prob']:.2f}" for d in scored[1:4])
            print(f"Model move ({'White' if white_to_move else 'Black'}): {san} "
                  f"[p={next(d['prob'] for d in scored if d['move'] == move):.2f}; also considered {others}]")

        board.push(move)
        game_str += san + " "
        print(board)
        print()

    print("Game over:", board.result(), "-", board.outcome())


if __name__ == "__main__":
    main()
