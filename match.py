"""Measure ChessGPT's playing strength by playing matches.

Nothing about move selection can be improved without a way to tell whether a change
helped, and "it looked better in a few games" is not that. This runs repeatable matches
and reports the result with a confidence interval, so a change either clears the interval
or it does not ship.

Two things it is careful about:

  Determinism. At move_temperature=0 the engine is deterministic, and Stockfish at a fixed
  node limit is deterministic too, so a naive 100-game match is one game played 100 times.
  Every match is played from a fixed opening book (openings.txt), each line played twice
  with the colours reversed.

  Variance. Comparing two of our own variants head-to-head is far less noisy than measuring
  each against an external opponent and subtracting, so that is the primary mode; Stockfish
  is for calibration - putting an absolute number on the result - not for A/B.

Usage:
    # calibration: where does the current engine sit against Stockfish?
    python match.py --white policy --black sf:nodes=10

    # sweep node limits to bracket it
    python match.py --white policy --black sf:nodes=1 --openings 10
    python match.py --white policy --black sf:nodes=100 --openings 10

    # A/B: does a change help? (the crippled "second" player is the harness self-test)
    python match.py --white policy --black second

Player specs:
    policy              engine.pick_move, best-scoring move (the current engine)
    policy:temp=0.5     same, sampling over whole moves
    second              deliberately plays its 2nd-choice move - a known-weaker variant,
                        used to prove the harness can detect a real difference
    second:gap=1.0      gives up the best move only when the top two are within 1.0 nats,
                        i.e. a *mild* degradation - use it to find the smallest effect the
                        harness can actually resolve at a given number of games
    random              uniform random legal move - the floor
    sf:nodes=10         Stockfish at a fixed node limit
    sf:depth=1          Stockfish at a fixed depth
    sf:elo=1400         Stockfish with UCI_LimitStrength (use if nodes=1 is still too strong)
    sf:skill=0          Stockfish Skill Level - the only rung below UCI_Elo's 1320 floor,
                        and noisier than the others since it blunders on purpose
Options combine: "sf:elo=1320,nodes=100000".
"""
import argparse
import math
import os
import shutil
import sys
import time

try:
    import chess
    import chess.engine
    import chess.pgn
except ImportError:
    sys.exit("Missing dependency: run `pip install chess` (python-chess) first.")

import torch

import engine as chessgpt_engine
from play import load_model, load_encoding

# winget installs Stockfish here but does not always put it on PATH for every shell.
STOCKFISH_HINTS = [
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\stockfish.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
                       r"\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe"
                       r"\stockfish\stockfish-windows-x86-64-avx2.exe"),
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
]


def open_engine(path, retries=6):
    """Start a UCI engine, retrying transient startup failures.

    Launching several Stockfish processes at the same moment (parallel matches, each with
    an opponent and an adjudicator) intermittently dies with Windows exit code 0xC0000142,
    DLL initialisation failure. It is a startup race, not a real problem with the binary,
    and a short backoff clears it.
    """
    delay = 0.5
    for attempt in range(retries):
        try:
            return chess.engine.SimpleEngine.popen_uci(path)
        except chess.engine.EngineTerminatedError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def find_stockfish(explicit=""):
    """Locate the Stockfish binary, or return "" if there isn't one."""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"--stockfish path does not exist: {explicit}")
        return explicit
    found = shutil.which("stockfish")
    if found:
        return found
    for path in STOCKFISH_HINTS:
        if os.path.exists(path):
            return path
    return ""


# ---------------------------------------------------------------- players

class ModelPlayer:
    """The GPT. `rank` selects which scored candidate to play: 0 is the engine's actual
    behaviour, 1 is the deliberately-weakened variant used to self-test the harness."""

    def __init__(self, name, model, encode, device, move_temperature=0.0, rank=0, gap=None):
        self.name = name
        self.model, self.encode, self.device = model, encode, device
        self.move_temperature, self.rank = move_temperature, rank
        self.gap = gap

    def play(self, board, game_str):
        if self.rank == 0:
            move, _, _ = chessgpt_engine.pick_move(
                self.model, self.encode, self.device, board, game_str,
                move_temperature=self.move_temperature)
            return move
        scored = chessgpt_engine.score_legal_moves(
            self.model, self.encode, self.device, board, game_str)
        if len(scored) < 2:
            return scored[0]["move"]
        # `gap` turns an always-wrong variant into a mildly-wrong one: only give up the best
        # move when the model was nearly indifferent anyway. That is the size of change the
        # later phases will produce, so it is what the harness has to be able to resolve.
        if self.gap is not None and scored[0]["logprob"] - scored[1]["logprob"] >= self.gap:
            return scored[0]["move"]
        return scored[min(self.rank, len(scored) - 1)]["move"]

    def new_game(self, token):
        pass

    def close(self):
        pass


class RandomPlayer:
    def __init__(self, name, seed=0):
        import random
        self.name, self._rng = name, random.Random(seed)

    def play(self, board, game_str):
        return self._rng.choice(list(board.legal_moves))

    def new_game(self, token):
        pass

    def close(self):
        pass


class StockfishPlayer:
    def __init__(self, name, path, limit, elo=None, skill=None, threads=1, hash_mb=16):
        self.name = name
        self.engine = open_engine(path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        if elo is not None:
            # Stockfish refuses UCI_Elo below its own floor; clamp rather than crash.
            floor = self.engine.options["UCI_Elo"].min
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": max(elo, floor)})
        if skill is not None:
            # Last rung below the UCI_Elo floor. Skill Level works by deliberately picking
            # a worse move now and then, so it is noisier than a node limit - only use it
            # if UCI_Elo's minimum is still too strong to measure against.
            self.engine.configure({"Skill Level": skill})
        self.limit = limit
        self.token = None

    def play(self, board, game_str):
        # `game` must change between games or python-chess never sends "ucinewgame" and
        # Stockfish carries its transposition table across the whole match. A warm table
        # makes a node-limited opponent far stronger than its limit implies and makes the
        # result depend on the order games were played in.
        return self.engine.play(board, self.limit, game=self.token).move

    def new_game(self, token):
        self.token = token

    def close(self):
        self.engine.quit()


def make_player(spec, ctx):
    """Build a player from a spec string like "policy", "sf:nodes=10", "policy:temp=0.5"."""
    head, _, tail = spec.partition(":")
    opts = {}
    for part in filter(None, tail.split(",")):
        key, _, value = part.partition("=")
        opts[key.strip()] = value.strip()

    if head in ("policy", "second"):
        model, encode, device = ctx["model"](), ctx["encode"](), ctx["device"]
        return ModelPlayer(spec, model, encode, device,
                           move_temperature=float(opts.get("temp", 0.0)),
                           rank=1 if head == "second" else 0,
                           gap=float(opts["gap"]) if "gap" in opts else None)
    if head == "random":
        return RandomPlayer(spec, seed=int(opts.get("seed", 0)))
    if head == "sf":
        path = ctx["stockfish"]
        if not path:
            sys.exit("No Stockfish binary found. Install it (`winget install Stockfish`) "
                     "or pass --stockfish <path>.")
        if "nodes" in opts:
            limit = chess.engine.Limit(nodes=int(opts["nodes"]))
        elif "depth" in opts:
            limit = chess.engine.Limit(depth=int(opts["depth"]))
        elif "movetime" in opts:
            limit = chess.engine.Limit(time=float(opts["movetime"]))
        else:
            limit = chess.engine.Limit(nodes=10000)
        elo = int(opts["elo"]) if "elo" in opts else None
        skill = int(opts["skill"]) if "skill" in opts else None
        return StockfishPlayer(spec, path, limit, elo=elo, skill=skill)
    sys.exit(f"Unknown player spec: {spec!r}")


# ---------------------------------------------------------------- book

def load_openings(path, limit=0):
    """Read openings.txt into lists of SAN moves, validating each line is legal."""
    lines = []
    with open(path) as handle:
        for lineno, raw in enumerate(handle, 1):
            text = raw.split("#")[0].strip()
            if not text:
                continue
            board = chess.Board()
            sans = []
            for token in text.split():
                try:
                    move = board.parse_san(token)
                except ValueError:
                    sys.exit(f"{path}:{lineno}: illegal opening move {token!r} in {text!r}")
                sans.append(board.san(move))  # canonicalise before storing
                board.push(move)
            lines.append(sans)
    if not lines:
        sys.exit(f"{path} contains no opening lines")
    return lines[:limit] if limit else lines


# ---------------------------------------------------------------- one game

# Reasons a game ended, beyond what python-chess reports itself.
PLY_CAP = "ply-cap"
ADJUDICATED = "adjudicated"


def play_game(white, black, opening, max_plies, adjudicator, adj_cp, adj_plies):
    """Play one game. Returns (result_str, reason, plies, ms_by_player, pgn_game).

    The transcript is maintained in the model's training format - ";" to start, the full
    move number before each White move, "san " after every move - exactly as play.py does
    it, since that is the string the model is conditioned on.
    """
    board = chess.Board()
    game_str = ";"
    # Identity-only token: python-chess compares it against the previous game and issues
    # "ucinewgame" when it changes, so every UCI engine starts each game with a clean table.
    token = object()
    white.new_game(token)
    black.new_game(token)
    ms = {white.name: [], black.name: []}
    node = game = chess.pgn.Game()
    game.headers["White"], game.headers["Black"] = white.name, black.name
    game.headers["Event"] = "match.py"
    game.headers["Opening"] = " ".join(opening)

    reason = None
    streak = 0

    while not board.is_game_over(claim_draw=True):
        if board.ply() >= max_plies:
            reason = PLY_CAP
            break

        if board.turn == chess.WHITE:
            game_str += f"{board.fullmove_number}."

        ply_index = board.ply()
        if ply_index < len(opening):
            move = board.parse_san(opening[ply_index])
        else:
            player = white if board.turn == chess.WHITE else black
            start = time.perf_counter()
            move = player.play(board, game_str)
            ms[player.name].append((time.perf_counter() - start) * 1000)

        san = board.san(move)
        board.push(move)
        game_str += san + " "
        node = node.add_variation(move)

        # Adjudication: the engine will happily play out a dead-lost position for 80 more
        # plies, which costs real minutes and adds nothing. Stop once Stockfish says it is
        # decided and has said so for a few plies running (one spike shouldn't end a game).
        # flat threshold: tying it to the opening length would adjudicate 2-ply and 4-ply
        # book lines from different points for no reason.
        if adjudicator is not None and board.ply() > 20:
            info = adjudicator.analyse(board, chess.engine.Limit(depth=10), game=token)
            score = info["score"].white()
            decided = score.is_mate() or abs(score.score(mate_score=100000)) >= adj_cp
            streak = streak + 1 if decided else 0
            if streak >= adj_plies:
                winner = chess.WHITE if score.score(mate_score=100000) > 0 else chess.BLACK
                result = "1-0" if winner == chess.WHITE else "0-1"
                game.headers["Result"] = result
                return result, ADJUDICATED, board.ply(), ms, game

    if reason is None:
        outcome = board.outcome(claim_draw=True)
        result = outcome.result()
        reason = outcome.termination.name.lower().replace("_", "-")
    else:
        result = "1/2-1/2"

    game.headers["Result"] = result
    return result, reason, board.ply(), ms, game


# ---------------------------------------------------------------- stats

def elo_from_score(score):
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def fmt_elo(value):
    if value == float("inf"):
        return ">+800"
    if value == float("-inf"):
        return "<-800"
    return f"{value:+.0f}"


def pair_up(scores):
    """Collapse each opening's two colour-reversed games into one number.

    The two games of a pair share an opening and are played by the same two engines, so
    they are correlated - treating them as independent overstates the sample size. Pairing
    is the standard fix and it cuts the interval materially, which matters because the
    changes worth shipping later are much smaller than the crippled-variant self-test.
    """
    return [(scores[i] + scores[i + 1]) / 2 for i in range(0, len(scores) - 1, 2)]


# Two-tailed 95% t critical values by degrees of freedom. The paired view has only ~20 data
# points, where using the normal 1.96 understates the interval by about 7%.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
        30: 2.042}


def t_critical(n):
    df = n - 1
    if df <= 0:
        return 0.0
    if df in _T95:
        return _T95[df]
    for bound, value in ((40, 2.021), (60, 2.000), (120, 1.980)):
        if df <= bound:
            return value
    return 1.960


def summarise(scores):
    """Elo difference and a 95% interval from a list of per-datum scores.

    Feed it raw per-game scores (1.0 / 0.5 / 0.0) for the unpaired view, or the output of
    pair_up() for the paired one. The mean is the same either way; only the interval moves.
    """
    n = len(scores)
    score = sum(scores) / n
    if n > 1:
        var = sum((s - score) ** 2 for s in scores) / (n - 1)
    else:
        var = 0.0
    se = math.sqrt(var / n) if n else 0.0
    crit = t_critical(n)
    lo = elo_from_score(min(max(score - crit * se, 0.0), 1.0))
    hi = elo_from_score(min(max(score + crit * se, 0.0), 1.0))
    if se > 0:
        los = 0.5 * (1.0 + math.erf((score - 0.5) / (se * math.sqrt(2.0))))
    else:
        los = 1.0 if score > 0.5 else (0.0 if score < 0.5 else 0.5)
    return score, elo_from_score(score), lo, hi, los


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--white", default="policy", help="player spec, see module docstring")
    ap.add_argument("--black", default="sf:nodes=10", help="player spec")
    ap.add_argument("--openings", type=int, default=0, help="use only the first N book lines")
    ap.add_argument("--book", default="openings.txt")
    ap.add_argument("--out_dir", default="out-chess-16layer")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stockfish", default="", help="path to the Stockfish binary")
    ap.add_argument("--max_plies", type=int, default=240)
    ap.add_argument("--adjudicate", default="auto", choices=["auto", "on", "off"],
                    help="stop games Stockfish considers decided (needs a Stockfish binary)")
    ap.add_argument("--adj_cp", type=int, default=900)
    ap.add_argument("--adj_plies", type=int, default=4)
    ap.add_argument("--pgn", default="", help="write all games to this PGN file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    stockfish_path = find_stockfish(args.stockfish)

    # The model is loaded at most once and shared, even when both sides are model players.
    cache = {}

    def get_model():
        if "model" not in cache:
            print(f"Loading model from {args.out_dir} on {args.device} ...")
            model, ckpt = load_model(args.out_dir, args.device)
            encode, _ = load_encoding(args.out_dir, ckpt)
            cache["model"], cache["encode"] = model, encode
        return cache["model"]

    ctx = {"model": get_model, "encode": lambda: (get_model(), cache["encode"])[1],
           "device": args.device, "stockfish": stockfish_path}

    player_a = make_player(args.white, ctx)
    player_b = make_player(args.black, ctx)
    # Distinct names matter: the timing dict and PGN headers are keyed by them.
    if player_a.name == player_b.name:
        player_a.name += " (A)"
        player_b.name += " (B)"

    adjudicator = None
    if args.adjudicate != "off" and stockfish_path:
        adjudicator = open_engine(stockfish_path)
        adjudicator.configure({"Threads": 1, "Hash": 16})
    elif args.adjudicate == "on":
        sys.exit("--adjudicate=on needs a Stockfish binary; none found.")

    openings = load_openings(args.book, args.openings)
    name_a, name_b = player_a.name, player_b.name

    print(f"\n{name_a}  vs  {name_b}")
    print(f"{len(openings)} openings x 2 colours = {len(openings) * 2} games"
          f"{'' if adjudicator else '  (no adjudication)'}")
    if stockfish_path:
        print(f"stockfish: {stockfish_path}")
    print()

    scores = []       # per game, from A's perspective
    reasons = {}
    ms = {name_a: [], name_b: []}
    games = []
    t_start = time.perf_counter()

    try:
        for index, opening in enumerate(openings):
            for a_is_white in (True, False):
                white = player_a if a_is_white else player_b
                black = player_b if a_is_white else player_a
                result, reason, plies, game_ms, pgn_game = play_game(
                    white, black, opening, args.max_plies,
                    adjudicator, args.adj_cp, args.adj_plies)

                if result == "1-0":
                    a_score = 1.0 if a_is_white else 0.0
                elif result == "0-1":
                    a_score = 0.0 if a_is_white else 1.0
                else:
                    a_score = 0.5
                scores.append(a_score)
                reasons[reason] = reasons.get(reason, 0) + 1
                for key, values in game_ms.items():
                    ms[key].extend(values)
                games.append(pgn_game)

                running = sum(scores) / len(scores)
                print(f"  [{len(scores):3d}/{len(openings) * 2}] "
                      f"{' '.join(opening):<20} {name_a} as {'W' if a_is_white else 'B'}: "
                      f"{result:>7} {reason:<22} {plies:3d} plies   "
                      f"running score {running:.3f}")
    except KeyboardInterrupt:
        print("\ninterrupted - reporting on the games completed so far\n")
    finally:
        player_a.close()
        player_b.close()
        if adjudicator is not None:
            adjudicator.quit()

    if not scores:
        sys.exit("no games completed")

    wins = scores.count(1.0)
    draws = scores.count(0.5)
    losses = scores.count(0.0)
    score, elo, lo, hi, los = summarise(scores)

    print("\n" + "=" * 72)
    print(f"{name_a}  vs  {name_b}")
    print(f"  {wins}W  {draws}D  {losses}L  in {len(scores)} games")
    print(f"  score {score:.3f}   Elo {fmt_elo(elo)}  [{fmt_elo(lo)}, {fmt_elo(hi)}]"
          f"   LOS {los * 100:.1f}%   (unpaired)")
    pairs = pair_up(scores)
    if len(pairs) >= 2:
        _, pelo, plo, phi, plos = summarise(pairs)
        print(f"  paired ({len(pairs)} openings): Elo {fmt_elo(pelo)} "
              f"[{fmt_elo(plo)}, {fmt_elo(phi)}]   LOS {plos * 100:.1f}%   <- use this one")
    for name, values in ms.items():
        if values:
            print(f"  {name}: {sum(values) / len(values):.0f} ms/move over {len(values)} moves")
    print("  endings: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    print(f"  wall clock {time.perf_counter() - t_start:.0f}s")
    print("=" * 72)

    if args.pgn:
        with open(args.pgn, "w") as handle:
            for game in games:
                print(game, file=handle, end="\n\n")
        print(f"wrote {len(games)} games to {args.pgn}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
