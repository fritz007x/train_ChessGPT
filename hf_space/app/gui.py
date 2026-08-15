"""
Web GUI for playing chess against a trained ChessGPT checkpoint.

Loads the model once and serves a self-contained page (no external assets) with a
click-to-move board. Moves come from engine.py, which scores every legal move under the
model, so the only knob is how much randomness to allow between whole moves. Game state
lives server-side, keyed per visitor by an opaque session cookie, as the same running
transcript the model was trained on (";1.e4 e5 2.Nf3 ...") plus a python-chess board.

Usage:
    python gui.py --out_dir=out-chess-16layer            # then open http://127.0.0.1:8686
    python gui.py --device=cuda --port=9000
    python gui.py --host=0.0.0.0 --port=7860             # public bind, e.g. inside a container
"""
import argparse
import json
import os
import threading
import time
import uuid

import torch
import chess
import chess.svg
from flask import Flask, jsonify, request

import engine
from play import load_model, load_encoding

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Per-visitor game sessions. One model is shared by every session (that's the
# expensive resource); LOCK serializes both model inference and session-dict
# mutation, which is fine at "a handful of concurrent friends" scale.
# ---------------------------------------------------------------------------
LOCK = threading.Lock()
MODEL = None
ENCODE = None
DECODE = None
DEVICE = "cpu"

SESSIONS = {}  # sid -> session state dict
SESSION_TTL_SECONDS = 2 * 3600
MAX_SESSIONS = 200

DEFAULT_PARAMS = {
    "move_temperature": 0.0,  # 0 = always play the highest-scoring legal move
}


def fresh_session_state():
    return {
        "board": chess.Board(),
        "sans": [],            # SAN move list, source of truth for the transcript
        "human_color": "w",
        "last_model_info": None,
        "params": dict(DEFAULT_PARAMS),
        "last_seen": time.time(),
    }


def prune_sessions():
    now = time.time()
    stale = [sid for sid, s in SESSIONS.items() if now - s["last_seen"] > SESSION_TTL_SECONDS]
    for sid in stale:
        del SESSIONS[sid]
    if len(SESSIONS) > MAX_SESSIONS:
        oldest = sorted(SESSIONS.items(), key=lambda kv: kv[1]["last_seen"])
        for sid, _ in oldest[: len(SESSIONS) - MAX_SESSIONS]:
            del SESSIONS[sid]


def get_or_create_session(sid):
    """Must be called while holding LOCK."""
    if sid and sid in SESSIONS:
        SESSIONS[sid]["last_seen"] = time.time()
        return sid, SESSIONS[sid]
    new_sid = uuid.uuid4().hex
    SESSIONS[new_sid] = fresh_session_state()
    prune_sessions()
    return new_sid, SESSIONS[new_sid]


def transcript(sans):
    """Rebuild the training-format game string from the SAN list."""
    s = ";"
    for i, san in enumerate(sans):
        if i % 2 == 0:
            s += f"{i // 2 + 1}."
        s += san + " "
    return s


def push_san(state, san):
    state["board"].push_san(san)
    state["sans"].append(san)


def do_model_move(state):
    board = state["board"]
    prompt = transcript(state["sans"])
    if board.turn == chess.WHITE:
        prompt += f"{board.fullmove_number}."  # training format numbers white moves: ";1.e4 e5 2.Nf3"
    move, san, scored = engine.pick_move(MODEL, ENCODE, DEVICE, board, prompt,
                                         move_temperature=state["params"]["move_temperature"])
    info = {"san": san, "top": [{"san": d["san"], "prob": round(d["prob"], 4)} for d in scored[:5]]}
    state["last_model_info"] = info
    push_san(state, san)
    return info


def state_json(state, error=None):
    board = state["board"]
    outcome = board.outcome()
    return jsonify({
        "fen": board.fen(),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "human_color": state["human_color"],
        "sans": state["sans"],
        "game_str": transcript(state["sans"]),
        "legal_moves_uci": [m.uci() for m in board.legal_moves],
        "last_move_uci": board.peek().uci() if board.move_stack else None,
        "game_over": {
            "result": board.result(),
            "reason": outcome.termination.name.replace("_", " ").title() if outcome else "",
        } if outcome else None,
        "check": board.is_check(),
        "check_square": chess.square_name(board.king(board.turn)) if board.is_check() else None,
        "params": state["params"],
        "last_model_info": state["last_model_info"],
        "error": error,
    })


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
SESSION_COOKIE = "sid"


def respond(sid, state, error=None):
    resp = state_json(state, error=error)
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="Lax", max_age=SESSION_TTL_SECONDS)
    return resp


@app.route("/api/state")
def api_state():
    with LOCK:
        sid, state = get_or_create_session(request.cookies.get(SESSION_COOKIE))
        return respond(sid, state)


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data = request.get_json(force=True)
    with LOCK:
        sid, state = get_or_create_session(request.cookies.get(SESSION_COOKIE))
        state["board"] = chess.Board()
        state["sans"] = []
        state["last_model_info"] = None
        state["human_color"] = data.get("human_color", "w")
        if state["human_color"] == "b":
            do_model_move(state)
        return respond(sid, state)


@app.route("/api/params", methods=["POST"])
def api_params():
    data = request.get_json(force=True)
    with LOCK:
        sid, state = get_or_create_session(request.cookies.get(SESSION_COOKIE))
        params = state["params"]
        for k in DEFAULT_PARAMS:
            if k in data:
                params[k] = type(DEFAULT_PARAMS[k])(data[k])
        params["move_temperature"] = min(max(params["move_temperature"], 0.0), 2.0)
        return respond(sid, state)


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(force=True)
    with LOCK:
        sid, state = get_or_create_session(request.cookies.get(SESSION_COOKIE))
        board = state["board"]
        if board.outcome():
            return respond(sid, state, error="Game is over - start a new game.")
        move = None
        if "san" in data and data["san"].strip():
            try:
                move = board.parse_san(data["san"].strip())
            except ValueError:
                return respond(sid, state, error=f"Illegal or unparseable move: {data['san'].strip()}")
        elif "uci" in data:
            uci = data["uci"]
            try:
                move = board.parse_uci(uci)
            except ValueError:
                # pawn reaching last rank without promotion suffix: auto-queen
                try:
                    move = board.parse_uci(uci + "q")
                except ValueError:
                    return respond(sid, state, error="Illegal move.")
        if move is None:
            return respond(sid, state, error="No move given.")
        push_san(state, board.san(move))
        if not board.outcome():
            do_model_move(state)
        return respond(sid, state)


@app.route("/api/model_move", methods=["POST"])
def api_model_move():
    with LOCK:
        sid, state = get_or_create_session(request.cookies.get(SESSION_COOKIE))
        if state["board"].outcome():
            return respond(sid, state, error="Game is over.")
        do_model_move(state)
        return respond(sid, state)


@app.route("/api/undo", methods=["POST"])
def api_undo():
    with LOCK:
        sid, state = get_or_create_session(request.cookies.get(SESSION_COOKIE))
        board = state["board"]
        human_white = state["human_color"] == "w"
        while state["sans"]:
            board.pop()
            state["sans"].pop()
            if (board.turn == chess.WHITE) == human_white:
                break
        return respond(sid, state)


@app.route("/")
def index():
    return PAGE


# ---------------------------------------------------------------------------
# Front end (single self-contained page, no external assets)
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ChessGPT</title>
<style>
  :root {
    --bg: #0b0c10; --panel: #16181f; --panel2: #1f222b; --text: #f5f6f8;
    --dim: #9298a8; --accent: #ffb454;
    --light: #f0d9b5; --dark: #b58863;
    --light-hl: #cdd16a; --dark-hl: #aaa23b;
    --sel: rgba(255,170,0,.45);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 14px/1.45 "Segoe UI", system-ui, sans-serif;
         display: flex; justify-content: center; padding: 18px; gap: 20px; flex-wrap: wrap; }
  h1 { font-size: 17px; margin-bottom: 10px; font-weight: 600; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--dim); margin: 14px 0 6px; }

  #board-wrap { display: flex; flex-direction: column; align-items: center; }
  #status { margin: 10px 0 6px; min-height: 22px; font-size: 15px; }
  #status .thinking { color: var(--accent); }
  #status .gameover { color: #ff6b6b; font-weight: 600; }
  #board { display: grid; grid-template-columns: repeat(8, 64px); grid-template-rows: repeat(8, 64px);
           border: 4px solid var(--accent); border-radius: 4px; user-select: none; }
  .sq { width: 64px; height: 64px; display: flex; align-items: center; justify-content: center;
        cursor: pointer; position: relative; }
  .sq.light { background: var(--light); }  .sq.dark { background: var(--dark); }
  .sq.lastmove.light { background: var(--light-hl); }  .sq.lastmove.dark { background: var(--dark-hl); }
  .sq.check::before { content:""; position:absolute; inset:0; z-index: 1;
      background: radial-gradient(ellipse at center, #ff0000 0%, #e70000 50%, rgba(158,0,0,0) 100%); }
  .sq.selected::after { content:""; position:absolute; inset:0; background: var(--sel); z-index: 1; }
  .sq.target::before { content:""; position:absolute; width:20px; height:20px; border-radius:50%;
                       background: rgba(20,20,20,.35); z-index: 2; }
  .sq.target.capture::before { width:56px; height:56px; background:none;
                               border: 5px solid rgba(20,20,20,.35); }
  .sq .piece { position: relative; z-index: 3; width: 76%; height: 76%; pointer-events: none;
               filter: drop-shadow(0 1px 2px rgba(0,0,0,.35)); }
  .sq .piece svg { width: 100%; height: 100%; display: block; }
  .sq .coord { position:absolute; font-size:10px; font-weight:700; z-index:4; opacity:.85; }
  .sq .coord.file { right:3px; bottom:1px; } .sq .coord.rank { left:3px; top:1px; }
  .sq.light .coord { color: var(--dark); } .sq.dark .coord { color: var(--light); }

  #panel { width: 340px; background: var(--panel); border: 1px solid #2a2d38; border-radius: 8px;
           padding: 16px; align-self: flex-start; }
  button { background: var(--panel2); color: var(--text); border: 1px solid #3a3e4c; border-radius: 5px;
           padding: 7px 10px; cursor: pointer; font-size: 13px; }
  button:hover { border-color: var(--accent); color: var(--accent); }
  button.primary { background: var(--accent); color: #14161c; font-weight: 700; border-color: var(--accent); }
  button.primary:hover { color: #14161c; filter: brightness(1.1); }
  .btnrow { display: flex; gap: 8px; flex-wrap: wrap; }

  .param { margin-bottom: 10px; }
  .param label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 2px; }
  .param label .val { color: var(--accent); font-variant-numeric: tabular-nums; font-weight: 700; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .check { display: flex; align-items: center; gap: 8px; font-size: 13px; margin-bottom: 10px; }
  .hint { color: var(--dim); font-size: 12px; margin-top: 2px; }

  #saninput { display: flex; gap: 8px; margin-top: 6px; }
  #saninput input { flex: 1; background: var(--panel2); border: 1px solid #3a3e4c; border-radius: 5px;
                    color: var(--text); padding: 7px 9px; font-size: 13px; }
  #saninput input:focus { outline: none; border-color: var(--accent); }
  #err { color: #ff6b6b; font-size: 13px; min-height: 18px; margin-top: 6px; font-weight: 600; }

  #moves { max-height: 150px; overflow-y: auto; background: var(--panel2); border: 1px solid #2a2d38;
           border-radius: 5px; padding: 8px 10px; font-size: 13px;
           display: grid; grid-template-columns: 2.2em 1fr 1fr; gap: 2px 8px; }
  #moves .num { color: var(--dim); }
  #transcript { background: var(--panel2); border: 1px solid #2a2d38; border-radius: 5px; padding: 8px 10px;
                font: 12px/1.5 Consolas, monospace; word-break: break-all; max-height: 90px; overflow-y: auto;
                color: var(--dim); }
  #modelinfo { background: var(--panel2); border: 1px solid #2a2d38; border-radius: 5px; padding: 8px 10px;
               font-size: 12px; }
  #modelinfo .cand { font-family: Consolas, monospace; }
  #modelinfo .bad { color: #ff6b6b; } #modelinfo .ok { color: #4ade80; }
  #modelinfo .fb { color: var(--accent); font-weight: 700; }
</style>
</head>
<body>
  <div id="board-wrap">
    <h1>&#9812; ChessGPT &mdash; 16-layer / 50M &mdash; play the trained model</h1>
    <div id="board"></div>
    <div id="status"></div>
  </div>

  <div id="panel">
    <h2>Game</h2>
    <div class="btnrow">
      <button class="primary" onclick="newGame('w')">New game as White</button>
      <button class="primary" onclick="newGame('b')">New game as Black</button>
      <button onclick="post('/api/undo',{})">Undo</button>
      <button onclick="post('/api/model_move',{})">Force model move</button>
    </div>
    <div id="saninput">
      <input id="san" placeholder="or type SAN: e4, Nf3, O-O, e8=Q..." onkeydown="if(event.key==='Enter')sendSan()">
      <button onclick="sendSan()">Play</button>
    </div>
    <div class="hint">Click a piece then a destination. Pawn promotions auto-queen; use the SAN box to underpromote.</div>
    <div id="err"></div>

    <h2>Move selection</h2>
    <div class="param">
      <label>Move temperature <span class="val" id="v_move_temperature"></span></label>
      <input type="range" id="p_move_temperature" min="0" max="2.0" step="0.05" onchange="sendParams()" oninput="showVals()">
      <div class="hint">0 = always play the highest-scoring move (strongest). Higher samples
      between whole moves &mdash; every option is still legal.</div>
    </div>

    <h2>Last model move</h2>
    <div id="modelinfo">&mdash;</div>

    <h2>Moves</h2>
    <div id="moves"></div>

    <h2>Model transcript</h2>
    <div id="transcript">;</div>
  </div>

<script>
const PIECE_SVGS = __PIECE_SVGS__;
let S = null;          // last state from server
let selected = null;   // selected square name e.g. "e2"
let busy = false;

function sqName(file, rank) { return "abcdefgh"[file] + (rank + 1); }

function fenBoard(fen) {   // -> map square name -> piece char
  const map = {};
  const rows = fen.split(" ")[0].split("/");
  for (let r = 0; r < 8; r++) {
    let f = 0;
    for (const ch of rows[r]) {
      if (/\d/.test(ch)) f += +ch;
      else { map[sqName(f, 7 - r)] = ch; f++; }
    }
  }
  return map;
}

function render() {
  if (!S) return;
  const board = document.getElementById("board");
  board.innerHTML = "";
  const pieces = fenBoard(S.fen);
  const flip = S.human_color === "b";
  const last = S.last_move_uci;
  const targets = selected ? S.legal_moves_uci.filter(u => u.startsWith(selected)).map(u => u.slice(2, 4)) : [];

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = flip ? 7 - col : col;
      const rank = flip ? row : 7 - row;
      const name = sqName(file, rank);
      const d = document.createElement("div");
      d.className = "sq " + ((file + rank) % 2 ? "light" : "dark");
      if (last && (name === last.slice(0,2) || name === last.slice(2,4))) d.classList.add("lastmove");
      if (name === selected) d.classList.add("selected");
      if (S.check_square === name) d.classList.add("check");
      if (targets.includes(name)) { d.classList.add("target"); if (pieces[name]) d.classList.add("capture"); }
      if (pieces[name]) {
        d.innerHTML = `<span class="piece">${PIECE_SVGS[pieces[name]]}</span>`;
      }
      if (col === 7) d.innerHTML += `<span class="coord rank">${rank + 1}</span>`;
      if (row === 7) d.innerHTML += `<span class="coord file">${"abcdefgh"[file]}</span>`;
      d.onclick = () => clickSq(name);
      board.appendChild(d);
    }
  }

  // status
  const st = document.getElementById("status");
  if (busy) st.innerHTML = `<span class="thinking">Model is thinking&hellip;</span>`;
  else if (S.game_over) st.innerHTML = `<span class="gameover">Game over: ${S.game_over.result} (${S.game_over.reason})</span>`;
  else {
    const yours = S.turn === S.human_color;
    st.textContent = (S.turn === "w" ? "White" : "Black") + " to move" + (yours ? " - your turn" : "") + (S.check ? " - check!" : "");
  }

  // moves list
  const mv = document.getElementById("moves");
  mv.innerHTML = "";
  for (let i = 0; i < S.sans.length; i += 2) {
    mv.innerHTML += `<span class="num">${i/2+1}.</span><span>${S.sans[i]}</span><span>${S.sans[i+1] || ""}</span>`;
  }
  mv.scrollTop = mv.scrollHeight;

  document.getElementById("transcript").textContent = S.game_str;

  // model info: the legal moves the model rated highest, played one marked
  const mi = document.getElementById("modelinfo");
  const info = S.last_model_info;
  if (!info) mi.innerHTML = "&mdash;";
  else {
    mi.innerHTML = info.top.map(c =>
      `<div class="cand ${c.san === info.san ? "ok" : ""}">${c.san === info.san ? "&#10003;" : "&nbsp;&nbsp;"} ` +
      `${c.san} <span style="color:var(--dim)">${(100 * c.prob).toFixed(1)}%</span></div>`).join("");
  }

  // params
  for (const k of ["move_temperature"]) {
    document.getElementById("p_" + k).value = S.params[k];
  }
  showVals();
}

function showVals() {
  for (const k of ["move_temperature"]) {
    document.getElementById("v_" + k).textContent = document.getElementById("p_" + k).value;
  }
}

async function post(url, body, thinking) {
  if (busy) return;
  if (thinking) { busy = true; render(); }
  try {
    const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    S = await r.json();
    document.getElementById("err").textContent = S.error || "";
  } finally {
    busy = false; selected = null; render();
  }
}

function clickSq(name) {
  if (!S || S.game_over || busy) return;
  if (S.turn !== S.human_color) return;
  const mine = S.legal_moves_uci.some(u => u.startsWith(name));
  if (selected && S.legal_moves_uci.some(u => u.startsWith(selected + name))) {
    post("/api/move", {uci: selected + name}, true);
  } else if (mine && name !== selected) {
    selected = name; render();
  } else {
    selected = null; render();
  }
}

function sendSan() {
  const inp = document.getElementById("san");
  if (!inp.value.trim()) return;
  post("/api/move", {san: inp.value}, true);
  inp.value = "";
}

function newGame(color) { post("/api/new_game", {human_color: color}, color === "b"); }

function sendParams() {
  const body = {};
  for (const k of ["move_temperature"]) body[k] = +document.getElementById("p_" + k).value;
  post("/api/params", body);
}

fetch("/api/state").then(r => r.json()).then(s => { S = s; render(); });
</script>
</body>
</html>
"""

# Embed the real cburnett chess piece set bundled with python-chess (used by
# lichess, Wikipedia, and chess.svg) so the board matches a familiar look
# without any external asset or network fetch.
_PIECE_SVGS = {
    fen_char: f'<svg viewBox="0 0 45 45">{chess.svg.PIECES[fen_char]}</svg>'
    for fen_char in "PNBRQKpnbrqk"
}
PAGE = PAGE.replace("__PIECE_SVGS__", json.dumps(_PIECE_SVGS))


def main():
    global MODEL, ENCODE, DECODE, DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-chess-16layer")
    parser.add_argument("--hf_repo_id", default="", help="if set, download ckpt.pt from this HF repo first")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other hosts")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8686)))
    args = parser.parse_args()

    if args.hf_repo_id:
        from huggingface_hub import hf_hub_download
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Downloading ckpt.pt from {args.hf_repo_id} ...")
        hf_hub_download(repo_id=args.hf_repo_id, filename="ckpt.pt",
                         local_dir=args.out_dir, local_dir_use_symlinks=False)

    DEVICE = args.device
    print(f"Loading model from {args.out_dir} on {DEVICE} ...")
    MODEL, checkpoint = load_model(args.out_dir, DEVICE)
    ENCODE, DECODE = load_encoding(args.out_dir, checkpoint)
    print(f"Ready - listening on http://{args.host}:{args.port}")

    from waitress import serve
    serve(app, host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
