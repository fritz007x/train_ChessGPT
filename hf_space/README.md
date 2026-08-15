---
title: ChessGPT
emoji: ♟️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ChessGPT — play a chess language model

A 16-layer / 50M-parameter GPT trained on Lichess games (character-level PGN,
following [Karvonen 2024](https://arxiv.org/abs/2403.15498)). Play it in your
browser: click-to-move board, live sampling controls (temperature, top-k,
top-p), and a panel showing the raw move text the model generated for each ply.

Each visitor gets an independent game, isolated by a session cookie. The model
runs on CPU, so expect a second or two per move.

## How this Space is built

- `Dockerfile` — CPU-only image; downloads the checkpoint at **build time** from
  the public model repo so cold starts don't re-download it.
- `app/` — the application source (`gui.py`, `play.py`, `model.py`) plus the
  character vocab (`data/lichess_hf_dataset/meta.pkl`).
- The model weights live in a separate public repo
  (`fritz007x/chess-gpt-16layer-ckpt`) and are pulled in during the build.

To point at a different checkpoint, edit the `CKPT_REPO` build arg in the
`Dockerfile`.
