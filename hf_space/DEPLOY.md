# Deploying ChessGPT to a Hugging Face Space

This folder is everything needed for a free CPU **Docker Space**. The model
weights are pulled from the public repo `fritz007x/chess-gpt-16layer-ckpt` at
build time, so nothing large lives in the Space's git repo.

## One-time steps

1. **Assemble the app source** (copies `gui.py`, `play.py`, `model.py`, and the
   vocab into `app/` — the Docker build context):

   ```powershell
   ./assemble.ps1      # Windows
   ```
   ```bash
   ./assemble.sh       # macOS/Linux/Git Bash
   ```

2. **Create the Space** at https://huggingface.co/new-space
   - Owner: your account
   - Space name: e.g. `chess-gpt`
   - License: your choice
   - **SDK: Docker** → **Blank** template
   - Hardware: **CPU basic (free)**
   - Visibility: Public (or Private if you only want to share the link)

3. **Push these files** to the Space repo. From inside `hf_space/`:

   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-user>/chess-gpt
   git add Dockerfile requirements.txt README.md .dockerignore app/
   git commit -m "ChessGPT web GUI"
   git branch -M main
   git push -u origin main
   ```

   You'll be prompted for your HF username and a **write token**
   (https://huggingface.co/settings/tokens) as the password.

4. The Space builds automatically (a few minutes — it installs CPU torch and
   bakes in the checkpoint). When it finishes, your public URL is:
   `https://huggingface.co/spaces/<your-user>/chess-gpt`

## Updating later

After editing `gui.py` / `play.py` / `model.py`, re-run the assemble script and
push again:

```bash
./assemble.sh && git add -A && git commit -m "update" && git push
```

## What's verified vs. not

- **Verified locally**: the app logic (session isolation, move validation,
  sampling controls), the assembled `app/` layout, and running the exact
  container command (`--host=0.0.0.0`, `PORT` env) against the real checkpoint.
- **NOT verified**: the Docker image build itself and the HF Space runtime —
  these can't be tested on this Windows machine (no Docker; a Linux container
  build behaves differently). Expect to watch the first build.

## Troubleshooting

If the Space fails to build or start, open the **build/runtime logs** on the
Space page. Most likely suspects, in order:

1. **The torch install line** — if the CPU wheel version fails to resolve, bump
   `torch==2.12.1` in the `Dockerfile` to a current version, or drop the pin.
2. **`libgomp` / OpenMP errors on torch import** — the `Dockerfile` already
   installs `libgomp1`; if you changed the base image, keep that line or switch
   to the non-slim `python:3.11` base (which includes it).
3. **Checkpoint download fails** — confirm the `CKPT_REPO` repo is public and
   the filename is `ckpt.pt`.

## Notes

- **Files to commit**: `Dockerfile`, `requirements.txt`, `README.md`,
  `.dockerignore`, and the `app/` folder. Do **not** commit `assemble.*` output
  conflicts or `app/__pycache__`.
- **Free-tier sleep**: the Space sleeps after ~48h idle and cold-starts on the
  next visit. Because the checkpoint is baked into the image, cold start is just
  model load (a few seconds), not a re-download.
- **Concurrency**: one CPU model instance serves everyone; moves are serialized,
  so with several simultaneous players each waits their turn. Fine for friends;
  upgrade the Space hardware if you need more.
- **Different checkpoint**: change the `CKPT_REPO` build arg in the `Dockerfile`.
