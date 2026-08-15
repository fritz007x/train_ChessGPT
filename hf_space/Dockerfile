# ChessGPT web GUI - Hugging Face Docker Space
# Builds a CPU-only image that bakes the trained checkpoint into an image layer
# so cold starts are fast (no 600 MB download when the Space wakes from sleep).
FROM python:3.11-slim

# HF Spaces run the container as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

WORKDIR /app

# torch needs libgomp (OpenMP) at import; python:3.11-slim doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# --- Python deps (torch from the CPU wheel index, rest from PyPI) ---
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- Application source (assembled by assemble.ps1 / assemble.sh) ---
COPY --chown=user app/ /app/

# --- Bake the public checkpoint into the image at build time ---
# Override with --build-arg CKPT_REPO=you/your-repo if you fork the model.
ARG CKPT_REPO=fritz007x/chess-gpt-16layer-ckpt
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='${CKPT_REPO}', filename='ckpt.pt', \
local_dir='out-chess-16layer', local_dir_use_symlinks=False)"

USER user
EXPOSE 7860

# Bind to all interfaces so the Space's proxy can reach us; port from $PORT.
CMD ["python", "gui.py", "--out_dir=out-chess-16layer", "--device=cpu", "--host=0.0.0.0"]
