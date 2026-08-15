#!/usr/bin/env bash
# Assembles hf_space/app/ from the single source of truth one level up.
# Run this before creating/updating the Space so the Docker build context is
# self-contained. Re-run whenever gui.py / play.py / model.py / engine.py change.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
app="$here/app"

mkdir -p "$app/data/lichess_hf_dataset"
cp "$root/gui.py"    "$app/"
cp "$root/play.py"   "$app/"
cp "$root/model.py"  "$app/"
cp "$root/engine.py" "$app/"
cp "$root/data/lichess_hf_dataset/meta.pkl" "$app/data/lichess_hf_dataset/meta.pkl"

echo "Assembled app/ :"
find "$app" -type f | sed "s|$here/|  |"
