# Assembles hf_space/app/ from the single source of truth one level up.
# Run this before creating/updating the Space so the Docker build context is
# self-contained. Re-run whenever gui.py / play.py / model.py / engine.py change.
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$root = Split-Path $here -Parent
$app  = Join-Path $here "app"

New-Item -ItemType Directory -Force -Path $app | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $app "data/lichess_hf_dataset") | Out-Null

Copy-Item (Join-Path $root "gui.py")    $app -Force
Copy-Item (Join-Path $root "play.py")   $app -Force
Copy-Item (Join-Path $root "model.py")  $app -Force
Copy-Item (Join-Path $root "engine.py") $app -Force
Copy-Item (Join-Path $root "data/lichess_hf_dataset/meta.pkl") `
          (Join-Path $app "data/lichess_hf_dataset/meta.pkl") -Force

Write-Output "Assembled app/ :"
Get-ChildItem -Recurse $app | ForEach-Object { "  " + $_.FullName.Substring($here.Length + 1) }
