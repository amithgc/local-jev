#!/bin/sh
# Regenerate every PNG in this folder from its SVG source (2x scale, ~1600 px wide).
# Uses cairosvg in a throwaway uv environment; nothing is installed into the project venv.
# Requires: uv, and cairo (macOS: `brew install cairo`).
set -e
cd "$(dirname "$0")"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/lib}"
uv run --no-project --with cairosvg python - "$@" <<'PY'
import glob, sys, cairosvg
names = sys.argv[1:] or sorted(glob.glob("*.svg"))
for svg in names:
    png = svg[:-4] + ".png"
    cairosvg.svg2png(url=svg, write_to=png, scale=2)
    print("wrote", png)
PY
