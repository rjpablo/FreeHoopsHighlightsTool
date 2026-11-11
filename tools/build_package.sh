#!/bin/bash
# Rebuild dist/FreeHoopsHighlightsTool and the zips from src/ (run on the box).
set -e; B=$(cd "$(dirname "$0")/.." && pwd); D=$B/dist/FreeHoopsHighlightsTool
rm -rf "$B/dist"; mkdir -p "$D/runtime" "$D/models" "$D/tools"
cd "$B/src"; find app -name __pycache__ -prune -exec rm -rf {} +
cp -r app assets install.bat run.bat requirements.txt README.txt NOTICE.txt UPDATE.txt HANDOFF.md "$D/"
R=/workspace/freehoops_promo_tool/dist/FreeHoopsPromoTool_v13/runtime
cp "$R/ffmpeg.exe" "$R/ffprobe.exe" "$D/runtime/"; cp -r "$R/fonts" "$R/licenses" "$D/runtime/"
cp "$B/runtime_manifest.json" "$D/runtime/manifest.json"
cp /workspace/yolov8n.pt /workspace/yolov8s.pt "$D/models/" 2>/dev/null || true
cp "$B/tools/make_music.py" "$B/tools/build_package.sh" "$D/tools/"
