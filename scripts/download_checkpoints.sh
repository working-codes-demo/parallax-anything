#!/usr/bin/env bash
# Downloads the Depth Anything V2 Small checkpoint (99 MB, Apache-2.0) into ./checkpoints
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints
URL=https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
SHA=715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378
OUT=checkpoints/depth_anything_v2_vits.pth
[ -f "$OUT" ] || curl -L --fail -o "$OUT" "$URL"
echo "$SHA  $OUT" | shasum -a 256 -c -
