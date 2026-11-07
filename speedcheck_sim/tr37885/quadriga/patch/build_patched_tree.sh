#!/usr/bin/env bash
# Build the patched QuaDRiGa tree used by the speed-check simulation.
# Usage: QUADRIGA_DIR=<QuaDRiGa git checkout> QUADRIGA_PATCHED_DIR=<output folder> bash build_patched_tree.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${QUADRIGA_DIR:?set QUADRIGA_DIR to a QuaDRiGa git checkout}"
DST="${QUADRIGA_PATCHED_DIR:?set QUADRIGA_PATCHED_DIR to the output folder}"
[[ "$(git -C "$SRC" rev-parse HEAD)" == "277866650eb115adb5b3e8ac252b0d1df073596d" ]] || { echo "unexpected QuaDRiGa commit"; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC/quadriga_src" "$DST/"
cp "$SRC/QuaDRiGa_License.txt" "$HERE/NOTICE.md" "$DST/"
patch --no-backup-if-mismatch "$DST/quadriga_src/@qd_builder/get_channels.m" "$HERE/get_channels.diff"
echo "patched tree: $DST"
sha256sum "$DST/quadriga_src/@qd_builder/get_channels.m"
