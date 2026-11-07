#!/usr/bin/env bash
# Creates or refreshes the three Python 3.12 environments of the timing scripts at the repository root (idempotent):
#   .venv           requirements/core.lock      (OpenFHE, BLS12-381; risk path, baselines B1-B3, throughput)
#   .venv-speed     requirements/speed.lock     (BLS12-381, numpy, scipy, matplotlib; speed path)
#   .venv-paillier  requirements/paillier.lock  (python-phe + gmpy2; Paillier baseline B4)
# Then runs the unit tests once (correctness before timing) and the machine check.
# Usage: bash experiments/timings/setup_envs.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
REQ="$ROOT/requirements"
say() { printf '\n== %s\n' "$*"; }
command -v python3.12 >/dev/null || { echo "python3.12 is required (the openfhe wheel is built for CPython 3.12)"; exit 1; }

mkenv() {           # mkenv DIR LOCK: create the environment and make its package set equal to the lock file
  local dir=$1 lock=$2 extra
  [ -x "$dir/bin/python" ] || python3.12 -m venv "$dir"
  "$dir/bin/pip" install -q --upgrade pip >/dev/null
  "$dir/bin/pip" install -q -r "$lock"
  extra=$(comm -23 <("$dir/bin/pip" freeze | cut -d= -f1 | tr 'A-Z_' 'a-z-' | sort -u) \
                   <(grep -v '^#' "$lock" | sed '/^$/d' | cut -d= -f1 | tr 'A-Z_' 'a-z-' | sort -u))
  if [ -n "$extra" ]; then "$dir/bin/pip" uninstall -q -y $extra; fi
}

say "Implementation environment (.venv, requirements/core.lock)"
mkenv "$ROOT/.venv" "$REQ/core.lock"
say "Speed-path environment (.venv-speed, requirements/speed.lock)"
mkenv "$ROOT/.venv-speed" "$REQ/speed.lock"
say "Paillier environment (.venv-paillier, requirements/paillier.lock)"
mkenv "$ROOT/.venv-paillier" "$REQ/paillier.lock"

say "Unit tests (correctness before timing)"
( cd "$ROOT" && .venv/bin/python -m pytest -q tests 2>&1 | tail -3 )

say "Machine check"
bash "$HERE/check_machine.sh" || true
