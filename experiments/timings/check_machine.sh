#!/usr/bin/env bash
# Checks the machine and the environments before a timing run. Changes nothing; prints OK / PROBLEM / NOTE lines and
# one STATE line (exit 0 only if every check passes). If TIMING_REQUIRED_CPU is set, the CPU model must contain it.
# Usage: bash experiments/timings/check_machine.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ok=1
r() { if eval "$2" >/dev/null 2>&1; then echo "OK      $1"; else echo "PROBLEM $1${3:+  -> $3}"; ok=0; fi; }

CPU="$(sed -n 's/^model name[[:space:]]*: *//p' /proc/cpuinfo | head -1)"
echo "NOTE    CPU: $CPU"
if [ -n "${TIMING_REQUIRED_CPU:-}" ]; then
  r "CPU is the required model ($TIMING_REQUIRED_CPU)" "grep -q \"\$TIMING_REQUIRED_CPU\" <<< \"\$CPU\""
fi
GOVS="$(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
echo "NOTE    governor(s): ${GOVS:-unknown}"
r "CPU governor 'performance' on every core" "[ \"\$GOVS\" = 'performance ' ]" \
  "sudo cpupower frequency-set -g performance  (or: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor); lost at every reboot"
if command -v powerprofilesctl >/dev/null 2>&1; then
  PP="$(powerprofilesctl get 2>/dev/null)"; echo "NOTE    power profile: $PP"
  r "power profile 'performance'" "[ \"\$PP\" = performance ]" "powerprofilesctl set performance (in addition to the governor)"
fi
AC=""
for d in /sys/class/power_supply/*; do
  [ "$(cat "$d/type" 2>/dev/null)" = Mains ] && AC="${AC}$(cat "$d/online" 2>/dev/null)"
done
if [ -n "$AC" ]; then r "on AC power" "grep -q 1 <<< '$AC'" "plug in the charger"
else echo "NOTE    AC power not detectable"; fi
LOAD="$(cut -d' ' -f1 /proc/loadavg)"
r "machine idle (1-min load $LOAD < 0.5)" "awk 'BEGIN{exit !($LOAD < 0.5)}'" "close other programs and wait a minute"
r "free RAM >= 6 GB" "[ \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) -ge 6 ]" "close programs"
r "free disk >= 2 GB" "[ \$(df -BG --output=avail '$ROOT' | tail -1 | tr -dc 0-9) -ge 2 ]"

same() {            # same VENV LOCK: the environment's package set equals the lock file
  diff <("$1/bin/pip" freeze 2>/dev/null | grep -vi '^pip==' | tr 'A-Z_' 'a-z-' | sort) \
       <(grep -v '^#' "$2" | sed '/^$/d' | tr 'A-Z_' 'a-z-' | sort)
}
r ".venv = requirements/core.lock" "same '$ROOT/.venv' '$ROOT/requirements/core.lock'" "bash experiments/timings/setup_envs.sh"
r "  openfhe imports" "'$ROOT/.venv/bin/python' -c 'import openfhe'"
r ".venv-speed = requirements/speed.lock" "same '$ROOT/.venv-speed' '$ROOT/requirements/speed.lock'" "bash experiments/timings/setup_envs.sh"
r ".venv-paillier = requirements/paillier.lock" "same '$ROOT/.venv-paillier' '$ROOT/requirements/paillier.lock'" "bash experiments/timings/setup_envs.sh"
[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ] && echo "NOTE    turbo: $([ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = 0 ] && echo on || echo off) (recorded, not changed)"
if [ $ok = 1 ]; then echo "STATE: CHECK PASSED"; exit 0; else echo "STATE: CHECK FAILED"; exit 1; fi
