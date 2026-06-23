#!/usr/bin/env bash
# Neural Microkernel — Tier A reproduction (CPU, no model required).
# Auto-discovers and runs every kernel/test_*.py CPU suite; prints a PASS/FAIL summary.
# Usage:  bash verify.sh        (set PYTHON=... to choose an interpreter)
set -u
cd "$(dirname "$0")/scripts" || exit 2
export PYTHONPATH="$PWD"
PY="${PYTHON:-python3}"
SUITES=$(cd kernel && ls test_*.py 2>/dev/null | sed 's/\.py$//' | sort)
pass=0; fail=0; failed=""
echo "Neural Microkernel — Tier A CPU suites ($PY)"
echo "-------------------------------------------------------------"
for t in $SUITES; do
  if out=$("$PY" "kernel/$t.py" 2>&1); then
    line=$(printf '%s\n' "$out" | grep -E "GREEN|OK|passed" | tail -1)
    printf "  PASS  %-24s %s\n" "$t" "${line:-ok}"
    pass=$((pass+1))
  else
    printf "  FAIL  %-24s\n" "$t"
    printf '%s\n' "$out" | tail -6 | sed 's/^/        /'
    fail=$((fail+1)); failed="$failed $t"
  fi
done
echo "-------------------------------------------------------------"
echo "RESULT: $pass/$((pass+fail)) suites passed.${failed:+  FAILED:$failed}"
[ "$fail" -eq 0 ] && echo "Tier A reproduction OK." || echo "Tier A reproduction had failures (see above)."
exit "$fail"
