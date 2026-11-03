#!/usr/bin/env bash
# Run the REPORT.md appendix script verbatim (only TIK patched), for both
# schema forms, and save its output under results/offline/.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="${GS_DATA:-$HOME/.cache/grammar_spec}"
DATA="$ROOT/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
TIK="$ROOT/qwen_tiktoken/qwen.tiktoken"
WORK="$(mktemp -d)"
OUT="$REPO/grammar_spec/results/offline/appendix_verbatim_tiktoken.txt"
sed -n 301,417p "$REPO/grammar_spec/REPORT.md" > "$WORK/full.py"
sed -i "s|^TIK = .*|TIK = \"$TIK\"|" "$WORK/full.py"
sed '67s|.*|        out.append((params, inst))|' "$WORK/full.py" > "$WORK/args.py"
{
  echo "# REPORT.md lines 301-417, TIK=$TIK, sha256(qwen.tiktoken)=$(sha256sum "$TIK" | cut -d' ' -f1)"
  echo "# data: $DATA @ $(git -C "$ROOT/gorilla" rev-parse HEAD)"
  echo "## schema = name + arguments"
  "$REPO/.venv/bin/python" -I "$WORK/full.py" "$DATA"
  echo "## schema = arguments only (line 67 replaced)"
  "$REPO/.venv/bin/python" -I "$WORK/args.py" "$DATA"
} | tee "$OUT"
