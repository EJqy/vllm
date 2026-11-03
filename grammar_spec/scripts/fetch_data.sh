#!/usr/bin/env bash
# Fetch external data outside the repo (default ~/.cache/grammar_spec):
#   * BFCL v4 data at a pinned gorilla commit
#   * qwen.tiktoken (the vocab used by the REPORT.md appendix script)
# Prints the bfcl_eval/data directory to pass as --bfcl-root / data_root.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GORILLA_COMMIT=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
ROOT="${GS_DATA:-$HOME/.cache/grammar_spec}"
DEST="$ROOT/gorilla"
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout --sparse \
    https://github.com/ShishirPatil/gorilla.git "$DEST" >&2
fi
git -C "$DEST" sparse-checkout set berkeley-function-call-leaderboard/bfcl_eval/data >&2
git -C "$DEST" checkout -q "$GORILLA_COMMIT"
"$REPO/.venv/bin/python" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen-7B', 'qwen.tiktoken', local_dir='$ROOT/qwen_tiktoken')
" >&2
echo "$DEST/berkeley-function-call-leaderboard/bfcl_eval/data"
