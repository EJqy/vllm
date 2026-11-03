"""Plugin switches, read from the environment of every vLLM process.

GRAMMAR_DRAFT          "1" enables the plugin (default: off; nothing is
                       patched when off).
GRAMMAR_DRAFT_MODE     "hybrid" (default): model draft + grammar fallback.
                       "grammar_only": ignore the model draft.
GRAMMAR_DRAFT_PAD      "1" (default) pads every draft to K tokens.
GRAMMAR_DRAFT_STATS    Path of a JSON file the scheduler rewrites with
                       cumulative counters (at most once per second).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GrammarDraftConfig:
    enabled: bool = False
    use_model: bool = True
    pad: bool = True
    stats_path: str | None = None

    @classmethod
    def from_env(cls) -> "GrammarDraftConfig":
        mode = os.environ.get("GRAMMAR_DRAFT_MODE", "hybrid")
        if mode not in ("hybrid", "grammar_only"):
            raise ValueError(f"GRAMMAR_DRAFT_MODE must be hybrid or grammar_only, "
                             f"got {mode!r}")
        return cls(
            enabled=os.environ.get("GRAMMAR_DRAFT", "0") == "1",
            use_model=mode == "hybrid",
            pad=os.environ.get("GRAMMAR_DRAFT_PAD", "1") == "1",
            stats_path=os.environ.get("GRAMMAR_DRAFT_STATS") or None,
        )
