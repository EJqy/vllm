"""Cumulative per-source draft counters, flushed to a JSON file."""

import json
import os
import threading
from dataclasses import asdict, dataclass, field

from .drafter import Draft, Source


def _zeros() -> dict[str, int]:
    return {s.name.lower(): 0 for s in Source}


@dataclass
class DraftStats:
    num_drafts: int = 0
    num_skipped: int = 0  # structured requests handed back to vLLM's own trim
    num_model_rejected: int = 0  # drafts where the grammar rejected a model token
    drafted: dict[str, int] = field(default_factory=_zeros)
    accepted: dict[str, int] = field(default_factory=_zeros)
    propose_seconds: float = 0.0

    def record_draft(self, draft: Draft, seconds: float) -> None:
        self.num_drafts += 1
        self.num_model_rejected += draft.model_rejected
        self.propose_seconds += seconds
        for s in draft.sources:
            self.drafted[s.name.lower()] += 1

    def record_verified(self, draft: Draft, output_token_ids: list[int]) -> None:
        """Count the accepted draft prefix: the target's output matches the
        draft up to the first rejected position."""
        for token, source, out in zip(draft.tokens, draft.sources, output_token_ids):
            if token != out:
                break
            self.accepted[source.name.lower()] += 1


class StatsWriter:
    """Holds the counters and, if a path is set, rewrites the file from a
    daemon thread so the last steps of a run are flushed too."""

    def __init__(self, path: str | None, interval_s: float = 0.5):
        self.stats = DraftStats()
        if path is not None:
            thread = threading.Thread(
                target=self._flush_loop, args=(path, interval_s), daemon=True,
                name="grammar-draft-stats",
            )
            thread.start()

    def _flush_loop(self, path: str, interval_s: float) -> None:
        stop = threading.Event()
        while not stop.wait(interval_s):
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(asdict(self.stats), f)
            os.replace(tmp, path)
