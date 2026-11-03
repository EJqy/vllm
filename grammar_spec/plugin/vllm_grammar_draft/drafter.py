"""Merge a model drafter's proposal with the tokens the grammar determines.

Model tokens are kept while the grammar accepts them. At the first rejected
token, or when the model draft runs out, the canonical tokens of xgrammar's
jump-forward string take over. The rest is padded to K with a token the
grammar rejects. Verification is unchanged, so the output distribution is too.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache


class Source(IntEnum):
    MODEL = 0
    GRAMMAR = 1
    PAD = 2


@dataclass(slots=True)
class Draft:
    tokens: list[int] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    model_rejected: bool = False

    def add(self, tokens: Sequence[int], source: Source) -> None:
        self.tokens.extend(tokens)
        self.sources.extend([source] * len(tokens))


class GrammarDrafter:
    def __init__(
        self,
        encode: Callable[[str], list[int]],
        pad_token_id: int | None,
        use_model: bool = True,
    ):
        self.pad_token_id = pad_token_id
        self.use_model = use_model
        self._encode = lru_cache(maxsize=1 << 16)(lambda s: tuple(encode(s)))

    def propose(self, matcher, model_draft: Sequence[int], k: int) -> Draft:
        """Draft k tokens from the matcher's state, which is then restored."""
        draft = Draft()
        accepted = 0
        follow_model = self.use_model
        try:
            while len(draft.tokens) < k and not matcher.is_terminated():
                pos = len(draft.tokens)
                if follow_model and pos < len(model_draft):
                    if matcher.accept_token(model_draft[pos]):
                        accepted += 1
                        draft.add(model_draft[pos : pos + 1], Source.MODEL)
                        continue
                    draft.model_rejected = True
                # Later model tokens condition on a prefix we no longer follow.
                follow_model = False
                run = self._grammar_run(matcher, k - pos)
                if not run:
                    break
                accepted += len(run)
                draft.add(run, Source.GRAMMAR)
            self._pad(matcher, draft, k)
        finally:
            if accepted:
                matcher.rollback(accepted)
        return draft

    def _grammar_run(self, matcher, budget: int) -> list[int]:
        run = []
        for token in self._encode(matcher.find_jump_forward_string())[:budget]:
            if not matcher.accept_token(token):
                break
            run.append(token)
        return run

    def _pad(self, matcher, draft: Draft, k: int) -> None:
        """Keep every decode step at 1 + K tokens (ragged drafts fall off the
        uniform-decode CUDA graph). Skipped if the grammar accepts the pad."""
        missing = k - len(draft.tokens)
        if missing <= 0 or self.pad_token_id is None:
            return
        if not matcher.is_terminated() and matcher.accept_token(self.pad_token_id):
            matcher.rollback(1)
            return
        draft.add([self.pad_token_id] * missing, Source.PAD)
