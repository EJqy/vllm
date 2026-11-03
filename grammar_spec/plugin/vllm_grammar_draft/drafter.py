"""Grammar-aware draft completion.

A model drafter (EAGLE-3, n-gram, ...) proposes K tokens without looking at
the request's grammar. `GrammarDrafter` walks that proposal through the
grammar matcher and

* keeps every model token the grammar accepts;
* at the first model token the grammar rejects (or once the model proposal
  runs out) switches to the grammar: if the grammar fully determines the
  next characters (xgrammar's jump-forward string), their canonical tokens
  become the draft;
* stops at the first position where neither source has a valid token, and
  pads the rest of the K slots with a token the grammar rejects there.

The result is just a draft. The target model verifies it under the same
grammar bitmask, so the output distribution is unchanged: grammar tokens that
do not match the target's own tokenization are rejected like any other draft,
and pad tokens have zero target probability.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import Protocol


class Source(IntEnum):
    MODEL = 0
    GRAMMAR = 1
    PAD = 2


class Matcher(Protocol):
    """The subset of `xgrammar.GrammarMatcher` the drafter uses."""

    def accept_token(self, token_id: int) -> bool: ...

    def find_jump_forward_string(self) -> str: ...

    def is_terminated(self) -> bool: ...

    def rollback(self, num_tokens: int) -> None: ...


@dataclass(slots=True)
class Draft:
    tokens: list[int] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    # The model proposed a token the grammar rejects.
    model_rejected: bool = False

    def add(self, tokens: Sequence[int], source: Source) -> None:
        self.tokens.extend(tokens)
        self.sources.extend([source] * len(tokens))

    def count(self, source: Source) -> int:
        return sum(s == source for s in self.sources)


class GrammarDrafter:
    """Merge a model draft with grammar-determined tokens.

    Args:
        encode: Tokenizes a string without special tokens.
        pad_token_id: Vocabulary id used to pad drafts to K; it is checked
            against the grammar before each use. None disables padding.
        use_model: If False, ignore the model draft (grammar-only drafting).
        cache_size: Number of jump-forward strings whose tokenization is
            cached; schema keys and punctuation repeat across steps.
    """

    def __init__(
        self,
        encode: Callable[[str], list[int]],
        pad_token_id: int | None,
        use_model: bool = True,
        cache_size: int = 1 << 16,
    ):
        self.pad_token_id = pad_token_id
        self.use_model = use_model
        self._encode = lru_cache(maxsize=cache_size)(lambda s: tuple(encode(s)))

    def propose(self, matcher: Matcher, model_draft: Sequence[int], k: int) -> Draft:
        """Return a draft of `k` tokens (fewer only if padding is unsafe).

        `matcher` must be at the state after the last committed token; it is
        restored before returning.
        """
        draft = Draft()
        num_accepted = 0
        follow_model = self.use_model
        try:
            while len(draft.tokens) < k and not matcher.is_terminated():
                pos = len(draft.tokens)
                if follow_model and pos < len(model_draft):
                    token = model_draft[pos]
                    if matcher.accept_token(token):
                        num_accepted += 1
                        draft.add((token,), Source.MODEL)
                        continue
                    draft.model_rejected = True
                # From here on the model's tokens condition on a prefix the
                # draft no longer follows.
                follow_model = False
                run = self._grammar_run(matcher, k - pos)
                if not run:
                    break
                num_accepted += len(run)
                draft.add(run, Source.GRAMMAR)
            self._pad(matcher, draft, k)
        finally:
            if num_accepted:
                matcher.rollback(num_accepted)
        return draft

    def _grammar_run(self, matcher: Matcher, budget: int) -> list[int]:
        """Accept and return the canonical tokens of the jump-forward string."""
        text = matcher.find_jump_forward_string()
        if not text:
            return []
        run = []
        for token in self._encode(text)[:budget]:
            if not matcher.accept_token(token):
                break
            run.append(token)
        return run

    def _pad(self, matcher: Matcher, draft: Draft, k: int) -> None:
        """Pad to `k` with a token the grammar rejects here, so the masked
        target gives it zero probability. This keeps every decode step at
        1 + K tokens (ragged drafts fall off the uniform-decode CUDA graph).
        Skipped if the grammar would accept the pad token."""
        missing = k - len(draft.tokens)
        if missing <= 0 or self.pad_token_id is None:
            return
        if not matcher.is_terminated() and matcher.accept_token(self.pad_token_id):
            matcher.rollback(1)
            return
        draft.add((self.pad_token_id,) * missing, Source.PAD)
