"""Scheduler-side hook: rewrite each structured-output request's draft.

In synchronous scheduling, the engine hands the drafter's proposal to
`Scheduler.update_draft_token_ids`, which calls
`StructuredOutputManager.trim_spec_tokens` to cut the grammar-invalid suffix.
This hook replaces that trim with `GrammarDrafter.propose` for requests whose
grammar is an xgrammar matcher, and leaves every other request to vLLM.
It also observes `accept_tokens` to count accepted tokens per draft source.
"""

import time
import weakref
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
from vllm.v1.structured_output.utils import strip_speculative_padding

from .config import GrammarDraftConfig
from .drafter import Draft, GrammarDrafter
from .stats import StatsWriter

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.grammar_draft")  # under vLLM's log config


class GrammarDraftHook:
    def __init__(self, manager: StructuredOutputManager, cfg: GrammarDraftConfig):
        vllm_config = manager.vllm_config
        _check_compatible(vllm_config)
        tokenizer = manager.tokenizer
        pad = _pick_pad_token(tokenizer) if cfg.pad else None
        self.manager = manager
        self.k = vllm_config.num_speculative_tokens
        self.drafter = GrammarDrafter(
            encode=lambda s: tokenizer.encode(s, add_special_tokens=False),
            pad_token_id=pad,
            use_model=cfg.use_model,
        )
        self.writer = StatsWriter(cfg.stats_path)
        self._drafts: weakref.WeakKeyDictionary[Request, Draft] = (
            weakref.WeakKeyDictionary()
        )
        logger.info(
            "Grammar draft enabled: mode=%s, K=%d, pad_token_id=%s",
            "hybrid" if cfg.use_model else "grammar_only", self.k, pad,
        )

    def propose(self, request: "Request", spec_tokens: list[int]) -> list[int] | None:
        """The rewritten draft, or None to fall back to vLLM's trim."""
        grammar = self._eligible_grammar(request, spec_tokens)
        if grammar is None:
            self.writer.stats.num_skipped += request.use_structured_output
            return None
        t0 = time.perf_counter()
        draft = self.drafter.propose(
            grammar.matcher, strip_speculative_padding(spec_tokens), self.k
        )
        self.writer.stats.record_draft(draft, time.perf_counter() - t0)
        self._drafts[request] = draft
        return draft.tokens

    def observe_accepted(self, request: "Request", new_token_ids: list[int]) -> None:
        draft = self._drafts.pop(request, None)
        if draft is not None:
            self.writer.stats.record_verified(draft, new_token_ids)

    def _eligible_grammar(
        self, request: "Request", spec_tokens: list[int]
    ) -> XgrammarGrammar | None:
        if not request.use_structured_output:
            return None
        grammar = request.structured_output_request.grammar
        if not isinstance(grammar, XgrammarGrammar) or grammar.is_terminated():
            return None
        # Only once the whole draft window is grammar-constrained (i.e. not
        # inside a reasoning section).
        bounds = self.manager._get_constraint_bounds(
            request, strip_speculative_padding(spec_tokens)
        )
        if bounds.grammar_start or bounds.constraint_start:
            return None
        return grammar


def _check_compatible(vllm_config: "VllmConfig") -> None:
    spec = vllm_config.speculative_config
    problems = []
    if vllm_config.scheduler_config.async_scheduling:
        problems.append(
            "it rewrites drafts in the scheduler, which only reach the forward "
            "pass with synchronous scheduling; pass --no-async-scheduling"
        )
    if spec.draft_sample_method != "greedy":
        problems.append(
            "lossless verification needs one-hot draft probabilities; use "
            "draft_sample_method='greedy'"
        )
    if spec.rejection_sample_method != "standard":
        problems.append("it requires rejection_sample_method='standard'")
    if problems:
        raise ValueError("GRAMMAR_DRAFT=1 is incompatible with this config: "
                         + "; ".join(problems))


def _pick_pad_token(tokenizer) -> int | None:
    """The token for a raw NUL byte: JSON forbids unescaped control characters
    anywhere, and unlike special tokens, xgrammar rejects it without logging.
    The drafter still checks each pad against the grammar before using it."""
    ids = tokenizer.encode("\x00", add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def install(cfg: GrammarDraftConfig) -> None:
    orig_init = StructuredOutputManager.__init__
    orig_trim = StructuredOutputManager.trim_spec_tokens
    orig_accept = StructuredOutputManager.accept_tokens

    def __init__(self, vllm_config):
        orig_init(self, vllm_config)
        self._grammar_draft = (
            GrammarDraftHook(self, cfg)
            if vllm_config.speculative_config is not None
            and getattr(self, "tokenizer", None) is not None
            else None
        )

    def trim_spec_tokens(self, request, spec_tokens):
        hook = self._grammar_draft
        draft = hook.propose(request, spec_tokens) if hook else None
        return orig_trim(self, request, spec_tokens) if draft is None else draft

    def accept_tokens(self, request, new_token_ids):
        if self._grammar_draft is not None:
            self._grammar_draft.observe_accepted(request, new_token_ids)
        return orig_accept(self, request, new_token_ids)

    StructuredOutputManager.__init__ = __init__
    StructuredOutputManager.trim_spec_tokens = trim_spec_tokens
    StructuredOutputManager.accept_tokens = accept_tokens
