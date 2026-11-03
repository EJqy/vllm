"""GrammarDrafter against real xgrammar matchers built the way vLLM builds
them (Qwen3 tokenizer, special tokens filtered, compact JSON, rollback = K)."""

import json

import pytest
import xgrammar as xgr
from transformers import AutoTokenizer

from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm_grammar_draft.drafter import GrammarDrafter, Source

MODEL = "Qwen/Qwen3-32B"
VOCAB_SIZE = 151936
STOP = [151643, 151645]
PAD = 188  # raw NUL byte
K = 6
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "employment_status": {"enum": ["employed", "unemployed", "student"]},
    },
    "required": ["name", "age", "employment_status"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def compiled(tok):
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=VOCAB_SIZE)
    info = XgrammarBackend._filter_special_tokens(tok, info)
    return xgr.GrammarCompiler(info).compile_json_schema(
        json.dumps(SCHEMA), any_whitespace=False, separators=(",", ":")
    )


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def matcher_at(compiled, tok, prefix):
    m = xgr.GrammarMatcher(compiled, override_stop_tokens=STOP)
    for t in encode(tok, prefix):
        assert m.accept_token(t)
    return m


def drafter(tok, **kw):
    return GrammarDrafter(lambda s: encode(tok, s), pad_token_id=PAD, **kw)


def bitmask(m):
    bm = xgr.allocate_token_bitmask(1, VOCAB_SIZE)
    m.fill_next_token_bitmask(bm)
    return bm


def assert_valid(compiled, tok, prefix, draft):
    """Non-pad tokens are accepted in order; the first pad is rejected."""
    m = matcher_at(compiled, tok, prefix)
    for t, s in zip(draft.tokens, draft.sources):
        if s == Source.PAD:
            assert not m.accept_token(t)
            return
        assert m.accept_token(t)


def test_matcher_state_is_restored(compiled, tok):
    m = matcher_at(compiled, tok, '{"name":"Bob"')
    before = bitmask(m)
    drafter(tok).propose(m, encode(tok, ',"age":3'), K)
    assert (bitmask(m) == before).all() and not m.is_terminated()


def test_valid_model_draft_is_kept(compiled, tok):
    prefix, model = '{"name":"Bob"', encode(tok, ',"age":41,"')
    m = matcher_at(compiled, tok, prefix)
    d = drafter(tok).propose(m, model, K)
    assert d.tokens[: len(model)] == model[:K]
    assert d.sources[: len(model)] == [Source.MODEL] * min(len(model), K)
    assert not d.model_rejected
    assert_valid(compiled, tok, prefix, d)


def test_grammar_replaces_rejected_model_token(compiled, tok):
    # "age" is required next, so the model's "}" is invalid; the grammar
    # knows the next characters are ',"age":'.
    prefix = '{"name":"Bob"'
    m = matcher_at(compiled, tok, prefix)
    d = drafter(tok).propose(m, encode(tok, '}'), K)
    run = encode(tok, ',"age":')
    assert d.model_rejected
    assert d.tokens[: len(run)] == run
    assert d.sources[: len(run)] == [Source.GRAMMAR] * len(run)
    assert d.sources[len(run):] == [Source.PAD] * (K - len(run))
    assert_valid(compiled, tok, prefix, d)


def test_model_tokens_after_a_rejection_are_dropped(compiled, tok):
    prefix = '{"name":"Bob"'
    m = matcher_at(compiled, tok, prefix)
    # Invalid first token, then tokens that would be valid after ',"age":'.
    model = encode(tok, '}') + encode(tok, '41')
    d = drafter(tok).propose(m, model, K)
    assert Source.MODEL not in d.sources
    assert_valid(compiled, tok, prefix, d)


def test_grammar_extends_a_short_model_draft(compiled, tok):
    prefix = '{"name":"Bob","age":4'
    m = matcher_at(compiled, tok, prefix)
    model = encode(tok, '1')
    d = drafter(tok).propose(m, model, K)
    # After "41" the grammar still cannot tell whether more digits follow.
    assert d.tokens[:1] == model and d.sources[1:] == [Source.PAD] * (K - 1)
    m2 = matcher_at(compiled, tok, prefix + '1,')
    d2 = drafter(tok).propose(m2, [], K)
    run = encode(tok, '"employment_status":"')
    assert d2.tokens[: min(len(run), K)] == run[:K]
    assert_valid(compiled, tok, prefix + '1,', d2)


def test_grammar_only_ignores_the_model(compiled, tok):
    prefix = '{"name":"Bob"'
    m = matcher_at(compiled, tok, prefix)
    d = drafter(tok, use_model=False).propose(m, encode(tok, ',"age":41'), K)
    assert Source.MODEL not in d.sources
    assert d.tokens[:3] == encode(tok, ',"age":')[:3]


def test_budget_caps_the_grammar_run(compiled, tok):
    m = matcher_at(compiled, tok, '{"name":"Bob","age":41,')
    d = drafter(tok).propose(m, [], 2)
    assert d.tokens == encode(tok, '"employment_status":"')[:2]
    assert d.sources == [Source.GRAMMAR] * 2


def test_pad_is_skipped_where_the_grammar_accepts_it(tok):
    info = XgrammarBackend._filter_special_tokens(
        tok, xgr.TokenizerInfo.from_huggingface(tok, vocab_size=VOCAB_SIZE))
    m = xgr.GrammarMatcher(xgr.GrammarCompiler(info).compile_regex(r"[\s\S]*"))
    d = drafter(tok).propose(m, [], K)
    assert d.tokens == [] and m.accept_token(PAD)


def test_no_pad_token_means_no_padding(compiled, tok):
    m = matcher_at(compiled, tok, '{"name":"')
    d = GrammarDrafter(lambda s: encode(tok, s), pad_token_id=None).propose(
        m, [], K)
    assert d.tokens == []


def test_finished_json_is_not_extended(compiled, tok):
    prefix = '{"name":"Bob","age":41,"employment_status":"student"}'
    m = matcher_at(compiled, tok, prefix)
    d = drafter(tok).propose(m, [], K)
    assert Source.GRAMMAR not in d.sources
    assert_valid(compiled, tok, prefix, d)
