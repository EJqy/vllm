"""Stage-0 checks whose outputs back the numbers in PROGRESS.md.

Subcommands (each writes a JSON file under grammar_spec/results/stage0/):
  resolve-config      runner / async_scheduling vLLM resolves per spec method
                      (CPU only; uses the same args as server.py)
  xgrammar-probe      xgrammar 0.2.8 API presence + compact-whitespace probe
  tokenizer-identity  qwen.tiktoken vs HF Qwen3-32B tokenization on BFCL texts
  server-checks       against a running server: hand-written schema requests
                      and unconstrained chat acceptance length
  greedy-compare      compare saved greedy outputs across configs of a
                      run_matrix.py result directory
"""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT_DIR = HERE.parent / "results" / "stage0"
MODEL = "Qwen/Qwen3-32B"


class EmptyProposer:
    """custom_class proposer that never drafts (only for resolve-config)."""

    def __init__(self, vllm_config):
        pass

    def propose(self, sampled_token_ids, *args, **kwargs):
        return [[] for _ in sampled_token_ids]


def save(name, obj):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path, n = OUT_DIR / name, 1
    while path.exists():  # results/ is append-only
        n += 1
        path = OUT_DIR / f"{Path(name).stem}.{n}{Path(name).suffix}"
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(json.dumps(obj, indent=2, ensure_ascii=False)[:4000])
    print(f"-> {path}")


def resolve_config(_args):
    from vllm.engine.arg_utils import EngineArgs

    from server import spec_config

    cases = {m: spec_config(m, 5) for m in ("none", "eagle3", "ngram", "suffix")}
    cases["custom_class"] = {"method": "custom_class",
                             "model": "stage0_checks.EmptyProposer",
                             "num_speculative_tokens": 5}
    out = {}
    for name, sc in cases.items():
        for sync in (False, True):
            ea = EngineArgs(
                model=MODEL, quantization="fp8_per_tensor", max_model_len=8192,
                speculative_config=sc,
                async_scheduling=False if sync else None,
                structured_outputs_config={"backend": "xgrammar",
                                           "disable_any_whitespace": True})
            cfg = ea.create_engine_config()
            out[f"{name}{'_no_async_flag' if sync else ''}"] = {
                "use_v2_model_runner": cfg.use_v2_model_runner,
                "async_scheduling": cfg.scheduler_config.async_scheduling,
                "max_concurrent_batches": cfg.max_concurrent_batches,
            }
    save("config_resolution.json", out)


def xgrammar_probe(_args):
    import importlib.metadata as md

    import xgrammar as xgr
    from transformers import AutoTokenizer

    res = {"xgrammar": md.version("xgrammar"), "api": {}, "compact_ws": []}
    for a in ["fill_next_token_bitmask", "find_jump_forward_string",
              "accept_token", "fork", "is_terminated", "rollback",
              "traverse_draft_tree"]:
        res["api"][f"GrammarMatcher.{a}"] = hasattr(xgr.GrammarMatcher, a)
    res["api"]["apply_token_bitmask_inplace"] = hasattr(
        xgr, "apply_token_bitmask_inplace")
    tok = AutoTokenizer.from_pretrained(MODEL)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=151936)
    comp = xgr.GrammarCompiler(info)
    two = {"a": {"type": "integer"}, "b": {"type": "integer"}}
    cases = [
        ({"type": "object", "properties": two, "required": ["a", "b"]},
         '{"a":1, "b":2}'),
        ({"type": "object", "properties": two}, '{"a":1, "b":2}'),
        ({"type": "array", "items": {"type": "integer"}}, "[1, 2]"),
        ({"type": "object", "properties": {"x": {"type": "array", "items": {
            "type": "string"}}}, "required": ["x"]}, '{"x":["a", "b"]}'),
        ({"type": "object", "properties": {"o": {"type": "object",
          "properties": two, "required": ["a", "b"]}}, "required": ["o"]},
         '{"o":{"a":1, "b":2}}'),
        ({"type": "object", "properties": {"a": {"anyOf": [{"type": "integer"},
          {"type": "string"}]}, "b": {"type": "integer"}},
          "required": ["a", "b"]}, '{"a":1, "b":2}'),
        ({"type": "object"}, '{"a":1, "b":2}'),
        ({"type": "object", "properties": two, "required": ["a", "b"]},
         '{"a": 1,"b":2}'),
    ]
    for schema, text in cases:
        g = comp.compile_json_schema(json.dumps(schema), any_whitespace=False,
                                     separators=(",", ":"))
        m = xgr.GrammarMatcher(g)
        res["compact_ws"].append({"schema": schema, "text": text,
                                  "accepted_in_compact_mode": m.accept_string(text)})
    save("xgrammar_probe.json", res)


def tokenizer_identity(args):
    import tiktoken
    from transformers import AutoTokenizer

    from offline_upper_bound import SPLITS, build

    pat = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
           r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
    ranks = {}
    for line in open(args.tiktoken, "rb"):
        if line.strip():
            t, r = line.split()
            ranks[base64.b64decode(t)] = int(r)
    enc = tiktoken.Encoding("qwen", pat_str=pat, mergeable_ranks=ranks,
                            special_tokens={})
    hf = AutoTokenizer.from_pretrained(MODEL)
    same = diff = 0
    for form in ("full", "args"):
        for split in SPLITS:
            for _, obj in build(args.data_root, split, form):
                for sep in ((",", ":"), (", ", ": ")):
                    s = json.dumps(obj, separators=sep, ensure_ascii=False)
                    if enc.encode(s) == hf.encode(s, add_special_tokens=False):
                        same += 1
                    else:
                        diff += 1
    save("tokenizer_identity.json", {"tiktoken": args.tiktoken,
                                     "identical": same, "different": diff})


def _post(base, body):
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def server_checks(args):
    import jsonschema

    from bench_structured import scrape_spec_metrics, spec_delta

    base = f"http://127.0.0.1:{args.port}"
    meta = json.loads(Path(args.meta).read_text()) if args.meta else None
    schema = {"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer"},
        "email": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "employment_status": {"type": "string", "enum": [
            "employed", "unemployed", "self_employed", "student"]}},
        "required": ["name", "age", "email", "skills", "employment_status"],
        "additionalProperties": False}
    res = {"server": meta, "schema_requests": [], "plain_chat": None}
    m0 = scrape_spec_metrics(base)
    for i, temp in enumerate([0.0, 0.0, 0.7, 1.0]):
        r = _post(base, {
            "model": MODEL, "temperature": temp, "max_tokens": 512, "seed": i,
            "messages": [{"role": "user", "content":
                          f"Generate a profile for a fictional software "
                          f"engineer #{i}. Reply in JSON."}],
            "structured_outputs": {"json": schema},
            "chat_template_kwargs": {"enable_thinking": False}})
        txt = r["choices"][0]["message"]["content"]
        ok = True
        try:
            jsonschema.validate(json.loads(txt), schema)
        except Exception:  # noqa: BLE001
            ok = False
        res["schema_requests"].append({
            "temperature": temp, "schema_ok": ok, "text": txt,
            "finish_reason": r["choices"][0]["finish_reason"],
            "completion_tokens": r["usage"]["completion_tokens"]})
    res["schema_requests_spec"] = spec_delta(m0, scrape_spec_metrics(base))
    prompts = ["Explain how a hash map works, with an example in Python.",
               "Write a short story about a lighthouse keeper.",
               "What are the main causes of the French Revolution?",
               "Give me a step-by-step recipe for banana bread."]
    m0 = scrape_spec_metrics(base)
    toks = 0
    for p in prompts:
        r = _post(base, {"model": MODEL, "temperature": 0, "max_tokens": 400,
                         "messages": [{"role": "user", "content": p}],
                         "chat_template_kwargs": {"enable_thinking": False}})
        toks += r["usage"]["completion_tokens"]
    res["plain_chat"] = {"num_requests": len(prompts), "completion_tokens": toks,
                         **spec_delta(m0, scrape_spec_metrics(base))}
    save(f"server_checks_{args.tag}.json", res)


def greedy_compare(args):
    root = Path(args.result_dir)
    cfgs = sorted(p.name for p in root.iterdir() if p.is_dir())
    cells = sorted(f.name.removesuffix(".outputs.jsonl")
                   for f in (root / cfgs[0]).glob("*.outputs.jsonl"))
    out = {"reference": cfgs[0], "cells": {}}
    for cell in cells:
        def load(cfg):
            with open(root / cfg / f"{cell}.outputs.jsonl") as f:
                return {json.loads(ln)["sid"]: json.loads(ln)["text"] for ln in f}

        ref = load(cfgs[0])
        out["cells"][cell] = {
            cfg: {"identical": sum(ref[k] == o.get(k) for k in ref),
                  "total": len(ref)}
            for cfg in cfgs[1:] for o in [load(cfg)]}
    save(f"greedy_compare_{root.name}.json", out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("resolve-config")
    sub.add_parser("xgrammar-probe")
    t = sub.add_parser("tokenizer-identity")
    t.add_argument("data_root")
    t.add_argument("tiktoken")
    s = sub.add_parser("server-checks")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--meta", default=None, help="server *.meta.json")
    s.add_argument("--tag", required=True)
    g = sub.add_parser("greedy-compare")
    g.add_argument("result_dir")
    args = p.parse_args()
    {"resolve-config": resolve_config, "xgrammar-probe": xgrammar_probe,
     "tokenizer-identity": tokenizer_identity, "server-checks": server_checks,
     "greedy-compare": greedy_compare}[args.cmd.replace("_", "-")](args)


if __name__ == "__main__":
    main()
