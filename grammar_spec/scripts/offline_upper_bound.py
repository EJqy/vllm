"""Offline upper bound: how many tokens of a BFCL tool-call JSON are decided
by the grammar alone, replayed through xgrammar with a real HF tokenizer.

Same method as the REPORT.md appendix, but the tokenizer and grammar matcher
are built the way vLLM's xgrammar backend builds them:
  * TokenizerInfo.from_huggingface(vocab_size=<model vocab>) with all special
    tokens filtered out (XgrammarBackend._filter_special_tokens);
  * compact mode = any_whitespace=False + separators=(",", ":");
  * stop tokens = `--stop vllm` (generation_config eos ids + tokenizer eos,
    i.e. SamplingParams.all_stop_token_ids) or `--stop single` (tokenizer eos).

Usage:
  .venv/bin/python grammar_spec/scripts/offline_upper_bound.py \
      <bfcl_eval/data dir> [--tokenizer Qwen/Qwen3-32B] [--stop vllm|single] \
      [--K 8] [--out results.json]
"""

import argparse
import json

import numpy as np
import xgrammar as xgr
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": None}
SPLITS = ("live_simple", "simple_python")


def conv(s):
    s = dict(s)
    t = s.get("type")
    if t in TYPE_MAP:
        if TYPE_MAP[t] is None:
            s.pop("type")
        else:
            s["type"] = TYPE_MAP[t]
    for k in ("description", "default", "format", "optional"):
        s.pop(k, None)
    if "properties" in s:
        s["properties"] = {k: conv(v) for k, v in s["properties"].items()}
        s["additionalProperties"] = False
    if "items" in s:
        s["items"] = conv(s["items"])
    return s


def build(root, split, form):
    docs = [json.loads(line) for line in open(f"{root}/BFCL_v4_{split}.json")]
    ans = {
        json.loads(line)["id"]: json.loads(line)
        for line in open(f"{root}/possible_answer/BFCL_v4_{split}.json")
    }
    out = []
    for d in docs:
        if d["id"] not in ans:
            continue
        f = d["function"][0]
        gt = ans[d["id"]]["ground_truth"][0]
        name, args = next(iter(gt.items()))
        if name != f["name"]:
            continue
        params = conv(f["parameters"])
        inst = {}
        for k, opts in args.items():
            if not opts:
                continue
            v = opts[0]
            if v == "" and k not in params.get("required", []):
                continue
            inst[k] = v
        if form == "args":
            out.append((params, inst))
        else:
            schema = {
                "type": "object",
                "properties": {"name": {"const": f["name"]}, "arguments": params},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            out.append((schema, {"name": f["name"], "arguments": inst}))
    return out


class Replayer:
    def __init__(self, tokenizer_name, stop_mode):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = AutoConfig.from_pretrained(tokenizer_name).vocab_size
        full_info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=vocab_size)
        self.info = XgrammarBackend._filter_special_tokens(self.tok, full_info)
        self.compiler = xgr.GrammarCompiler(self.info, max_threads=8)
        self.eos = self.tok.eos_token_id
        stop = {self.eos}
        if stop_mode == "vllm":
            gen_eos = GenerationConfig.from_pretrained(tokenizer_name).eos_token_id
            stop |= set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
        self.stop = sorted(stop)
        self.bitmask = xgr.allocate_token_bitmask(1, self.info.vocab_size)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def analyze(self, schema, obj, compact, K):
        sep = (",", ":") if compact else (", ", ": ")
        text = json.dumps(obj, separators=sep, ensure_ascii=False)
        toks = self.encode(text) + [self.eos]
        g = self.compiler.compile_json_schema(
            json.dumps(schema),
            any_whitespace=not compact,
            separators=sep if compact else None,
        )
        m = xgr.GrammarMatcher(g, override_stop_tokens=self.stop)
        forced, jf_hit, n_allowed_list = [], [], []
        for t in toks:
            m.fill_next_token_bitmask(self.bitmask)
            bits = self.bitmask[0].numpy().view(np.uint8)
            n_allowed = int(np.unpackbits(bits).sum())
            is_forced = n_allowed == 1
            jf = m.find_jump_forward_string() if not is_forced else ""
            jf_ok = bool(jf) and self.encode(jf)[:1] == [t]
            forced.append(is_forced)
            jf_hit.append(jf_ok)
            n_allowed_list.append(n_allowed)
            if not m.accept_token(t):
                return None
        free = [a or b for a, b in zip(forced, jf_hit)]
        res = dict(
            n=len(toks),
            forced=sum(forced),
            forced_eos=int(forced[-1]),
            jf=sum(jf_hit),
            free=sum(free),
        )
        for k in K:
            res[f"steps_K{k}"] = simulate(free, k)
        return res


def simulate(free, K):
    """Grammar-only drafting: each target pass verifies the run of free tokens
    (at most K) that starts at the current position and emits one more."""
    n, i, steps = len(free), 0, 0
    while i < n:
        acc = 0
        while acc < K and i + acc < n and free[i + acc]:
            acc += 1
        steps += 1
        i += acc + 1
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    p.add_argument("--stop", choices=["vllm", "single"], default="vllm")
    p.add_argument("--K", type=int, nargs="+", default=[8, 5])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    r = Replayer(args.tokenizer, args.stop)
    print(
        f"tokenizer={args.tokenizer} vocab_size={r.info.vocab_size} "
        f"eos={r.eos} stop_tokens={r.stop}"
    )
    rows = []
    for form in ("full", "args"):
        data = [x for split in SPLITS for x in build(args.data_root, split, form)]
        for compact in (False, True):
            tot = dict(n=0, forced=0, forced_eos=0, jf=0, free=0)
            tot.update({f"steps_K{k}": 0 for k in args.K})
            cnt = bad = 0
            for schema, obj in data:
                try:
                    res = r.analyze(schema, obj, compact, args.K)
                except Exception:
                    res = None
                if res is None:
                    bad += 1
                    continue
                cnt += 1
                for k in tot:
                    tot[k] += res[k]
            n = tot["n"]
            row = dict(
                schema_form="name+arguments" if form == "full" else "arguments",
                compact=compact,
                samples=cnt,
                skipped=bad,
                tokens=n,
                avg_len=n / cnt,
                tier1_forced=tot["forced"] / n,
                tier1_forced_excl_eos=(tot["forced"] - tot["forced_eos"]) / n,
                tier2_jump_fwd_extra=tot["jf"] / n,
                grammar_free=tot["free"] / n,
            )
            for k in args.K:
                row[f"target_passes_saved_K{k}"] = 1 - tot[f"steps_K{k}"] / n
            rows.append(row)
            print(
                f"{row['schema_form']:15s} compact={compact!s:5s} samples={cnt} "
                f"skipped={bad} avg_len={row['avg_len']:.1f} "
                f"forced={row['tier1_forced']:.1%} "
                f"(excl. EOS {row['tier1_forced_excl_eos']:.1%}) "
                f"jump_fwd_extra={row['tier2_jump_fwd_extra']:.1%} "
                f"grammar_free={row['grammar_free']:.1%} "
                + " ".join(
                    f"saved_K{k}={row[f'target_passes_saved_K{k}']:.1%}"
                    for k in args.K
                )
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                dict(tokenizer=args.tokenizer, stop_tokens=r.stop, rows=rows),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
