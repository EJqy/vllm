"""Structured-output serving benchmark against a running vLLM server.

Sends chat requests with a JSON schema (`structured_outputs.json`) and
`enable_thinking=False`, measures TTFT / TPOT / ITL / throughput, and reads
spec-decode counters from /metrics before and after the run to get the mean
acceptance length and per-position acceptance. Every output is checked with
`jsonschema` and, for compact runs, for whitespace outside strings.

Datasets:
  jsonmode  NousResearch/json-mode-eval (same source and unsupported-feature
            filter as vLLM's `--dataset xgrammar_bench`)
  bfcl      BFCL v4 simple_python + live_simple, schema = {"name": const,
            "arguments": <params>} as in the offline analysis

Usage:
  .venv/bin/python grammar_spec/scripts/bench_structured.py --dataset jsonmode \
      --num-prompts 100 --max-concurrency 16 --out result.json
"""

import argparse
import asyncio
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longschema import make_samples as make_longschema_samples  # noqa: E402
from offline_upper_bound import conv  # noqa: E402

MODEL = "Qwen/Qwen3-32B"
BFCL_SPLITS = ("simple_python", "live_simple")
TOOL_SYSTEM_PROMPT = (
    "You are a function-calling assistant. You have access to this function:\n"
    "{fn}\n"
    'Answer only with a JSON object of the form {{"name": <function name>, '
    '"arguments": <arguments object>}} that calls the function to answer the '
    "user's request."
)


@dataclass
class Sample:
    sid: str
    messages: list
    schema: dict


@dataclass
class Record:
    sid: str
    ok: bool = False
    error: str = ""
    ttft: float = 0.0
    latency: float = 0.0
    itl: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    text: str = ""
    valid_json: bool = False
    schema_ok: bool = False
    compact_ok: bool = True


def load_jsonmode():
    import datasets

    from vllm.v1.structured_output.backend_xgrammar import (
        has_xgrammar_unsupported_json_features,
    )

    ds = datasets.load_dataset("NousResearch/json-mode-eval", split="train")
    out = []
    for i, row in enumerate(ds):
        schema = json.loads(row["schema"])
        if has_xgrammar_unsupported_json_features(schema):
            continue
        out.append(Sample(f"jsonmode-{i}", list(row["prompt"]), schema))
    return out


def load_bfcl(root):
    out = []
    for split in BFCL_SPLITS:
        for line in open(Path(root) / f"BFCL_v4_{split}.json"):
            d = json.loads(line)
            f = d["function"][0]
            schema = {
                "type": "object",
                "properties": {
                    "name": {"const": f["name"]},
                    "arguments": conv(f["parameters"]),
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            turn = d["question"][0]
            sys_msgs = [m["content"] for m in turn if m["role"] == "system"]
            system = TOOL_SYSTEM_PROMPT.format(fn=json.dumps(f, ensure_ascii=False))
            if sys_msgs:
                system = "\n\n".join(sys_msgs + [system])
            msgs = [{"role": "system", "content": system}] + [
                m for m in turn if m["role"] != "system"
            ]
            out.append(Sample(d["id"], msgs, schema))
    return out


def select(samples, n, seed):
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    return [samples[order[i % len(order)]] for i in range(n)]


def has_ws_outside_strings(text):
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in " \t\r\n":
            return True
    return False


def scrape_spec_metrics(base_url):
    txt = urllib.request.urlopen(f"{base_url}/metrics", timeout=10).read().decode()
    out = {"per_pos": {}}
    for line in txt.splitlines():
        if not line.startswith("vllm:spec_decode_num_") or "_created" in line:
            continue
        key, val = line.rsplit(" ", 1)
        name = key.split("{")[0]
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = int(key.split('position="')[1].split('"')[0])
            out["per_pos"][pos] = float(val)
        else:
            out[name] = float(val)
    return out


def spec_delta(m0, m1):
    def d(name):
        return m1.get(name, 0.0) - m0.get(name, 0.0)

    drafts = d("vllm:spec_decode_num_drafts_total")
    res = {
        "num_drafts": drafts,
        "num_draft_tokens": d("vllm:spec_decode_num_draft_tokens_total"),
        "num_accepted_tokens": d("vllm:spec_decode_num_accepted_tokens_total"),
    }
    if drafts > 0:
        res["mean_acceptance_length"] = 1 + res["num_accepted_tokens"] / drafts
        res["draft_acceptance_rate"] = (
            res["num_accepted_tokens"] / res["num_draft_tokens"]
        )
        res["per_pos_acceptance"] = [
            (m1["per_pos"][p] - m0.get("per_pos", {}).get(p, 0.0)) / drafts
            for p in sorted(m1["per_pos"])
        ]
    return res


async def one_request(session, base_url, s, args, idx):
    rec = Record(sid=s.sid)
    body = {
        "model": MODEL,
        "messages": s.messages,
        "max_tokens": args.output_len,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "structured_outputs": {"json": s.schema},
    }
    if args.thinking == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.temperature > 0:
        body["seed"] = args.seed + idx
    chunks = []
    t0 = time.perf_counter()
    last = None
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=body) as r:
            if r.status != 200:
                rec.error = f"HTTP {r.status}: {(await r.text())[:300]}"
                return rec
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    rec.prompt_tokens = obj["usage"]["prompt_tokens"]
                    rec.completion_tokens = obj["usage"]["completion_tokens"]
                for ch in obj.get("choices", []):
                    piece = ch.get("delta", {}).get("content")
                    if piece:
                        now = time.perf_counter()
                        if last is None:
                            rec.ttft = now - t0
                        else:
                            rec.itl.append(now - last)
                        last = now
                        chunks.append(piece)
                    if ch.get("finish_reason"):
                        rec.finish_reason = ch["finish_reason"]
        rec.latency = time.perf_counter() - t0
        rec.text = "".join(chunks)
        rec.ok = True
    except Exception as e:  # noqa: BLE001
        rec.error = f"{type(e).__name__}: {e}"
        return rec
    try:
        obj = json.loads(rec.text)
        rec.valid_json = True
        jsonschema.validate(obj, s.schema)
        rec.schema_ok = True
    except Exception:  # noqa: BLE001
        pass
    if args.expect_compact:
        rec.compact_ok = not has_ws_outside_strings(rec.text)
    return rec


def pct(xs, q):
    return float(np.percentile(xs, q)) * 1000 if xs else float("nan")


def summarize(recs, duration):
    ok = [r for r in recs if r.ok]
    tpots = [
        (r.latency - r.ttft) / (r.completion_tokens - 1)
        for r in ok
        if r.completion_tokens > 1
    ]
    ttfts = [r.ttft for r in ok]
    itls = [x for r in ok for x in r.itl]
    e2e = [r.latency for r in ok]
    out_toks = sum(r.completion_tokens for r in ok)
    n = max(len(ok), 1)
    return {
        "num_requests": len(recs),
        "num_ok": len(ok),
        "num_errors": len(recs) - len(ok),
        "errors_sample": [r.error for r in recs if not r.ok][:5],
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": out_toks / duration,
        "total_output_tokens": out_toks,
        "mean_output_tokens": out_toks / n,
        "mean_prompt_tokens": sum(r.prompt_tokens for r in ok) / n,
        "finish_length": sum(r.finish_reason == "length" for r in ok),
        "valid_json_rate": sum(r.valid_json for r in ok) / n,
        "schema_compliance_rate": sum(r.schema_ok for r in ok) / n,
        "compact_rate": sum(r.compact_ok for r in ok) / n,
        "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else float("nan"),
        "median_ttft_ms": pct(ttfts, 50),
        "p99_ttft_ms": pct(ttfts, 99),
        "mean_tpot_ms": float(np.mean(tpots)) * 1000 if tpots else float("nan"),
        "median_tpot_ms": pct(tpots, 50),
        "p99_tpot_ms": pct(tpots, 99),
        "mean_itl_ms": float(np.mean(itls)) * 1000 if itls else float("nan"),
        "median_itl_ms": pct(itls, 50),
        "p99_itl_ms": pct(itls, 99),
        "mean_e2e_ms": float(np.mean(e2e)) * 1000 if e2e else float("nan"),
    }


async def run(samples, args, base_url):
    sem = asyncio.Semaphore(args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        async def guarded(i, s):
            async with sem:
                return await one_request(session, base_url, s, args, i)

        if args.num_warmup:
            await asyncio.gather(
                *(guarded(i, s) for i, s in enumerate(samples[: args.num_warmup]))
            )
        if args.reset_prefix_cache:
            async with session.post(f"{base_url}/reset_prefix_cache") as r:
                await r.read()
        gd0 = await read_grammar_draft_stats(args.gd_stats)
        m0 = scrape_spec_metrics(base_url)
        t0 = time.perf_counter()
        recs = await asyncio.gather(*(guarded(i, s) for i, s in enumerate(samples)))
        duration = time.perf_counter() - t0
        m1 = scrape_spec_metrics(base_url)
        gd1 = await read_grammar_draft_stats(args.gd_stats)
    gd = grammar_draft_delta(gd0, gd1) if gd0 is not None else None
    return recs, duration, spec_delta(m0, m1), gd


async def read_grammar_draft_stats(path):
    """The plugin's counters (GRAMMAR_DRAFT_STATS), after its next flush."""
    if not path:
        return None
    await asyncio.sleep(0.6)  # the plugin flushes every 0.5 s
    return json.loads(Path(path).read_text())


def grammar_draft_delta(s0, s1):
    def sub(a, b):
        return {k: sub(a[k], b[k]) if isinstance(a[k], dict) else a[k] - b[k]
                for k in a}

    d = sub(s1, s0)
    drafts = max(d["num_drafts"], 1)
    d["per_draft"] = {
        "drafted": {k: v / drafts for k, v in d["drafted"].items()},
        "accepted": {k: v / drafts for k, v in d["accepted"].items()},
        "propose_us": d["propose_seconds"] / drafts * 1e6,
    }
    d["acceptance_rate"] = {
        k: d["accepted"][k] / d["drafted"][k] if d["drafted"][k] else None
        for k in d["drafted"]
    }
    return d


def bench(args, server_meta=None):
    base_url = f"http://127.0.0.1:{args.port}"
    if args.dataset == "jsonmode":
        pool = load_jsonmode()
    elif args.dataset == "bfcl":
        pool = load_bfcl(args.bfcl_root)
    else:
        pool = [Sample(*x) for x in make_longschema_samples()]
    samples = select(pool, args.num_prompts, args.seed)
    recs, duration, spec, gd = asyncio.run(run(samples, args, base_url))
    result = {
        "args": vars(args),
        "server": server_meta,
        "dataset_pool_size": len(pool),
        **summarize(recs, duration),
        "spec_decode": spec,
        "grammar_draft": gd,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        if args.save_outputs:
            with open(Path(args.out).with_suffix(".outputs.jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps({
                        "sid": r.sid, "text": r.text,
                        "completion_tokens": r.completion_tokens,
                        "schema_ok": r.schema_ok, "error": r.error,
                    }, ensure_ascii=False) + "\n")
    return result


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["jsonmode", "bfcl", "longschema"],
                   default="jsonmode")
    p.add_argument("--gd-stats", default=None,
                   help="GRAMMAR_DRAFT_STATS file of the server, if any")
    p.add_argument("--bfcl-root", default=None)
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--output-len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=4)
    p.add_argument("--reset-prefix-cache", action="store_true")
    p.add_argument("--expect-compact", action="store_true")
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", default=None)
    p.add_argument("--save-outputs", action="store_true")
    p.add_argument("--thinking", choices=["off", "template-default"],
                   default="off",
                   help="'off' passes enable_thinking=False (default); "
                   "'template-default' leaves the Qwen3 chat template as is")
    return p


def main():
    args = make_parser().parse_args()
    res = bench(args)
    keys = ["num_ok", "num_errors", "mean_output_tokens", "output_throughput",
            "mean_tpot_ms", "median_tpot_ms", "mean_ttft_ms",
            "schema_compliance_rate", "compact_rate"]
    print(json.dumps({k: res[k] for k in keys} | {"spec": res["spec_decode"]},
                     indent=2))


if __name__ == "__main__":
    main()
