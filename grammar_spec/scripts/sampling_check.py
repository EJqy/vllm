"""Temperature-1 distribution check (REPORT.md 5.1 item 2).

`collect` draws N independent samples of a small JSON object from a running
server. `compare` tests whether two sample sets come from the same
distribution, per field and jointly: chi-square test of homogeneity plus
total variation distance.

Usage:
  .venv/bin/python grammar_spec/scripts/sampling_check.py compare \
      --out result.json A.jsonl B.jsonl [C.jsonl ...]
"""

import argparse
import asyncio
import itertools
import json
from collections import Counter
from pathlib import Path

import aiohttp
from scipy.stats import chi2_contingency

MODEL = "Qwen/Qwen3-32B"
CITIES = ["Paris", "Tokyo", "Cairo", "Lima", "Oslo", "Sydney", "Toronto", "Mumbai"]
CONDITIONS = ["sunny", "cloudy", "rainy", "snowy", "windy"]
SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "enum": CITIES},
        "condition": {"type": "string", "enum": CONDITIONS},
        "temperature_celsius": {"type": "integer"},
        "advice": {"type": "string"},
    },
    "required": ["city", "condition", "temperature_celsius", "advice"],
    "additionalProperties": False,
}
PROMPT = ("Invent a short weather report for a randomly chosen city, with one "
          "sentence of advice. Reply in JSON.")


async def _collect(port, n, seed, concurrency):
    sem = asyncio.Semaphore(concurrency)
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    async def one(session, i):
        body = {"model": MODEL, "temperature": 1.0, "top_p": 1.0, "max_tokens": 96,
                "seed": seed + i, "messages": [{"role": "user", "content": PROMPT}],
                "structured_outputs": {"json": SCHEMA},
                "chat_template_kwargs": {"enable_thinking": False}}
        async with sem, session.post(url, json=body) as r:
            out = await r.json()
            return {"i": i, "text": out["choices"][0]["message"]["content"]}

    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*(one(session, i) for i in range(n)))


def collect(port, n, path, seed=0, concurrency=32):
    rows = asyncio.run(_collect(port, n, seed, concurrency))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _features(text):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    t = obj["temperature_celsius"]
    bucket = "<0" if t < 0 else ">=30" if t >= 30 else f"{t // 10 * 10}s"
    return {
        "city": obj["city"],
        "condition": obj["condition"],
        "temperature_bucket": bucket,
        "city_x_condition": f"{obj['city']}|{obj['condition']}",
        "advice_first_word": (obj["advice"].split() or [""])[0].lower(),
    }


def _load(path):
    feats = [_features(json.loads(line)["text"]) for line in open(path)]
    return [f for f in feats if f is not None], sum(f is None for f in feats)


def _test(counts_a, counts_b, min_count=5):
    """Chi-square homogeneity test; rare categories pooled into 'other'."""
    keys = sorted(set(counts_a) | set(counts_b))
    common = [k for k in keys if counts_a[k] + counts_b[k] >= min_count]
    rare = [k for k in keys if k not in common]
    table = [[counts_a[k] for k in common], [counts_b[k] for k in common]]
    if rare:
        table[0].append(sum(counts_a[k] for k in rare))
        table[1].append(sum(counts_b[k] for k in rare))
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    tv = 0.5 * sum(abs(counts_a[k] / na - counts_b[k] / nb) for k in keys)
    if len(table[0]) < 2:
        return {"tv": tv, "p_value": 1.0, "categories": len(table[0])}
    _, p, dof, _ = chi2_contingency(table)
    return {"tv": tv, "p_value": float(p), "dof": int(dof),
            "categories": len(table[0])}


def compare(paths, out):
    data = {Path(p).parent.name + "/" + Path(p).stem: _load(p) for p in paths}
    result = {"n": {k: len(v[0]) for k, v in data.items()},
              "invalid": {k: v[1] for k, v in data.items()}, "pairs": {}}
    for (ka, (fa, _)), (kb, (fb, _)) in itertools.combinations(data.items(), 2):
        result["pairs"][f"{ka} vs {kb}"] = {
            field: _test(Counter(f[field] for f in fa), Counter(f[field] for f in fb))
            for field in fa[0]
        }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--n", type=int, default=1000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("paths", nargs="+")
    m.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "collect":
        collect(args.port, args.n, args.out, args.seed)
    else:
        compare(args.paths, args.out)


if __name__ == "__main__":
    main()
