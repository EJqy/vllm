"""Estimate GPU hours of the REPORT.md 5.2 baseline matrix (and a pruned
plan) from measured smoke-matrix numbers.

Model per cell:  n(c)/c * (TTFT(c) + L_ds * TPOT(c)) + CELL_OVERHEAD_S
  * TPOT/TTFT measured at c=1 and c=16; c=4 interpolated and c=64
    extrapolated log-linearly in c (assumption, to be replaced by data).
  * Two speed classes: "plain" (no-spec numbers; used conservatively for
    ngram/suffix/grammar-only) and "eagle" (EAGLE-3 numbers).
  * L_ds for jsonmode/bfcl is measured; jsonschemabench and long_schema are
    assumptions until those loaders exist.
Per server: SERVER_OVERHEAD_S for startup + shutdown (measured 98-167 s).

Usage:
  .venv/bin/python grammar_spec/scripts/estimate_gpu_hours.py \
      grammar_spec/results/baseline/stage0_smoke
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

OUT_LEN = {"jsonmode": None, "bfcl": None, "jsonschemabench": 200,
           "long_schema": 600}
ALL_DS = list(OUT_LEN)
ALL_C = [1, 4, 16, 64]
CELL_OVERHEAD_S = 15
SERVER_OVERHEAD_S = 150


def n_prompts(ds, c):
    if ds == "long_schema":
        return max(50, 5 * c)
    return max(100, 10 * c)


def load(root):
    meas = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in d.glob("*_c*.json"):
            ds, c = f.stem.rsplit("_c", 1)
            meas[(d.name, ds, int(c))] = json.loads(f.read_text())
    return meas


def loglin(v1, v16, c):
    slope = (math.log(v16) - math.log(v1)) / math.log(16)
    return math.exp(math.log(v1) + slope * math.log(c))


def speed_class(meas, run_id):
    def avg(key, c):
        return sum(meas[(run_id, ds, c)][key] for ds in ("jsonmode", "bfcl")) / 2e3

    return {k: (avg(k, 1), avg(k, 16)) for k in ("mean_tpot_ms", "mean_ttft_ms")}


def cell_s(sc, ds, c):
    tpot = loglin(*sc["mean_tpot_ms"], c)
    ttft = loglin(*sc["mean_ttft_ms"], c)
    return n_prompts(ds, c) / c * (ttft + OUT_LEN[ds] * tpot) + CELL_OVERHEAD_S


def plan_hours(groups, classes):
    """groups: list of (label, n_servers, speed, datasets, concurrencies)."""
    total, by_ds = 0.0, defaultdict(float)
    rows = []
    for label, n, speed, dss, cs in groups:
        per_server = SERVER_OVERHEAD_S
        for ds in dss:
            for c in cs:
                t = cell_s(classes[speed], ds, c)
                per_server += t
                by_ds[ds] += n * t
        total += n * per_server
        rows.append((label, n, n * len(dss) * len(cs), n * per_server / 3600))
    return total / 3600, rows, {k: v / 3600 for k, v in by_ds.items()}


def show(name, groups, classes):
    h, rows, by_ds = plan_hours(groups, classes)
    print(f"\n=== {name}: {h:.1f} GPU-h per repetition")
    for label, n, cells, gh in rows:
        print(f"  {label:42s} servers={n:3d} cells={cells:4d} {gh:6.1f} h")
    print("  by dataset:", {k: round(v, 1) for k, v in by_ds.items()})
    return h


def main():
    meas = load(sys.argv[1])
    for ds in ("jsonmode", "bfcl"):
        cells = [r for (_, d, _), r in meas.items() if d == ds]
        toks = sum(r["total_output_tokens"] for r in cells)
        OUT_LEN[ds] = toks / sum(r["num_ok"] for r in cells)
    classes = {
        "plain": speed_class(meas, "none_compact"),
        "eagle": speed_class(meas, "eagle3_k5_compact"),
    }
    print("mean output tokens:", {k: round(v) for k, v in OUT_LEN.items()})
    for k, sc in classes.items():
        tp = sc["mean_tpot_ms"]
        print(f"{k}: TPOT c1={tp[0]*1e3:.1f} c16={tp[1]*1e3:.1f} "
              f"c64~{loglin(*tp, 64)*1e3:.1f} ms")

    K, WS = 7, 2
    full_baselines = [
        ("no spec x ws", WS, "plain", ALL_DS, ALL_C),
        ("ngram, suffix x K1-7 x ws", 2 * K * WS, "plain", ALL_DS, ALL_C),
        ("eagle3 async, sync x K1-7 x ws", 2 * K * WS, "eagle", ALL_DS, ALL_C),
    ]
    own = [
        ("grammar-only x K1-7 x ws", K * WS, "plain", ALL_DS, ALL_C),
        ("route A, route B x K1-7 x ws", 2 * K * WS, "eagle", ALL_DS, ALL_C),
    ]
    show("full 5.2 matrix, baselines only", full_baselines, classes)
    show("full 5.2 matrix incl. own 3 methods", full_baselines + own, classes)

    short = ["jsonmode", "bfcl"]
    pruned = [
        # Phase 1: pick K on the short datasets only.
        ("K-select: ngram, suffix K{1,3,5,7}", 8, "plain", short, [1, 16, 64]),
        ("K-select: eagle3 async K{1,3,5,7}", 4, "eagle", short, [1, 16, 64]),
        # Phase 2: main grid at the chosen K.
        ("main: no spec x ws", 2, "plain", ALL_DS, ALL_C),
        ("main: eagle3 async x 2 best K x ws", 4, "eagle", ALL_DS, ALL_C),
        ("main: ngram, suffix best K, compact", 2, "plain", ALL_DS, ALL_C),
        ("main: eagle3 sync best K, compact", 1, "eagle", short, ALL_C),
        # Repeat the headline cells twice more.
        ("repeat: no spec + eagle3 async, compact", 2 * 2, "eagle", ALL_DS,
         ALL_C),
    ]
    show("pruned baseline plan (incl. 3x headline)", pruned, classes)


if __name__ == "__main__":
    main()
