"""Run a benchmark matrix: for each server config, start vLLM, run every
(dataset, concurrency) cell with bench_structured.py, optionally collect
temperature-1 samples (sampling_check.py), stop the server.

Server configs are comma-separated tokens, e.g.
  none                     eagle3,k=5            eagle3,k=5,sync
  eagle3,k=5,sync,gd       ngram,k=5,sync,gdonly  eagle3,k=3,ws=default
(`gd` = grammar-draft plugin, hybrid mode; `gdonly` = grammar-only mode.)
Finished cells (result file exists) are skipped, so a sweep can be resumed.

Example:
  .venv/bin/python grammar_spec/scripts/run_matrix.py --tag main \
      --configs none eagle3,k=5 eagle3,k=5,sync,gd \
      --datasets jsonmode bfcl longschema --concurrency 1 4 16 64
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_structured import bench, make_parser  # noqa: E402
from sampling_check import collect as collect_samples  # noqa: E402
from server import Server  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "baseline"
SUMMARY_KEYS = [
    "num_ok", "num_errors", "mean_output_tokens", "output_throughput",
    "request_throughput", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_ttft_ms", "median_itl_ms", "schema_compliance_rate", "compact_rate",
    "duration_s",
]
# (min prompts, prompts per concurrency slot); long schemas emit ~10x tokens.
PROMPTS = {"jsonmode": (40, 6), "bfcl": (40, 6), "longschema": (16, 2)}


def parse_config(s):
    parts = s.split(",")
    cfg = {"method": parts[0], "k": 5, "ws": "compact", "sync": False,
           "runner": "auto", "grammar_draft": None}
    for p in parts[1:]:
        if p == "sync":
            cfg["sync"] = True
        elif p in ("v1", "v2"):
            cfg["runner"] = p
        elif p == "gd":
            cfg["grammar_draft"] = "hybrid"
        elif p == "gdonly":
            cfg["grammar_draft"] = "grammar_only"
        elif p.startswith("k="):
            cfg["k"] = int(p[2:])
        elif p.startswith("ws="):
            cfg["ws"] = p[3:]
        else:
            raise ValueError(f"bad config token {p!r} in {s!r}")
    return cfg


def num_prompts(ds, c, scale):
    lo, per_slot = PROMPTS[ds]
    return max(round(lo * scale), round(per_slot * scale) * c)


def run_cell(srv, cfg, ds, c, n, path, args):
    bargs = make_parser().parse_args([
        "--dataset", ds, "--num-prompts", str(n), "--max-concurrency", str(c),
        "--temperature", "0", "--num-warmup", str(min(c, 8)),
        "--reset-prefix-cache", "--out", str(path), "--save-outputs",
        *(["--expect-compact"] if cfg["ws"] == "compact" else []),
        *(["--bfcl-root", args.bfcl_root] if args.bfcl_root else []),
        *(["--gd-stats", srv.meta["gd_stats_path"]]
          if srv.meta["gd_stats_path"] else []),
    ])
    return bench(bargs, server_meta=srv.meta)


def summary_row(srv, cfg, ds, c, n, res, wall_s):
    gd = res.get("grammar_draft") or {}
    return {
        "run_id": srv.run_id, **cfg, "dataset": ds, "concurrency": c,
        "num_prompts": n, "async_scheduling": srv.meta["async_scheduling"],
        "model_runner": srv.meta["model_runner"],
        **{k: res[k] for k in SUMMARY_KEYS},
        "mean_acceptance_length": res["spec_decode"].get("mean_acceptance_length"),
        "gd_grammar_tokens_per_draft": gd.get("per_draft", {}).get(
            "drafted", {}).get("grammar"),
        "gd_grammar_acceptance": gd.get("acceptance_rate", {}).get("grammar"),
        "gd_propose_us": gd.get("per_draft", {}).get("propose_us"),
        "cell_wall_s": round(wall_s, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", default=list(PROMPTS))
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--prompt-scale", type=float, default=1.0)
    p.add_argument("--bfcl-root", default=None)
    p.add_argument("--sampling-n", type=int, default=0,
                   help="also collect this many temperature-1 samples")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    out_root = RESULTS / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    for cfg_str in args.configs:
        cfg = parse_config(cfg_str)
        srv = Server(**cfg)
        cells = [(ds, c) for ds in args.datasets for c in args.concurrency]
        path = {cell: out_root / srv.run_id / f"{cell[0]}_c{cell[1]}.json"
                for cell in cells}
        todo = [cell for cell in cells if not path[cell].exists()]
        sampling_path = out_root / srv.run_id / "sampling.jsonl"
        want_sampling = args.sampling_n and not sampling_path.exists()
        if not todo and not want_sampling:
            print(f"[skip] {srv.run_id}: all cells done", flush=True)
            continue
        t_cfg = time.time()
        with srv:
            print(f"[server] {srv.run_id} up in {srv.meta['startup_s']}s "
                  f"runner={srv.meta['model_runner']} "
                  f"async={srv.meta['async_scheduling']} "
                  f"grammar_draft={srv.meta['grammar_draft']}", flush=True)
            for ds, c in todo:
                n = num_prompts(ds, c, args.prompt_scale)
                t0 = time.time()
                res = run_cell(srv, cfg, ds, c, n, path[(ds, c)], args)
                row = summary_row(srv, cfg, ds, c, n, res, time.time() - t0)
                new = not summary_csv.exists()
                with open(summary_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"  {ds} c={c} n={n}: TPOT={res['mean_tpot_ms']:.2f}ms "
                      f"out_tps={res['output_throughput']:.0f} "
                      f"AL={row['mean_acceptance_length']} "
                      f"schema_ok={res['schema_compliance_rate']:.2f} "
                      f"wall={row['cell_wall_s']}s", flush=True)
            if want_sampling:
                collect_samples(srv.port, args.sampling_n, sampling_path)
                print(f"  sampling n={args.sampling_n} -> {sampling_path}",
                      flush=True)
        (out_root / srv.run_id / "server_meta.json").write_text(
            json.dumps(srv.meta | {"config_wall_s": round(time.time() - t_cfg, 1)},
                       indent=2))


if __name__ == "__main__":
    main()
