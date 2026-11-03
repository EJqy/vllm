"""Run a baseline matrix: for each server config, start vLLM, run every
(dataset, concurrency) cell with bench_structured.py, stop the server.

Server configs are given as comma-separated key list, e.g.
  none,ws=compact          eagle3,k=5,ws=default          eagle3,k=5,sync
  ngram,k=3                suffix,k=5,ws=compact,v1
Finished cells (result file exists) are skipped, so a sweep can be resumed.

Example (stage-0 smoke matrix):
  .venv/bin/python grammar_spec/scripts/run_matrix.py \
      --configs none,ws=compact eagle3,k=5,ws=compact eagle3,k=5,ws=compact,sync \
      --datasets jsonmode --concurrency 1 16 --min-prompts 20 --waves 2 \
      --tag smoke
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_structured import bench, make_parser  # noqa: E402
from server import Server  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "baseline"
SUMMARY_KEYS = [
    "num_ok", "num_errors", "mean_output_tokens", "output_throughput",
    "request_throughput", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_ttft_ms", "median_itl_ms", "schema_compliance_rate", "compact_rate",
    "duration_s",
]


def parse_config(s):
    parts = s.split(",")
    cfg = {"method": parts[0], "k": 5, "ws": "compact", "sync": False,
           "runner": "auto"}
    for p in parts[1:]:
        if p == "sync":
            cfg["sync"] = True
        elif p in ("v1", "v2"):
            cfg["runner"] = p
        elif p.startswith("k="):
            cfg["k"] = int(p[2:])
        elif p.startswith("ws="):
            cfg["ws"] = p[3:]
        else:
            raise ValueError(f"bad config token {p!r} in {s!r}")
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", default=["jsonmode"])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--min-prompts", type=int, default=50)
    p.add_argument("--waves", type=int, default=8,
                   help="num_prompts = max(min_prompts, waves * concurrency)")
    p.add_argument("--bfcl-root", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tag", default="baseline")
    args = p.parse_args()

    out_root = RESULTS / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    for cfg_str in args.configs:
        cfg = parse_config(cfg_str)
        cells = [
            (ds, c) for ds in args.datasets for c in args.concurrency
        ]
        srv = Server(**cfg)
        cell_path = {
            (ds, c): out_root / srv.run_id / f"{ds}_c{c}.json" for ds, c in cells
        }
        todo = [cell for cell in cells if not cell_path[cell].exists()]
        if not todo:
            print(f"[skip] {srv.run_id}: all cells done")
            continue
        t_cfg = time.time()
        with srv:
            print(f"[server] {srv.run_id} up in {srv.meta['startup_s']}s "
                  f"runner={srv.meta['model_runner']} "
                  f"async={srv.meta['async_scheduling']}", flush=True)
            for ds, c in todo:
                n = max(args.min_prompts, args.waves * c)
                bargs = make_parser().parse_args([
                    "--dataset", ds, "--num-prompts", str(n),
                    "--max-concurrency", str(c),
                    "--temperature", str(args.temperature),
                    "--num-warmup", str(min(c, 8)),
                    "--reset-prefix-cache",
                    "--out", str(cell_path[(ds, c)]), "--save-outputs",
                    *(["--expect-compact"] if cfg["ws"] == "compact" else []),
                    *(["--bfcl-root", args.bfcl_root] if args.bfcl_root else []),
                ])
                t0 = time.time()
                res = bench(bargs, server_meta=srv.meta)
                row = {"run_id": srv.run_id, **cfg, "dataset": ds,
                       "concurrency": c, "num_prompts": n,
                       "async_scheduling": srv.meta["async_scheduling"],
                       "model_runner": srv.meta["model_runner"],
                       **{k: res[k] for k in SUMMARY_KEYS},
                       "mean_acceptance_length":
                           res["spec_decode"].get("mean_acceptance_length"),
                       "cell_wall_s": round(time.time() - t0, 1)}
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
        (out_root / srv.run_id / "server_meta.json").write_text(
            json.dumps(srv.meta | {"config_wall_s": round(time.time() - t_cfg, 1)},
                       indent=2))


if __name__ == "__main__":
    main()
