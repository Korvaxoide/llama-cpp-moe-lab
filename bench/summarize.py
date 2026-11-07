#!/usr/bin/env python3
"""Summarize results/bench_*.jsonl -> markdown table + results/summary.csv (median over reps).

Usage:
    python3 bench/summarize.py             # print all tags, regenerate summary.csv (all tags)
    python3 bench/summarize.py TAG [TAG…]  # print only those tags; summary.csv is still
                                           # regenerated with ALL tags (never truncated)
"""
import glob, json, os, statistics, sys, csv

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for f in sorted(glob.glob(os.path.join(LAB, "results", "bench_*.jsonl"))):
    for line in open(f):
        rows.append(json.loads(line))

all_tags = sorted({r["tag"] for r in rows}, key=lambda t: [r["tag"] for r in rows].index(t))
selected = sys.argv[1:] or all_tags
med = lambda xs: round(statistics.median(xs), 2) if xs else None


def summarize(tags):
    groups = {}
    for r in rows:
        if r["tag"] in tags:
            groups.setdefault((r["tag"], r["depth"]), []).append(r)
    out = []
    for (tag, depth), rs in sorted(groups.items(), key=lambda kv: (tags.index(kv[0][0]), kv[0][1])):
        acc = [r["acc_rate"] for r in rs if r.get("acc_rate") is not None]
        out.append({"tag": tag, "depth": depth, "n": len(rs), "pp_tps": med([r["pp_tps"] for r in rs]),
                    "tg_tps": med([r["tg_tps"] for r in rs]), "tg_min": min(r["tg_tps"] for r in rs),
                    "tg_max": max(r["tg_tps"] for r in rs), "acc": med(acc),
                    "ttft_s": med([r["prompt_ms"] / 1000 for r in rs]),
                    "vram_max": max((r.get("vram_max") or 0) for r in rs),
                    "ram_avail_min": min((r.get("ram_avail_min") or 0) for r in rs),
                    "n_ctx": rs[0]["n_ctx"]})
    return out


def print_table(out):
    print("| tag | depth | n | PP t/s | TG t/s (min-max) | MTP acc | TTFT s | VRAM max MiB | RAM avail min | n_ctx |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for o in out:
        print(f"| {o['tag']} | {o['depth']} | {o['n']} | {o['pp_tps']} | {o['tg_tps']} ({o['tg_min']}-{o['tg_max']}) | {o['acc']} | {o['ttft_s']} | {o['vram_max']} | {o['ram_avail_min']} | {o['n_ctx']} |")


print_table(summarize(selected))

# Always regenerate the consolidated CSV with every tag: a filtered invocation
# (or a wrong argument) must never destroy previous results.
full = summarize(all_tags)
if full:
    path = os.path.join(LAB, "results", "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(full[0].keys()))
        w.writeheader()
        w.writerows(full)
