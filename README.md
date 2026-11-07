# llama-cpp-moe-lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Korvaxoide/llama-cpp-moe-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Korvaxoide/llama-cpp-moe-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.x%20%C2%B7%20stdlib-blue)

Optimization lab for **llama.cpp** on a single workstation
(RTX 3080 10 GB + Ryzen 7 3700X, PCIe Gen3 x16), with a single model:
`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (Qwen3.6 hybrid MoE, 262 144 context).

> **TL;DR (English).** Reproducible tuning of `llama-server` for a 35B MoE model on a
> single 10 GB GPU with a 200K-token context. One variable per experiment, every number
> traced to `results/*.jsonl` or `logs/`. Result versus the measured baseline:
> prompt processing **+292 % at 8K** and **+213 % at 192K**, generation **+60…+75 %**,
> time-to-first-token **−75 %**, with coding quality unchanged (10/10).
> The dominant bottleneck was **PCIe Gen3 x16**, not the GPU, and the main levers were
> batch size, KV quantization and MoE placement. ReBAR was measured and gave **no benefit**.
> Documentation is in English; the data and scripts are language-neutral.

## Results

Production configuration (`qwen36-35b-opt`, profile **F1r**) compared to the
original configuration measured on the same hardware:

| Metric | Baseline | Production (F1r) | Δ |
|---|---|---|---|
| Prompt processing 8K / 192K | 445 / 386 t/s | **1745 / 1209 t/s** | **+292 % / +213 %** |
| Generation 8K / 32K / 192K | 31.4 / 28.5 / 28.9 t/s | **50.1 / 50.1 / 32.7 t/s** | **+60 % / +75 % / +13 %** |
| TTFT 32K / 128K | 73.9 / 328.6 s | **18.4 / 93.9 s** | **−75 % / −71 %** |
| Agentic work | 6.7 task/min | **8.82 task/min** | **+32 %** |
| Quality (10 coding tasks) | 10/10 | **10/10** | unchanged |

Sources: [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md).

## What the tests demonstrate

- **The bottleneck was the bus, not the GPU.** The baseline was saturated on PCIe
  Gen3 x16 (~12.7 GB/s): every large ubatch must transfer the MoE experts from RAM.
  The bandwidth calculation is in [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) §4.1.
- **ubatch size is the main lever on prompt processing.** 512 → 1024 → 2048 → 3072
  is worth about +69 %, +71 % and +18…25 % per step, as long as VRAM allows it.
- **The MoE cache hurts generation.** With a 1024 MiB cache TG was 32.3 t/s;
  with cache 0 it is 42.2 t/s, because every miss costs a PCIe transfer while the CPU
  computes the experts from DDR4 faster.
- **KV q8_0 is not a speedup, it is an enabler.** It frees 1.8 GiB of VRAM that
  are needed for the larger ubatch and to keep 3 layers of experts on the GPU
  (`--n-cpu-moe 37`), with no measured quality loss (needle 5/5, coding 10/10).
- **Decoupling the MTP ubatch frees 883 MiB** (`--spec-draft-n-ubatch 512`):
  the compute buffer of the draft context drops from 942 to 235 MiB at the cost of 1 % of PP.
- **ReBAR is not needed** (dedicated A/B test): it does not raise Gen3 bandwidth and
  reduces available VRAM by ~300 MiB. Documented in [`docs/RESULTS.md`](docs/RESULTS.md) §1.3.

## Mission

To find, with reproducible measurements and one variable at a time, the best
`llama-server` configuration for this model on this hardware, **without reducing
the context below 200 000 tokens** and without degrading quality. Every decision is
made according to rule §3.5 of [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): a
gain ≥ 5 % on PP or TG, or declared neutrality; regressions are DISCARDED even
if they "seem" reasonable.

## Lab constraints

- A single model: `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`. **Never** copied,
  converted or modified (read-only mount).
- Context always ≥ 200 000: `--ctx-size 200000` → `n_ctx = 200192` verified in the logs.
- A single LLM server at a time (10 GB of VRAM), ports published **only on
  `127.0.0.1`**.
- Always `bin/dk` instead of `docker` (it adds the docker group to the process).
- No `sudo`, no changes to governor/swap/zram/driver/BIOS. Ideas that require
  them are only proposals in [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) §6.
- The original container `qwen36-35b` and image b11977 are never deleted;
  no `docker prune`, no pulling of the `:server-cuda` image
  (the pinned digest is used).

## Repository structure

```
.
├── README.md
├── LICENSE
├── docs/                  documentation (method, data, configuration)
│   ├── METHODOLOGY.md     operating plan used by the executing agent
│   ├── HARDWARE.md        detected hardware (CPU, PCIe, RAM, GPU, model)
│   ├── BASELINE.md        original configuration and measured baseline
│   ├── EXPERIMENTS.md     chronological log, one entry per experiment
│   ├── RESULTS.md         consolidated tables and comparisons
│   ├── OPTIMIZATION.md    final configuration, limits, rollback, clients
│   └── PROGRESS.md        work status and checklist
├── bench/                 benchmarks (standard library only)
│   ├── bench.py           PP / TG / TTFT / VRAM at multiple depths
│   ├── agentic.py         tool calling, multi-turn, coding, task/min
│   ├── session.py         long session with prompt reuse
│   ├── needle.py          5 needles at ~150K tokens
│   ├── summarize.py       medians from JSONL → table + summary.csv
│   └── corpus/            corpus used by the tests
├── scripts/
│   ├── exp.sh             full experiment: server → bench → stop → log
│   ├── run_server.sh      isolated experimental server on 127.0.0.1:8081
│   ├── queue.sh           sequential execution of several experiments
│   ├── prod_start.sh      starts production qwen36-35b-opt on 127.0.0.1:8080
│   ├── prod_stop.sh       stops production (container kept)
│   └── check_repo.py      data and link validation (used by CI)
├── results/               raw JSONL data + summary.csv (schema in results/README.md)
├── logs/                  raw server logs (evidence; see logs/README.md)
├── baseline/original/     inspect JSON of the original container and image
├── analysis/              GGUF metadata and inspection scripts
└── bin/dk                 docker wrapper (newgrp docker)
```

## Quick start

Prerequisites: Linux, Docker, NVIDIA Container Toolkit, a GPU with enough VRAM
(here 10 GB) and the GGUF file downloaded **by you** (it is not included in the repo).

```bash
git clone https://github.com/Korvaxoide/llama-cpp-moe-lab.git
cd llama-cpp-moe-lab
mkdir -p ~/AI/llama-models && cp /percorso/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf ~/AI/llama-models/
```

Then, from the root:

```bash
# isolated experimental server on 127.0.0.1:8081 with the final configuration
scripts/run_server.sh llopt-exp -- \
  --ctx-size 200000 --n-predict -1 --n-gpu-layers auto \
  --n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe \
  --spec-draft-n-ubatch 512 \
  -ctk q8_0 -ctv q8_0 -b 3072 -ub 3072 --load-mode none

# 3 ripetizioni fino a 32K, 2 oltre
python3 bench/bench.py --tag MIO_RUN --depths 8192,32768,131072,196608 --reps 3 --long-reps 2

# mediane a schermo + results/summary.csv
python3 bench/summarize.py MIO_RUN

bin/dk stop -t 60 llopt-exp && bin/dk rm llopt-exp
```

Alternatively `scripts/exp.sh MIO_RUN 8192,32768,131072,196608 3 2 -- <flags>`
does everything in a single command. Long benchmarks must be started in the background:

```bash
setsid nohup bash -c 'scripts/exp.sh MIO_RUN 8192,32768,131072,196608 3 2 -- <flags>' \
  > logs/MIO_RUN.out 2>&1 < /dev/null &
tail -3 logs/MIO_RUN.out
```

## Production

```bash
scripts/prod_start.sh     # qwen36-35b-opt su http://127.0.0.1:8080
scripts/prod_stop.sh      # stops it, container kept
```

Rollback to the original: `scripts/prod_stop.sh && bin/dk start qwen36-35b`
(then `bin/dk stop qwen36-35b && scripts/prod_start.sh` to return
to the optimized one). The rollback was tried in Phase 9. Never start the two
containers together: the VRAM is not enough.

Client (OpenCode, Cline, …): endpoint `http://127.0.0.1:8080/v1`,
`context = 200000`, output ≥ 8192. Details in [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) §8.
If the machine is also used for gaming, see §9 (coexistence with desktop and gaming).

## Data and reproducibility

No number in this repository is invented. Every table derives from
`results/*.jsonl` or from lines of `logs/exp/*`; the medians are generated with
`python3 bench/summarize.py`, which also rewrites `results/summary.csv`. Where a
value is a derived calculation (e.g. PCIe bandwidth) the formula is shown next to it.
The schema of the data files is in [`results/README.md`](results/README.md); what
the logs contain is in [`logs/README.md`](logs/README.md).

On every push the CI checks that all JSONL files are valid, that the Python scripts
compile and that the relative links in the documents exist
(`python3 scripts/check_repo.py`).

## Limitations

- The results apply to **this** hardware and **this** runtime (RTX 3080 10 GB,
  Ryzen 7 3700X, PCIe Gen3 x16, llama.cpp b11977, that GGUF): they are a method and a
  comparison, not a guarantee on other machines.
- ubatch above 3072 and `--n-cpu-moe 36` were not adopted because of insufficient
  VRAM margin, not because they are slow.
- The needle test was not repeated on F1r: compared to F1 only the
  ubatch and the MTP buffer change, not the KV precision or the weights.
- The "task/min" depends on the benchmark tasks; it is not an absolute index.

## License and disclaimer

Code and documentation: [MIT](LICENSE).

The model weights are **not** distributed in this repo: the GGUF stays out of
version control and is mounted read-only. Personal paths were replaced
with `$HOME` ahead of publication; the logs are reported as produced by the
tools, with only the path substitutions.

The original Italian documentation of the first publication is preserved at tag
[`v1.0.0`](https://github.com/Korvaxoide/llama-cpp-moe-lab/tree/v1.0.0).
