# results/ — raw data

One JSONL file per experiment (one JSON line per measurement), plus the consolidated
CSV. Every number cited in the documents comes from here: no value is
entered by hand.

## Naming convention

| Pattern | Produced by | Content |
|---|---|---|
| `bench_<TAG>.jsonl` | `bench/bench.py` | prompt processing, generation, TTFT, VRAM/RAM at multiple depths |
| `agentic_<TAG>.jsonl` | `bench/agentic.py` | tool calling, multi-turn, coding, task/min |
| `session_<TAG>.jsonl` | `bench/session.py` | turns of a long session with prompt reuse |
| `needle_<TAG>.jsonl` | `bench/needle.py` | retrieval of 5 needles in ~150K tokens |
| `summary.csv` | `bench/summarize.py` | medians/min/max of **all** `bench_*.jsonl` |

`<TAG>` is the experiment name used also in `docs/EXPERIMENTS.md`
(e.g. `F1r_ub3072`).

## Schema — `bench_<TAG>.jsonl`

| Field | Unit | Meaning |
|---|---|---|
| `tag` | — | experiment name |
| `depth` | token | prompt tokens sent (8 192 … 196 608) |
| `rep` | — | index of the repetition at that depth |
| `gen` | token | maximum tokens requested (`max_tokens`) |
| `wall_s` | s | total duration of the request |
| `prompt_n` | token | tokens actually processed (0 if cached) |
| `cache_n` | token | tokens reused from the server cache |
| `prompt_ms` | ms | duration of the prompt processing |
| `pp_tps` | token/s | `prompt_n / prompt_ms` |
| `predicted_n` | token | generated tokens |
| `tg_tps` | token/s | generation speed |
| `draft_n` / `draft_acc` / `acc_rate` | token, token, 0–1 | speculative decoding MTP: proposed, accepted, ratio |
| `n_ctx` | token | server context (must be `200192`) |
| `vram_max` | MiB | peak VRAM used, sampled with `nvidia-smi` every second |
| `util_avg` | % | average GPU utilization during the request |
| `pcie_gen_max` | — | observed PCIe generation (3 = Gen3) |
| `power_avg` | W | average GPU power |
| `ram_avail_min` | MiB | minimum `MemAvailable` during the request |
| `swap_used_max` | MiB | peak swap used |
| `ts` | ISO 8601 | start time of the measurement |

## Schema — `agentic_<TAG>.jsonl`

One line per test, plus a `SUMMARY` line.

| Field | Meaning |
|---|---|
| `test` | `tool_call`, `tool_call_stream`, `multi_turn`, `code_<nome>`, `SUMMARY` |
| `ok` | test outcome (for coding: the produced code passes the unit tests) |
| `finish` | `stop` / `tool_calls` / `length` reported by the server |
| `wall`, `ttft` | total duration and time to first token (s) |
| `gen_tokens`, `tg_tps`, `draft_acc` | generated tokens, speed, MTP acceptance |
| `has_reasoning`, `tool_calls`, `usage` | response payload for the tool calling tests |
| `coding_pass`, `coding_tokens`, `tasks_per_min`, `coding_wall_s` | only on `SUMMARY` |

## Schema — `session_<TAG>.jsonl`

| Field | Meaning |
|---|---|
| `turn` | turn number (1…29) |
| `ctx_tokens` | context tokens after the turn |
| `cache_n`, `prompt_n` | tokens reused from the cache and tokens actually processed |
| `pp_tps`, `ttft`, `tg_tps` | performance of the turn |
| `gen`, `finish`, `wall` | generated tokens, stop reason, duration |
| `answer` | model reply (truncated) |

`prompt_n` stable across turns is the evidence that the context is not reprocessed
from scratch on every request.

## Schema — `needle_<TAG>.jsonl`

| Field | Meaning |
|---|---|
| `expected` | the 5 secret values inserted in the text |
| `found` | how many and which ones were returned exactly |
| `score` | needles found out of 5 |
| `pp_tps`, `tg_tps`, `prompt_n` | performance of the experiment |

## Schema — `summary.csv`

`tag, depth, n, pp_tps, tg_tps, tg_min, tg_max, acc, ttft_s, vram_max, ram_avail_min, n_ctx`

It contains the medians per `(tag, depth)`; `n` is the number of repetitions. It is
regenerated **in full** on every invocation of `summarize.py`, even when only the
table of a single tag is requested.
