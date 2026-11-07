# BASELINE.md — original configuration and measured baseline

Sources: `baseline/original/qwen36-35b.inspect.json`, `baseline/original/image.inspect.json`,
`logs/exp/E00b_baseline.log`, `results/bench_E00b_baseline.jsonl`,
`results/agentic_E00_agentic.jsonl`, `results/session_E00_session.jsonl`,
`logs/original_full_run.log`, `logs/E00_baseline_server.log`.

## 1. Original configuration (container `qwen36-35b`)

| Item | Value |
|---|---|
| Container name | `qwen36-35b` (preserved, never deleted) |
| Image | `ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276` |
| Version / commit | b11977 / `7ac5727d6037ec99c31a28c6520f45de154199d9` |
| Mount | `$HOME/AI/llama-models` → `/models` (ro) |
| Port | `127.0.0.1:8080` → `8080/tcp` (loopback only) |
| Restart policy | `no` |
| GPU | `--gpus all` (DeviceRequests capabilities `gpu`) |
| Entrypoint | `/app/llama-server` |

Exact arguments (`Config.Cmd`):

```
-m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024
--flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2
--spec-draft-cpu-moe
```

Implicit values read from the log (`logs/exp/E00b_baseline.log`):

| Parameter | Value |
|---|---|
| `n_ctx` | **200192** (requested 200000) |
| `n_batch` / `n_ubatch` | 2048 / 512 |
| `n_threads` / `n_threads_batch` | 8 / 8 (on 16 threads) |
| `load_mode` | `mmap` |
| KV target / draft MTP | f16 / f16 |
| `n_slots` | 1, `n_ctx_slot = 200192` |
| prompt cache | enabled, limit 8192 MiB |
| context checkpoints | 32, minimum spacing 8192 |
| CUDA ARCHS | 500,610,700,750,800,860,890,900,1200; `FA_QUANTS` include `q8_0-q8_0` |

## 2. VRAM breakdown from the log (E00b, startup)

| Buffer | MiB |
|---|---|
| `CUDA0 model buffer size` (non-experts on GPU) | 1961.70 |
| `CUDA0 KV buffer size` (target, f16, 200K) | 3910.00 |
| `CUDA0 KV buffer size` (MTP draft context) | 391.00 |
| `CUDA0 RS buffer size` (Gated DeltaNet recurrent state) | 188.44 |
| `CUDA0 MoE cache size` (for 13942 MiB of host experts) | 1019.49 |
| `CUDA0 compute buffer size` (main context) | 565.50 |
| `CUDA0 compute buffer size` (MTP contexts, 2 entries) | 235.52 + 235.52 |
| `CUDA_Host compute buffer size` | 211.52 / 203.52 / 203.52 |
| Sum of declared GPU buffers | ≈ 8507 MiB |
| `vram_max` observed during the baseline (includes ~0.9 GB desktop) | 9306 MiB (8K) … 9470 MiB (192K) |

The experts (13 942 MiB) stay in host RAM (`CPU_Mapped`, `--cpu-moe`); the MoE cache
keeps a partial copy of them in VRAM.

## 3. FULL baseline `E00b_baseline`

Command (reproducible):

```bash
setsid nohup bash -c 'scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- \
  --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 \
  --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-cpu-moe' > logs/E00b.out 2>&1 < /dev/null &
```

Output of `python3 bench/summarize.py E00b_baseline` (source `results/bench_E00b_baseline.jsonl`):

| tag | depth | n | PP t/s | TG t/s (min-max) | MTP acc | TTFT s | VRAM max MiB | RAM avail min | n_ctx |
|---|---|---|---|---|---|---|---|---|---|
| E00b_baseline | 8192 | 3 | 445.25 | 31.41 (31.32-32.98) | 0.86 | 18.4 | 9306 | 25690 | 200192 |
| E00b_baseline | 16384 | 3 | 451.23 | 29.34 (27.58-30.46) | 0.91 | 36.31 | 9298 | 24186 | 200192 |
| E00b_baseline | 32768 | 3 | 443.2 | 28.54 (28.25-29.24) | 0.97 | 73.94 | 9294 | 21797 | 200192 |
| E00b_baseline | 65536 | 2 | 424.63 | 28.48 (28.39-28.56) | 0.8 | 154.33 | 9285 | 19394 | 200192 |
| E00b_baseline | 131072 | 2 | 398.85 | 28.15 (27.87-28.43) | 0.96 | 328.63 | 9444 | 18631 | 200192 |
| E00b_baseline | 196608 | 2 | 385.84 | 28.9 (28.35-29.44) | 0.85 | 509.6 | 9470 | 19779 | 200192 |

Checks: `n_ctx = 200192` ✔, `prompt_n = depth` and no `warning` in all 15 rows ✔,
no `CUDA error` / `out of memory` / `GGML_ABORT` ✔.

The partial baseline `E00_baseline` (8K–64K, 1–2 rep, with `-lv 4`) is consistent within
noise (%: 8K PP 438.73 vs 445.25, TG 29.88 vs 31.41) — source `results/bench_E00_baseline.jsonl`.

## 4. Agentic baseline `E00_agentic`

Server with the same arguments + `-lv 4`; `python3 bench/agentic.py --tag E00_agentic --coding-reps 2`.
Source: `results/agentic_E00_agentic.jsonl`.

| Test | Outcome |
|---|---|
| tool_call (non-stream) | **ok** (finish `tool_calls`, valid JSON arguments) |
| tool_call_stream (SSE) | **ok** |
| multi_turn (tool result re-fed) | **ok** (reply cites CRC32 / 8 hex) |
| coding (5 problems × 2 rep) | **10/10** |
| tasks_per_min (correct code) | **6.7** |
| Median TG on the coding tasks | ≈ 32.1 t/s |
| MTP acceptance on the coding tasks | 0.82–0.97 |

MoE cache statistics at shutdown (`logs/exp/E00_agentic_server.log`):
`ubatch <= 8: hits = 73368, misses = 655786, hit rate = 10.06%`.

## 5. Long session `E00_session`

`python3 bench/session.py --tag E00_session --target 180000 --chunk 6000`, 29 turns,
558.5 s total. Source: `results/session_E00_session.jsonl`.

| Metric | Value |
|---|---|
| Turns | 29 |
| Initial → final context | 6 612 → 178 647 token |
| `prompt_n` per turn (min/median/max) | 6082 / **6085** / 6548 |
| `cache_n` reused (last turn) | 172 469 |
| TTFT (min/median/max) | 14.62 / **16.86** / 18.79 s |
| Median PP | 364.6 t/s |
| Median TG | 27.59 t/s (first 3 turns 27.1–29.4, last 3 25.1–27.5) |
| Median MTP acceptance | 0.833 |
| Total generated tokens | 2 012 |

**Critical check passed**: `prompt_n` stays ≈ 6085 (the chunk of 6000 + a few hundred
prompt tokens) for all 29 turns; no turn reprocessed the whole context.
The prompt cache works on the hybrid model too.

MoE cache statistics at shutdown (`logs/exp/E00_session_server.log`):
`ubatch <= 8: hits = 73123, misses = 479655, hit rate = 13.23%, uploaded = 627994.61 MiB`.

## 6. Original real session (`logs/original_full_run.log`)

Real agentic session (OpenCode) up to ~155K tokens, interrupted by the restart of the Docker
daemon (exit 255), not by a model crash. 158 requests with timing.

| Metric | Measured value |
|---|---|
| Overall median TG | 27.44 t/s |
| Median TG of the last 20 requests (context ~150K) | **26.85 t/s** |
| Incremental prompts > 2000 tokens (30 requests) | **347.9 t/s** median, 2544 median tokens |
| Implicit TTFT per ~2.4K tokens | ≈ 2.4K / 348 ≈ **7.0 s** |

## 7. The two original warnings

1. **`tensor overrides to CPU are used with mmap enabled - consider using --load-mode none`**
   With `--cpu-moe` the experts are forced onto CPU. With mmap the loader maps them into
   pageable memory (not into a pinned `CUDA_Host` buffer), so every H2D transfer is a DMA
   from a non-pinned page. Hypothesis H3 (tested in Phase 4) is that `--load-mode none` puts
   them in pinned `CUDA_Host` and speeds up PP and the MoE cache misses. Cost: ~14 GB of pinned RAM.

2. **`common_fit_params: failed to fit params to free device memory: model_params::tensor_buft_overrides already set by user, abort`**
   `--fit` (auto-placement) gives up because the user has already fixed the expert buffers
   with `--cpu-moe`: placement is manual. It is **harmless**: the context stays 200192 and
   `n_ctx` is not reduced. It only implies that `--n-gpu-layers auto` cannot recalibrate.
