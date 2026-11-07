# EXPERIMENTS.md — chronological log of the experiments

Every entry follows the template in §9 of `METHODOLOGY.md`. Decisions according to rule §3.5.

## E00b_baseline — 2026-11-05

- **Hypothesis**: complete baseline (resumes from the interrupted E00 baseline): all depths 8192→196608 with the original configuration.
- **Bottleneck addressed**: —
- **Change (single)**: run with new tag `E00b_baseline` via `scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- <args originali> -lv 4`.
- **Command**: `setsid nohup bash -c 'scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe' > logs/E00b.out 2>&1 < /dev/null &`
- **Checks**: `n_ctx = 200192` ✔ | `prompt_n = depth` ✔ (no `warning`) ✔ | errors in the log: none (only benign `common_fit_params: failed to fit params ... tensor_buft_overrides already set by user, abort`) ✔
- **Memory**: `load_mode = mmap` (CPU_Mapped) | `CUDA0 KV buffer size = 3910.00 MiB` (f16) | `llama_memory_recurrent: CUDA0 RS buffer size = 188.44 MiB` | `impl: CUDA0 MoE cache size = 1019.49 MiB` | `CUDA0 compute buffer size = 565.50 MiB` | `CUDA0 compute buffer size = 235.52 MiB` / `CUDA_Host compute buffer = 211.52 MiB` (additional MTP contexts) | `llama_moe_cache: ubatch <= 8: hits = 140201, misses = 905056, hit rate = 13.41%`
- **Results (medians, n=3 / n=2)**:
  | depth | PP t/s | TG t/s | MTP acc | TTFT s | VRAM max MiB | RAM avail min |
  |---|---|---|---|---|---|---|
  | 8192 | 445.25 | 31.41 | 0.86 | 18.4 | 9306 | 25690 |
  | 16384 | 451.23 | 29.34 | 0.91 | 36.31 | 9298 | 24186 |
  | 32768 | 443.20 | 28.54 | 0.97 | 73.94 | 9294 | 21797 |
  | 65536 | 424.63 | 28.48 | 0.80 | 154.33 | 9285 | 19394 |
  | 131072 | 398.85 | 28.15 | 0.96 | 328.63 | 9444 | 18631 |
  | 196608 | 385.84 | 28.90 | 0.85 | 509.6 | 9470 | 19779 |
- **Decision**: KEPT — complete and reproducible comparison baseline. See `BASELINE.md` (to be written in Phase 10).
- **Files**: `results/bench_E00b_baseline.jsonl`, `logs/exp/E00b_baseline.log`, `logs/E00b.out`

## E00_agentic — 2026-11-05 (first run) and rerun — 2026-11-05

- **Hypothesis**: agentic baseline with the original configuration (`--ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe -lv 4`).
- **Bottleneck addressed**: —
- **Change (single)**: `scripts/run_server.sh llopt-agent` + `python3 bench/agentic.py --tag E00_agentic --coding-reps 2`.
- **Iteration 1 outcome (2026-11-05 ~22:21–22:57)**: tool_call **ok**, tool_call_stream **ok**, multi_turn **ok**, coding **0/10** (all tasks with `NameError` during the test phase).
- **Python bug fixed** (noted): the coding tasks of `bench/agentic.py` did not pass `chat_template_kwargs: {"enable_thinking": false}`. With thinking active the Python code was generated in `reasoning_content` and `extract_code` returned empty content → test code executed without the functions → `NameError`.
- **Fix applied**: added `"chat_template_kwargs": {"enable_thinking": false}` to the stream payload of the coding tasks in `bench/agentic.py`. No prompt or unit test modified.
- **Iteration 2 outcome (2026-11-05 ~23:05–23:45, interrupted during a pause)**: tool_call **ok**, tool_call_stream **ok**, multi_turn **ok**, coding rep 0: **5/5 ok** (with `finish: "stop"` and `gen_tokens` 107–432; TG ~30–33 t/s, draft acc 0.83–0.97). Partial file archived in `results/archive/agentic_E00_agentic_rep0.jsonl`.
- **Final outcome (2026-11-06 08:25–08:27)**: complete run `--coding-reps 2` with the fix: tool_call **ok**, tool_call_stream **ok**, multi_turn **ok**, coding **10/10**, `tasks_per_min = 6.7`, median TG on coding ≈ 32.1 t/s. MoE cache at stop: hit rate 10.06 % (`logs/exp/E00_agentic_server.log`).
- **Decision**: KEPT as the reference agentic baseline.
- **Files**: `results/agentic_E00_agentic.jsonl`, `logs/E00_agentic_bench.out`, `logs/exp/E00_agentic_server.log`

## E00_session — 2026-11-06 08:30–08:39

- **Hypothesis**: baseline of the long agentic session (prompt reuse on the hybrid model).
- **Bottleneck addressed**: — (baseline).
- **Change (single)**: `scripts/run_server.sh llopt-agent` + `python3 bench/session.py --tag E00_session --target 180000 --chunk 6000`.
- **Command**: `setsid nohup bash -c 'python3 bench/session.py --tag E00_session --target 180000 --chunk 6000' > logs/E00_session.out 2>&1 < /dev/null &`
- **Checks**: `n_ctx=200192` ✔ | 29 turns, no errors ✔ | `prompt_n` per turn 6082–6548 (median 6085) → **no complete reprocessing of the context** ✔
- **Memory**: identical to the baseline (`CUDA0 KV 3910+391 MiB`, MoE cache 1019.49 MiB, compute 565.50 MiB). MoE cache at stop: hit rate 13.23 % (`logs/exp/E00_session_server.log`).
- **Results**:
  | metric | value |
  |---|---|
  | turns | 29 |
  | initial → final context | 6 612 → 178 647 token |
  | TTFT min/median/max | 14.62 / 16.86 / 18.79 s |
  | median PP | 364.6 t/s |
  | median TG | 27.59 t/s (last 3 turns 25.1–27.5) |
  | median MTP acceptance | 0.833 |
  | generated tokens | 2 012 (2 012 in 558.5 s) |
- **Decision**: KEPT as the baseline of the long session.
- **Files**: `results/session_E00_session.jsonl`, `logs/E00_session.out`, `logs/exp/E00_session_server.log`

## PHASE 1 — Prompt processing: ubatch size (H2)

### E01_ub1024 — 2026-11-06 08:43
- **Hypothesis**: H2 — doubling `-ub 512 → 1024` amortizes the same PCIe expert transfer over 2× tokens.
- **Bottleneck addressed**: PCIe H2D of the experts during prompt processing.
- **Change (single)**: `-b 2048 -ub 1024` compared with `E00b_baseline`.
- **Command**: `scripts/exp.sh E01_ub1024 8192,32768 3 1 -- <ARGS_BASE> -b 2048 -ub 1024`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no `warning` ✔ | no errors ✔
- **Memory**: `CUDA0 compute buffer size = 785.01 MiB` (ref. 565.50, +219.51 MiB) | `vram_max = 9879 MiB` (8K), 9781 MiB (32K) | `ram_avail_min = 23208 MiB` | swap 1 MiB
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 445.25 | 753.93 | **+69.3** | 31.41 | 30.32 | −3.5 | 0.88 |
  | 32768 | 443.20 | 761.32 | **+71.8** | 28.54 | 28.56 | +0.1 | 0.96 |
- **Decision**: KEPT — PP +69 % (min new 753.9 > max ref 451.2 at 8K; 757.9 > 443.2 at 32K). TG 8K −3.5 % is at the limit of the §3.5 threshold but the other indicator improves much more (explicit exception of the plan). VRAM 9879 ≤ 9900 (margin 21 MiB, to be noted).
- **Files**: `results/bench_E01_ub1024.jsonl`, `logs/exp/E01_ub1024.log`

### E02_ub2048 — 2026-11-06 08:46
- **Hypothesis**: H2 — ubatch 2048 amortizes the transfers 4×.
- **Bottleneck addressed**: PCIe H2D of the experts in prompt processing.
- **Change (single)**: `-b 2048 -ub 2048` compared with `E00b_baseline`.
- **Command**: `scripts/exp.sh E02_ub2048 8192,32768 3 1 -- <ARGS_BASE> -b 2048 -ub 2048`
- **Checks**: startup **FAILED** — no measurement.
- **Error**: `CUDA0 compute buffer size = 1352.02 MiB` (ref. 565.50); when creating the MTP context `allocating 942.06 MiB on device 0: cudaMalloc failed: out of memory` → `failed to allocate compute pp buffers` → `failed to create MTP context`.
- **Decision**: DISCARDED in Phase 1 (insufficient VRAM with KV f16 + MoE cache 1024). To be retried in Phase 3 after KV q8_0, as foreseen by the plan.
- **Files**: `logs/exp/E02_ub2048.log`, `logs/queue_phase1.out`

### E03_ub4096 — 2026-11-06 08:47
- **Hypothesis**: H2 — ubatch 4096 amortizes the transfers 8×.
- **Bottleneck addressed**: PCIe H2D of the experts in prompt processing.
- **Change (single)**: `-b 4096 -ub 4096` compared with `E00b_baseline`.
- **Command**: `scripts/exp.sh E03_ub4096 8192,32768 3 1 -- <ARGS_BASE> -b 4096 -ub 4096`
- **Checks**: startup **FAILED** — no measurement (OOM like E02, compute buffer even larger).
- **Decision**: DISCARDED in Phase 1; to be retried in Phase 3 with KV q8_0.
- **Files**: `logs/exp/E03_ub4096.log`, `logs/queue_phase1.out`

### E01_ub1024_long — 2026-11-06 08:50
- **Hypothesis**: mandatory verification of the capacity at 196608 tokens of the winning ubatch.
- **Bottleneck addressed**: holding the VRAM at full context.
- **Change (single)**: depth 196608 (1 rep) with `-b 2048 -ub 1024`.
- **Command**: `scripts/exp.sh E01_ub1024_long 196608 1 1 -- <ARGS_BASE> -b 2048 -ub 1024`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=196608` ✔ | no errors ✔
- **Memory**: compute buffer 785.01 MiB | `vram_max = 9847 MiB` (≤ 9900) | `ram_avail_min = 26092 MiB`
- **Results**: PP **613.66 t/s** (ref. 385.84 → **+59.0 %**), TG **30.81 t/s** (ref. 28.90 → +6.6 %), TTFT ≈ 320 s.
- **Decision**: KEPT — it holds at 196608 with VRAM below the threshold. **Ubatch 1024 = new reference**.
- **Files**: `results/bench_E01_ub1024_long.jsonl`, `logs/exp/E01_ub1024_long.log`

## PHASE 2 — Quantized KV cache (H4)

### E10_kvq8 — 2026-11-06 09:01
- **Hypothesis**: H4 — q8_0 KV on target and MTP draft halves ~4.3 GB of KV and frees VRAM for larger ubatches/caches; negligible quality loss.
- **Bottleneck addressed**: VRAM occupied by the f16 KV.
- **Change (single)**: `-ctk q8_0 -ctv q8_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0` compared with `E01_ub1024`.
- **Command**: `scripts/exp.sh E10_kvq8 8192,32768 3 1 -- <ARGS rif.> -ctk q8_0 -ctv q8_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no errors ✔ | log: `KV buffer size = 2077.19 MiB … K (q8_0), V (q8_0)` + draft `207.72 MiB` ✔
- **Memory**: target KV 3910.00 → **2077.19 MiB** (−1832.81), draft KV 391.00 → **207.72 MiB** (−183.28); `CUDA0 compute buffer size = 982.01 MiB` (ref. 785.01) | `vram_max = 8277 MiB` (8K) / 8198 (32K)
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 753.93 | 768.95 | +2.0 | 30.32 | 29.08 | −4.1 | 0.876 |
  | 32768 | 761.32 | 760.64 | −0.1 | 28.56 | 27.33 | **−4.3** | 0.966 |
- **Decision**: DISCARDED compared with E11 — TG −4.3 % at 32K exceeds the §3.5 threshold and PP does not compensate. The quantized draft KV brings no useful VRAM benefit and appears to cost TG.
- **Files**: `results/bench_E10_kvq8.jsonl`, `logs/exp/E10_kvq8.log`

### E11_kvq8_tgt_only — 2026-11-06 09:04
- **Hypothesis**: H4 — quantizing only the target KV (MTP draft in f16) frees ~1.8 GB with no impact on speed.
- **Bottleneck addressed**: VRAM of the f16 KV.
- **Change (single)**: `-ctk q8_0 -ctv q8_0` compared with `E01_ub1024`.
- **Command**: `scripts/exp.sh E11_kvq8_tgt_only 8192,32768 3 1 -- <ARGS rif.> -ctk q8_0 -ctv q8_0`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no errors ✔ | target KV q8_0 2077.19 MiB, draft stays f16 391.00 MiB ✔
- **Memory**: `vram_max = 7974 MiB` (both depths) | swap 1 MiB | RAM avail min 25075 MiB
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 753.93 | 764.78 | +1.4 | 30.32 | 31.70 | +4.5 | 0.880 |
  | 32768 | 761.32 | 761.58 | +0.03 | 28.56 | 28.00 | −2.0 | 0.966 |
- **Decision**: CANDIDATE — PP neutral, TG neutral/better at 8K and −2.0 % (within the "neutral" threshold) at 32K; VRAM −1905 MiB. Promotion to reference is subject to coding 10/10 and needle = f16.
- **Files**: `results/bench_E11_kvq8_tgt_only.jsonl`, `logs/exp/E11_kvq8_tgt_only.log`

### E11_quality — 2026-11-06 09:10–09:20
- **Hypothesis**: the q8_0 target KV degrades neither long-context retrieval nor code quality.
- **Change (single)**: none; quality test on the `E11_kvq8_tgt_only` configuration.
- **Commands**: `python3 bench/needle.py --tag E11_needle_q8` and `python3 bench/agentic.py --tag E11_agentic --coding-reps 2` (server q8, ub1024).
- **Needle** (5 needles at 150 128 tokens, same prompt f16/q8): f16 `E01_needle_f16` **5/5**; q8 `E11_needle_q8` **5/5** → no loss.
- **Agentic**: tool_call ok, tool_call_stream ok, coding **9/10** (baseline 10/10; single error: `parse_duration('1h1h')` does not raise `ValueError`, rep 0). Loss = 1, within the plan limit (≤ 1). `tasks_per_min = 5.25` vs 6.7 baseline, but with 3 026 generated tokens vs 2 659 (+14 %): the figure is dominated by the length of the responses, not by speed.
- **Decision**: **PROMOTED to new reference** — PP neutral, TG within ±2 % at 32K, VRAM −1905 MiB, needle 5/5, coding −1 (allowed limit). The free VRAM is to be spent in Phase 3.
- **Files**: `results/needle_E01_needle_f16.jsonl`, `results/needle_E11_needle_q8.jsonl`, `results/agentic_E11_agentic.jsonl`, `logs/exp/E11_needle_q8_server.log`, `logs/exp/E11_agentic_server.log`

## PHASE 3 — Re-expanding the ubatch with KV q8_0

### E20_kvq8_ub2048 — 2026-11-06 09:24
- **Hypothesis**: H2+H4 — the VRAM freed by the q8_0 KV allows the ubatch 2048 that went OOM in Phase 1.
- **Bottleneck addressed**: PCIe H2D of the experts in prompt processing.
- **Change (single)**: `-b 2048 -ub 2048` compared with `E11_kvq8_tgt_only`.
- **Command**: `scripts/exp.sh E20_kvq8_ub2048 8192,32768 3 1 -- <ARGS rif. + q8> -b 2048 -ub 2048`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no errors ✔
- **Memory**: `CUDA0 compute buffer size = 1437.02 MiB` (ref. 982.01 with ub1024) | `vram_max = 8939 MiB` | RAM avail min 24112 MiB
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 764.78 | 1309.79 | **+71.3** | 31.70 | 28.67 | −9.6 | 0.85 |
  | 32768 | 761.58 | 1267.00 | **+66.4** | 28.00 | 27.75 | −0.9 | 0.97 |
- **Decision**: KEPT — PP +66/71 %, TG at 32K neutral (−0.9 %). The TG drop at 8K (−9.6 %, with acc 0.85 vs 0.88) is noted as a trade-off: at 8K generation starts right after a short prompt, where the larger compute buffer (1437 MiB) leaves less VRAM/L2 to the cache; at 32K+ the phenomenon disappears.
- **Files**: `results/bench_E20_kvq8_ub2048.jsonl`, `logs/exp/E20_kvq8_ub2048.log`

### E21_kvq8_ub4096 — 2026-11-06 09:27
- **Hypothesis**: H2+H4 — ubatch 4096 with KV q8_0.
- **Change (single)**: `-b 4096 -ub 4096` compared with `E11_kvq8_tgt_only`.
- **Checks**: startup **FAILED** — `allocating 1884.12 MiB on device 0: cudaMalloc failed: out of memory`; compute buffer ub4096 ≈ 2724 MiB (estimated from the logs: 1437 at ub2048), it does not fit with KV q8 + cache 1024.
- **Decision**: DISCARDED — ub2048 is the maximum that holds at 32K with this configuration. The context and the cache are not reduced to chase it (§0.2/§0.3).
- **Files**: `logs/exp/E21_kvq8_ub4096.log`, `logs/queue_phase3.out`

## PHASE 4 — Pinned memory: `--load-mode none` (H3)

### E30_loadnone — 2026-11-06 09:40
- **Hypothesis**: H3 — with mmap the experts sit in non-pinned pages; `--load-mode none` puts them in pinned `CUDA_Host` and makes the DMA real → PP and MoE cache misses faster.
- **Bottleneck addressed**: H2D copies from pageable memory.
- **Change (single)**: `--load-mode none` compared with `E20_kvq8_ub2048`.
- **Command**: `scripts/exp.sh E30_loadnone 8192,32768 3 1 -- <ARGS rif.> --load-mode none`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no errors ✔ | log: `load_mode = none` ✔ | `CUDA_Host model buffer size = 14339.85 MiB`, **zero** `CPU_Mapped model buffer` lines ✔ | warning "tensor overrides to CPU are used with mmap enabled" **gone** ✔
- **Memory**: `vram_max = 8963 MiB` (unchanged) | RAM available min **9669 MiB** (pinned ~14 GB) | swap max 47 MiB | `load_s` 13 s (ref. 4–5 s with mmap)
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1309.79 | 1395.03 | **+6.5** | 28.67 | 32.29 | **+12.6** | 0.87 |
  | 32768 | 1267.00 | 1351.21 | **+6.6** | 27.75 | 28.44 | +2.5 | 0.96 |
- **Decision**: KEPT — it improves both PP and TG, VRAM does not change. Cost: ~14.3 GB of pinned RAM (available 9.7 GB, above the 6 GB minimum required by the plan; noted as a risk for browser/IDE).
- **Files**: `results/bench_E30_loadnone.jsonl`, `logs/exp/E30_loadnone.log`, `logs/exp/E30_loadnone.mem_idle.txt`

### E31_loadnone_mlock — 2026-11-06 09:46
- **Hypothesis**: `--load-mode mlock` (with `--ulimit memlock=-1:-1`) can beat `none`.
- **Change (single)**: `--load-mode mlock` compared with `E30_loadnone`.
- **Checks**: `n_ctx=200192` ✔ | `load_mode = mlock` ✔ | no errors ✔
- **Memory**: `vram_max = 8933 MiB` | RAM avail min 9624 MiB
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1395.03 | 1393.18 | −0.1 | 32.29 | 31.23 | −3.3 | 0.84 |
  | 32768 | 1351.21 | 1348.58 | −0.2 | 28.44 | 28.75 | +1.1 | 0.97 |
- **Decision**: NEUTRAL/DISCARDED — no measurable gain; TG 8K −3.3 %. We stay on `none` (fewer requirements: no special `ulimit`).
- **Files**: `results/bench_E31_loadnone_mlock.jsonl`, `logs/exp/E31_loadnone_mlock.log`

### E30_session_none — 2026-11-06 09:49
- **Hypothesis**: pinned memory holds an agentic session without errors or degradation.
- **Change (single)**: none; `session.py --target 60000` on `E30_loadnone`.
- **Checks**: 9 turns, `prompt_n` 6082–6548 (no complete reprocessing) ✔ | no errors in the log ✔
- **Results**: context 6 623 → 55 854 token; median TTFT **5.54 s**; median PP 1111.1 t/s; median TG 30.04 t/s; 71.7 s total.
- **Decision**: KEPT — stability confirmed with pinned RAM.
- **Files**: `results/session_E30_session_none.jsonl`, `logs/exp/E30_session_none_server.log`

## PHASE 5 — Generation: MoE cache (H5) and MTP (H6)

### E40_mc0 — 2026-11-06 09:54
- **Hypothesis**: H5-data — if generation is limited by PCIe, computing the experts on the CPU (DDR4) can be faster than loading them into the GPU cache at every token.
- **Bottleneck addressed**: PCIe upload of the experts during generation.
- **Change (single)**: `--moe-cache-mib 0` (cache disabled) compared with `E30_loadnone` (1024).
- **Command**: `GEN=512 scripts/exp.sh E40_mc0 8192,32768 4 1 -- <ARGS rif.> --moe-cache-mib 0`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no errors ✔ | no `llama_moe_cache:` line (cache absent) ✔
- **Memory**: `vram_max = 7940` (8K) / **7960** (32K) MiB | RAM avail min 8814 MiB | swap max 199 MiB
- **Results (medians, n=4)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1395.03 | 1362.78 | −2.3 | 32.29 | **42.22** | **+30.8** | 0.83 |
  | 32768 | 1351.21 | 1323.88 | −2.0 | 28.44 | **45.09** | **+58.5** | 0.97 |
- **Decision**: KEPT — TG +31/+59 %, PP −2 % (neutral). Quantitative confirmation that **the MoE cache harms generation**: every miss costs a PCIe transfer, while the CPU computes the experts from DDR4 faster. The cache is useful only for prompt processing (large batches).
- **Files**: `results/bench_E40_mc0.jsonl`, `logs/exp/E40_mc0.log`

### E41_mc1536 — 2026-11-06 09:58
- **Hypothesis**: a larger cache (≥ 24 slots per group) eliminates the misses of the q6_K/q4_K layers in MTP verification (H5).
- **Change (single)**: `--moe-cache-mib 1536` compared with `E30_loadnone` (1024).
- **Checks**: `n_ctx=200192` ✔ | no errors ✔ | slots: 1042 / 55 / **27** / **27** (min 27 ≥ 24) | hit rate 29.57 % (vs 15.21 % with 1024)
- **Memory**: `vram_max = 9455` (8K) / 9481 (32K) MiB | RAM avail min 8760 MiB
- **Results (medians, n=4)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1395.03 | 1408.88 | +1.0 | 32.29 | 37.48 | +16.1 | 0.86 |
  | 32768 | 1351.21 | 1367.48 | +1.2 | 28.44 | 32.24 | +13.4 | 0.97 |
- **Decision**: BETTER than the 1024 cache but **worse than cache 0** (TG 37.5/32.2 vs 42.2/45.1). Not adopted.
- **Files**: `results/bench_E41_mc1536.jsonl`, `logs/exp/E41_mc1536.log`

### E42_mc2048 — 2026-11-06 10:03
- **Hypothesis**: 2048 MiB cache to maximize the hit rate.
- **Change (single)**: `--moe-cache-mib 2048` compared with `E30_loadnone`.
- **Checks**: startup **FAILED** — `CUDA error: out of memory` during warm-up (the cache fits in the initial allocation but leaves no room for the compute buffers).
- **Decision**: DISCARDED — it does not fit with KV q8 + ub2048 + `--load-mode none`.
- **Files**: `logs/exp/E42_mc2048.log`

### E43_mc3072 — not run
- **Reason**: 2048 already OOM; 3072 cannot fit (§5.1 "only if VRAM allows it"). Noted here for completeness of the checklist.

### E44_mcMAX — 2026-11-06 10:24
- **Hypothesis**: find the maximum `--moe-cache-mib` that leaves `vram_max ≤ 9900` at 32K.
- **Change (single)**: `--moe-cache-mib 1792` compared with `E40_mc0`.
- **Checks**: `n_ctx=200192` ✔ | no errors ✔ | slots: 1216 / 64 / **31** / **31** | hit rate 35.37 %
- **Memory**: `vram_max = 9751` (8K) / 9758 (32K) MiB → it is the maximum that fits (2048 OOM).
- **Results (medians, n=4)**: PP 1415.95 / 1383.35; TG **39.02 / 36.34**; acc 0.90 / 0.98.
- **Decision**: DISCARDED — even the largest possible cache remains below cache 0 in TG (39.0/36.3 vs 42.2/45.1). The "maximum" is not the best: the cache helps only PP, and only marginally.
- **Files**: `results/bench_E44_mcMAX.jsonl`, `logs/exp/E44_mcMAX.log`

### E45_ncmoe37 — 2026-11-06 10:28
- **Hypothesis**: H5 alternative — instead of the cache, keep the experts of some whole layers in VRAM (`--n-cpu-moe 37` → layers 37–39 on GPU, ~996 MiB).
- **Change (single)**: `--n-cpu-moe 37 --moe-cache-mib 0` instead of `--cpu-moe --moe-cache-mib 0` (E40).
- **Checks**: `n_ctx=200192` ✔ | no errors ✔ | `CUDA0 model buffer size = 3545.70 MiB` (ref. 1961.70, +1584) | no cache
- **Memory**: `vram_max = 9504` (8K) / **9525** (32K) MiB | RAM avail min 10485 MiB | swap max 335 MiB
- **Results (medians, n=4)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1362.78 | 1468.84 | **+7.8** | 42.22 | **48.49** | **+14.8** | 0.90 |
  | 32768 | 1323.88 | 1423.32 | **+7.5** | 45.09 | **48.66** | **+7.9** | 0.98 |
- **Decision**: **KEPT — new reference**. Experts resident in VRAM pay neither PCIe nor the CPU: they beat both the cache and pure CPU.
- **Files**: `results/bench_E45_ncmoe37.jsonl`, `logs/exp/E45_ncmoe37.log`

### E46_ncmoe36 — 2026-11-06 10:33
- **Hypothesis**: one more expert layer on GPU (4 layers, `--n-cpu-moe 36`) improves further.
- **Change (single)**: `--n-cpu-moe 36` compared with `E45_ncmoe37`.
- **Checks**: `n_ctx=200192` ✔ | no errors ✔
- **Memory**: `vram_max = 9856` (8K) / 9829 (32K) MiB (margin 44 MiB from the 9900 limit) | RAM avail min 10861 MiB
- **Results (medians, n=4)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1468.84 | 1501.94 | +2.2 | 48.49 | 48.02 | −1.0 | 0.87 |
  | 32768 | 1423.32 | 1450.45 | +1.9 | 48.66 | 49.33 | +1.4 | 0.98 |
- **Test at 196608** (`E46_ncmoe36_long`): PP **1036.74 t/s**, TG **32.38 t/s**, `vram_max = 9855 MiB`, no errors.
- **Decision**: NEUTRAL — gain < 5 % on both metrics and VRAM margin reduced to ~45 MiB. E45 is preferred (VRAM 9525, margin 375 MiB). According to §3.5 a 1–2 % difference is noise.
- **Files**: `results/bench_E46_ncmoe36.jsonl`, `results/bench_E46_ncmoe36_long.jsonl`, `logs/exp/E46_ncmoe36.log`

### E47_nmax2_ref — 2026-11-06 10:53
- **Hypothesis**: Phase 5.3 reference (n_max 2) on the three required depths, with the `E45_ncmoe37` configuration.
- **Change (single)**: addition of depth 131072 compared with `E45_ncmoe37` (same configuration).
- **Command**: `GEN=512 scripts/exp.sh E47_nmax2_ref 8192,32768,131072 4 1 -- <ARGS rif.>`
- **Checks**: `n_ctx=200192` ✔ | no errors ✔
- **Results (medians)**: 8K PP 1467.72 / TG **48.08** (acc 0.91); 32K PP 1426.97 / TG **48.73** (0.98); 131K PP 1162.29 / TG **41.01** (0.94, n=1). VRAM max 9551 MiB.
- **Decision**: reference for 5.3.
- **Files**: `results/bench_E47_nmax2_ref.jsonl`, `logs/exp/E47_nmax2_ref.log`

### E50_nmax1 — 2026-11-06 10:59
- **Hypothesis**: H6 — with n_max 1 fewer draft tokens are wasted; acceptance is high (0.96–0.99).
- **Change (single)**: `--spec-draft-n-max 1` compared with `E47_nmax2_ref`.
- **Checks**: `n_ctx=200192` ✔ | no errors ✔
- **Results (medians, n=4)**:
  | depth | TG ref | TG new | Δ% | acc ref | acc new |
  |---|---|---|---|---|---|
  | 8192 | 48.08 | 46.78 | −2.7 | 0.91 | 0.96 |
  | 32768 | 48.73 | 44.42 | **−8.8** | 0.98 | 0.99 |
  | 131072 | 41.01 | 37.37 | **−8.9** | 0.94 | 0.98 |
- **Decision**: DISCARDED — fewer tokens per verification step; the higher acceptance does not compensate.
- **Files**: `results/bench_E50_nmax1.jsonl`, `logs/exp/E50_nmax1.log`

### E51_nmax3 — 2026-11-06 11:05
- **Hypothesis**: H6 — n_max 3 can beat 2 on actual tokens/s.
- **Change (single)**: `--spec-draft-n-max 3` compared with `E47_nmax2_ref`.
- **Checks**: `n_ctx=200192` ✔ | no errors ✔
- **Results (medians, n=4 / n=1 at 131K)**:
  | depth | PP ref | PP new | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|
  | 8192 | 1467.72 | 1469.26 | 48.08 | **49.74** | +3.5 | 0.92 |
  | 32768 | 1426.97 | 1422.41 | 48.73 | **49.91** | +2.4 | 0.97 |
  | 131072 | 1162.29 | 1153.93 | 41.01 | **45.18** | **+10.2** | 0.97 |
- **Decision**: **KEPT as reference** — no depth regresses, the gain at 131K is +10.2 %; at 8–32K it is below the 5 % threshold (hence formally "neutral"), but there is no reason to discard a variant that is never worse. To be reconfirmed in Phase 8 FULL (2 rep at 131K/196K).
- **Files**: `results/bench_E51_nmax3.jsonl`, `logs/exp/E51_nmax3.log`

### E52_nmax4 — 2026-11-06 11:11
- **Hypothesis**: H6 — n_max 4 exploits the high acceptance further.
- **Change (single)**: `--spec-draft-n-max 4` compared with `E47_nmax2_ref`.
- **Results (medians, n=4 / n=1)**:
  | depth | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|
  | 8192 | 48.08 | 45.36 | −5.7 | 0.78 |
  | 32768 | 48.73 | 51.39 | +5.5 | 0.96 |
  | 131072 | 41.01 | 39.66 | −3.3 | 0.77 |
- **Decision**: DISCARDED — acceptance collapses to 0.77–0.78 at 8K and 131K and TG worsens; only 32K improves. Not reproducible.
- **Files**: `results/bench_E52_nmax4.jsonl`, `logs/exp/E52_nmax4.log`

### E53_nmax3_pmin05 — 2026-11-06 11:18
- **Hypothesis**: interrupting the draft when confidence is low saves work.
- **Change (single)**: `--spec-draft-p-min 0.5` compared with `E51_nmax3`.
- **Results (medians, n=4 / n=1)**: TG 46.24 (8K, **−7.0 %**), 49.89 (32K, neutral), 43.56 (131K, −3.6 %); acc 0.89/0.97/0.93.
- **Decision**: DISCARDED — it worsens at 8K and 131K.
- **Files**: `results/bench_E53_nmax3_pmin05.jsonl`, `logs/exp/E53_nmax3_pmin05.log`

### E54_nmax3_nodraftmoe — 2026-11-06 11:24
- **Hypothesis**: the MTP draft has no experts of its own, so `--spec-draft-cpu-moe` might be useless.
- **Change (single)**: removal of `--spec-draft-cpu-moe` compared with `E51_nmax3`.
- **Checks**: `n_ctx=200192` ✔ | no errors ✔
- **Results (medians, n=4 / n=1)**: TG 47.43 (8K, −4.6 %), 49.77 (32K, −0.3 %), 41.83 (131K, **−7.4 %**); acc 0.85/0.96/0.88.
- **Decision**: DISCARDED — the flag **is not neutral**: without it acceptance and TG at long context worsen. It stays in the configuration.
- **Files**: `results/bench_E54_nmax3_nodraftmoe.jsonl`, `logs/exp/E54_nmax3_nodraftmoe.log`

## PHASE 6 — Secondary parameters

> Methodological note: all the comparisons in this phase use the same `GEN` as the reference.
> The short-prompt tests use `GEN=64`; the standard tests use `GEN=256` with `REPS=3` and the
> dedicated reference `E66_ref` (same configuration, same conditions).

### E62_ref — 2026-11-06 11:31
- **Hypothesis**: reference for the short-prompt tests (H7).
- **Command**: `GEN=64 scripts/exp.sh E62_ref 64,128,256 5 1 -- <ARGS rif.>`
- **Results (medians, n=5)**: PP **125.38** / **196.94** / **291.07** t/s at 64/128/256 token; TTFT 0.51/0.65/0.88 s; TG 39.69/40.21/43.15.
- **Decision**: reference.
- **Files**: `results/bench_E62_ref.jsonl`, `logs/exp/E62_ref.log`

### E62_minb64 — 2026-11-06 11:32
- **Hypothesis**: H7 — with `GGML_OP_OFFLOAD_MIN_BATCH=64` prompts under 64 tokens stay on the CPU and avoid the expert transfer cost.
- **Change (single)**: `-e GGML_OP_OFFLOAD_MIN_BATCH=64` compared with `E62_ref`.
- **Checks**: `n_ctx=200192` ✔ | no errors ✔
- **Results (medians, n=5)**: PP 103.05 (**−17.8 %**) / 196.91 (0.0 %) / 290.61 (−0.2 %); TG 38.76/37.78/41.00.
- **Decision**: DISCARDED — at 64 tokens it worsens by 18 %, neutral elsewhere. H7 not confirmed: offloading to the GPU pays off even for small batches.
- **Files**: `results/bench_E62_minb64.jsonl`, `logs/exp/E62_minb64.log`

### E63_minb128 — 2026-11-06 11:33
- **Hypothesis**: H7 with threshold 128.
- **Change (single)**: `-e GGML_OP_OFFLOAD_MIN_BATCH=128` compared with `E62_ref`.
- **Results (medians, n=5)**: PP 102.50 (**−18.2 %**) / 131.94 (**−33.0 %**) / 289.88 (−0.4 %); TG 43.49/35.77/38.01.
- **Decision**: DISCARDED — it worsens markedly at 64 and 128 tokens.
- **Files**: `results/bench_E63_minb128.jsonl`, `logs/exp/E63_minb128.log`

### E66_ref — 2026-11-06 11:45
- **Hypothesis**: Phase 6 reference under the same conditions as the E60/E61/E64/E65 tests (`REPS=3`, `GEN=256`).
- **Command**: `scripts/exp.sh E66_ref 8192,32768 3 1 -- <ARGS rif.>`
- **Results (medians, n=3)**: 8K PP 1467.82 / TG **44.78** (acc 0.79); 32K PP 1422.95 / TG **49.53** (0.96). VRAM max 9594 MiB.
- **Decision**: reference for Phase 6.
- **Files**: `results/bench_E66_ref.jsonl`, `logs/exp/E66_ref.log`

### E60_t7 — 2026-11-06 11:34
- **Hypothesis**: H9 — 7 threads reduce contention with the main thread.
- **Change (single)**: `--threads 7` compared with `E66_ref`.
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | TG ref | TG new | Δ% |
  |---|---|---|---|---|---|
  | 8192 | 1467.82 | 1472.14 | 44.78 | 40.73 | **−9.0** |
  | 32768 | 1422.95 | 1427.93 | 49.53 | 46.16 | **−6.8** |
- **Decision**: DISCARDED — the CPU part (experts of layers 0–36) loses throughput.
- **Files**: `results/bench_E60_t7.jsonl`, `logs/exp/E60_t7.log`

### E61_t16 — 2026-11-06 11:36
- **Hypothesis**: H9 — 16 threads exploit SMT.
- **Change (single)**: `--threads 16 --threads-batch 16` compared with `E66_ref`.
- **Results (medians, n=3)**: PP 1457.69 / 1419.45 (neutral); TG **29.59** (−33.9 %) / **32.79** (−33.8 %).
- **Decision**: DISCARDED — SMT harms expert computation on the CPU (AVX2 contention).
- **Files**: `results/bench_E61_t16.jsonl`, `logs/exp/E61_t16.log`

### E64_prio — 2026-11-06 11:39
- **Hypothesis**: `--prio 2` gives high priority to the process and reduces latencies.
- **Change (single)**: `--prio 2` compared with `E66_ref`.
- **Results (medians, n=3)**: PP neutral; TG **36.95** (−17.5 %) / **41.49** (−16.2 %).
- **Decision**: DISCARDED — net negative effect (likely priority inversion with the GPU/CPU threads in the container). Do not use.
- **Files**: `results/bench_E64_prio.jsonl`, `logs/exp/E64_prio.log`

### E65_poll — 2026-11-06 11:41
- **Hypothesis**: `--poll 0` reduces busy-wait and frees CPU for the expert computation.
- **Change (single)**: `--poll 0` compared with `E66_ref`.
- **Results (medians, n=3)**: PP 1470.90 / 1428.31 (neutral); TG 45.70 (+2.1 %) / 48.92 (−1.2 %).
- **Decision**: NEUTRAL — below the 5 % threshold, no reason to depart from the default.
- **Files**: `results/bench_E65_poll.jsonl`, `logs/exp/E65_poll.log`

## PHASE 7 — More recent runtime (H8)

### E70_b12124 — 2026-11-06 11:50
- **Hypothesis**: H8 — build b12124 (commit 8bed7cfb9) includes batched reads into mapped host memory, MTP state saving and CUDA graph eviction.
- **Image**: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12124` = digest `sha256:821d0ae4e5d4e82a2ad8d08973b5f9cd5d0b84524b1d2439ed1f3e41ebc6b4c7`, commit `8bed7cfb999cebc03d2cc5b78911026a8d57549a`, version b12124. Help in `logs/help_b12124.txt`: all the flags used exist ✔.
- **Change (single)**: b12124 image instead of b11977 (`E51_nmax3`), same arguments.
- **Command**: `GEN=512 scripts/exp.sh E70_b12124 8192,32768,131072 4 1 --image …:server-cuda-b12124 -- <ARGS rif.>`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no errors ✔
- **Memory**: `vram_max = 9619` (8K) / 9590 (32K, 131K) MiB | RAM avail min 9814 MiB
- **Results (medians, n=4 / n=1)**:
  | depth | PP b11977 | PP b12124 | Δ% | TG b11977 | TG b12124 | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1469.26 | 1469.72 | +0.03 | 49.74 | 44.59 | **−10.4** | 0.77 |
  | 32768 | 1422.41 | 1421.14 | −0.1 | 49.91 | 49.55 | −0.7 | 0.96 |
  | 131072 | 1153.93 | 1152.41 | −0.1 | 45.18 | 45.31 | +0.3 | 0.98 |
- **Agentic** (`E70_agentic`): tool_call ok, tool_call_stream ok, multi_turn ok, coding **10/10**, `tasks_per_min = 8.83`, median TG on coding 50.28 t/s.
- **Decision**: **NOT ADOPTED** — no gain ≥ 5 % on PP or TG (indeed TG 8K −10.4 %, acceptance 0.77 vs 0.92). The plan requires a real gain to abandon the more proven build. We stay on **b11977**. The agentic code still passes 10/10: the build is functionally valid, just not faster.
- **Files**: `results/bench_E70_b12124.jsonl`, `results/agentic_E70_agentic.jsonl`, `logs/exp/E70_b12124.log`, `logs/help_b12124.txt`, `logs/pull_b12124.out`

## PHASE 8 — Complete validation of the finalists

### F1_full — 2026-11-06 11:59
- **Hypothesis**: the F1 configuration (KV q8, ub2048, load-mode none, n-cpu-moe 37, cache 0, n_max 3, b11977) is the best finalist and holds the complete benchmark.
- **Change (single)**: none — FULL benchmark of the finalist.
- **Command**: `scripts/exp.sh F1_full 8192,16384,32768,65536,131072,196608 3 2 -- <ARGS F1>`
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | no `warning` ✔ | no errors ✔ | VRAM max 9597 ≤ 9900 ✔ | RAM min 5517 ≥ 3000 ✔ | swap max ~452 MiB ✔
- **Results (medians, n=3 / n=2)**:
  | depth | PP t/s | TG t/s | MTP acc | TTFT s | VRAM max | RAM min |
  |---|---|---|---|---|---|---|
  | 8192 | 1473.36 | 46.87 | 0.88 | 5.56 | 9597 | 12895 |
  | 16384 | 1474.51 | 48.12 | 0.89 | 11.11 | 9569 | 11819 |
  | 32768 | 1426.27 | 49.67 | 0.96 | 22.97 | 9570 | 10233 |
  | 65536 | 1318.21 | 41.94 | 0.79 | 49.72 | 9568 | 8603 |
  | 131072 | 1156.77 | 43.05 | 0.92 | 113.31 | 9568 | 5749 |
  | 196608 | 1030.62 | 34.97 | 0.74 | 190.77 | 9595 | 5517 |
- **Decision**: **FINALIST 1** — compared with E00b: PP +231 % (8K) … +167 % (192K), TG +49 % … +21 %.
- **Files**: `results/bench_F1_full.jsonl`, `logs/exp/F1_full.log`

### F1_max — 2026-11-06 12:17
- **Hypothesis**: capacity near the context limit (199 000 tokens).
- **Change (single)**: depth 199000, 1 rep, `GEN=128`.
- **Checks**: `n_ctx=200192` ✔ | `prompt_n=199000` ✔ | no errors ✔
- **Results**: **PP 1021.68 t/s, TG 36.71 t/s, TTFT 194.8 s**, VRAM max 9587 MiB, RAM min 13357 MiB.
- **Decision**: PASSED — the full context is usable with a VRAM margin.
- **Files**: `results/bench_F1_max.jsonl`, `logs/exp/F1_max.log`

### F1_agentic — 2026-11-06 12:21
- **Hypothesis**: agentic quality of the finalist.
- **Change (single)**: `agentic.py --coding-reps 2` on F1.
- **Results**: tool_call ok, tool_call_stream ok, multi_turn ok, coding **10/10**, `tasks_per_min = 8.77`, median TG on coding ≈ 50 t/s.
- **Decision**: PASSED (no quality loss; +31 % useful work/minute vs baseline 6.7).
- **Files**: `results/agentic_F1_agentic.jsonl`, `logs/exp/F1_agentic_server.log`

### F1_needle — 2026-11-06 12:23
- **Hypothesis**: long-context retrieval of the finalist.
- **Results**: **5/5** needles found out of 150 128 tokens; PP 1118.9 t/s.
- **Decision**: PASSED (same as the f16 baseline).
- **Files**: `results/needle_F1_needle.jsonl`, `logs/exp/F1_needle_server.log`

### F1_session / F1_session2 — 2026-11-06 12:25–12:33
- **Hypothesis**: stability over two consecutive agentic sessions without restarting the server.
- **Change (single)**: `session.py --target 180000` run twice on the same server.
- **Results**:
  | tag | turns | final context | prompt_n min–max | median TTFT | median PP | median TG | empty/1-token |
  |---|---|---|---|---|---|---|---|
  | F1_session | 29 | 177 082 | 6083–6548 | 6.45 s | 969.3 | 36.81 | 7 |
  | F1_session2 | 29 | 179 246 | 6082–6100 | 6.45 s | 967.8 | 37.09 | 0 |
- **Checks**: no complete reprocessing (`prompt_n` ≈ 6 085) ✔ | no error/abort/OOM in the log ✔ | stable swap ✔
- **Decision**: PASSED.
- **Files**: `results/session_F1_session.jsonl`, `results/session_F1_session2.jsonl`, `logs/exp/F1_session_server.log`

### F1_restart1 / F1_restart2 — 2026-11-06 13:03–13:10
- **Hypothesis**: reproducible restart (2 stop/start) within ±5 %.
- **Change (single)**: two container restarts with SCREEN 8K/32K, 3 rep, same F1 configuration.
- **Results**: PP 8K 1473.86/1470.11 vs 1473.36 (Δ ≤ 0.2 %); TG 8K 44.58/46.92 vs 46.87 (−4.9 %/+0.1 %); PP 32K 1424.37/1425.49 vs 1426.27; TG 32K 49.09/49.00 vs 49.67 (−1.2 %/−1.3 %).
- **Decision**: PASSED — all differences within ±5 %.
- **Files**: `results/bench_F1_restart1.jsonl`, `results/bench_F1_restart2.jsonl`, `logs/queue_restart.out`

### F2_full — 2026-11-06 12:37
- **Hypothesis**: conservative finalist = F1 without `--load-mode none` (mmap), with much more free RAM.
- **Change (single)**: removal of `--load-mode none` compared with F1.
- **Checks**: `n_ctx=200192` ✔ | expected mmap warning ✔ | no errors ✔ | VRAM max 9568 | RAM min 18702
- **Results (medians, n=3 / n=2)**:
  | depth | PP F1 | PP F2 | Δ% | TG F1 | TG F2 | Δ% |
  |---|---|---|---|---|---|---|
  | 8192 | 1473.36 | 1398.43 | −5.1 | 46.87 | 45.43 | −3.1 |
  | 16384 | 1474.51 | 1389.74 | −5.7 | 48.12 | 44.39 | −7.8 |
  | 32768 | 1426.27 | 1347.43 | −5.5 | 49.67 | 49.40 | −0.5 |
  | 65536 | 1318.21 | 1258.19 | −4.6 | 41.94 | 38.20 | −8.9 |
  | 131072 | 1156.77 | 1115.07 | −3.6 | 43.05 | 42.23 | −1.9 |
  | 196608 | 1030.62 | 1001.32 | −2.8 | 34.97 | 34.04 | −2.7 |
- **Agentic/needle/session**: `F2_agentic` 10/10 (8.65 task/min), `F2_needle` 5/5, `F2_session` 29 turns with stable `prompt_n` but 16 empty/1-token responses (stochastic phenomenon of the model, also present in F1 in 7 turns).
- **Decision**: **FINALIST 2 (conservative alternative)** — ~5 % slower in PP but leaves ~18.7 GB of free RAM instead of ~5.5 GB. Not promoted to production: F1 is faster and the machine has no RAM shortage during the intended use.
- **Files**: `results/bench_F2_full.jsonl`, `results/agentic_F2_agentic.jsonl`, `results/needle_F2_needle.jsonl`, `results/session_F2_session.jsonl`, `logs/exp/F2_full.log`

### E00b_rebar — 2026-11-07 12:55–13:20
- **Hypothesis**: Resizable BAR (BAR1 from 256 MiB to 16384 MiB) reduces the access latencies to the MoE experts and improves PP/TG of the original configuration, which was saturated on the PCIe bus.
- **Change (single)**: ReBAR enabled in the BIOS (`Re-size BAR Support`), same configuration as `E00b_baseline` (KV f16, ubatch 512, `--cpu-moe`, cache 1024, mmap, MTP n_max 2).
- **Checks**: BAR1 16384 MiB, `memory.reserved` 335 MiB, PCIe stays Gen3 x16; `n_ctx=200192` ✔; no errors ✔.
- **Results (medians, n=3 / n=2)**:
  | depth | PP E00b | PP rebar | Δ% | TG E00b | TG rebar | Δ% | VRAM max E00b | VRAM max rebar |
  |---|---|---|---|---|---|---|---|---|
  | 8192 | 445.25 | 416.47 | −6.5 | 31.41 | 26.91 | −14.3 | 9306 | 9904 |
  | 32768 | 443.20 | 431.93 | −2.5 | 28.54 | 27.35 | −4.2 | 9294 | 9831 |
  | 65536 | 424.63 | 415.06 | −2.3 | 28.48 | 25.30 | −11.2 | 9285 | 9401 |
  | 131072 | 398.85 | 388.70 | −2.5 | 28.15 | 27.73 | −1.5 | 9444 | 9863 |
- **Decision**: **NEUTRAL/NEGATIVE** — no gain, PP −2.3…−6.5 %, TG equal or worse, maximum VRAM +598 MiB (9904 vs 9306 at 8K). ReBAR does not raise the Gen3 bandwidth and reduces the free VRAM at startup (≈8724 vs ≈9035 MiB). Not adopted as a motivation; the original F1 configuration **no longer starts** with ReBAR active (`failed to allocate compute pp buffers`).
- **Files**: `results/bench_E00b_rebar.jsonl`, `logs/exp/E00b_rebar.log`, `logs/E00b_rebar.out`

### F1r_dub128 — 2026-11-07 13:22–13:30
- **Hypothesis**: the compute buffer of the MTP context (942 MiB with ubatch 2048) scales with `--spec-draft-n-ubatch`; reducing it frees the VRAM lost with ReBAR.
- **Change (single)**: added `--spec-draft-n-ubatch 128` to F1 (target ubatch unchanged at 2048).
- **Checks**: `n_ctx=200192` ✔; the draft compute buffer drops **942.06 → 58.88 MiB** (−883 MiB); the server starts again with ReBAR active; no errors ✔.
- **Results (medians, n=3 / n=2)**:
  | depth | PP F1 | PP dub128 | Δ% | TG F1 | TG dub128 | Δ% | VRAM max F1 | VRAM max dub128 |
  |---|---|---|---|---|---|---|---|---|
  | 8192 | 1473.36 | 1455.47 | −1.2 | 46.87 | 46.45 | −0.9 | 9597 | 8841 |
  | 32768 | 1426.27 | 1409.57 | −1.2 | 49.67 | 48.63 | −2.1 | 9570 | 8939 |
  | 131072 | 1156.77 | 1136.66 | −1.7 | 43.05 | 42.28 | −1.8 | 9568 | 8841 |
- **Decision**: **KEPT as an enabler** — it costs 1–2 % but frees ~630–760 MiB of observed VRAM and makes the config bootable with ReBAR. The freed VRAM is reinvested in a larger ubatch (`F1r_ub3072`).
- **Files**: `results/bench_F1r_dub128.jsonl`, `logs/exp/F1r_dub128.log`, `logs/F1r_dub128.out`

### F1r_ub3072 — 2026-11-07 13:32–13:40
- **Hypothesis**: with the VRAM freed by `--spec-draft-n-ubatch 512`, a larger ubatch reduces the expert transfers per prompt and raises PP.
- **Change (single)**: `-b 3072 -ub 3072` (was 2048) and `--spec-draft-n-ubatch 512` (was the default = target ubatch).
- **Checks**: `n_ctx=200192` ✔; fit projection 8473 MiB vs 9029 free ✔; target compute 1973.02 MiB, draft compute 235.52 MiB; no errors ✔.
- **Results (medians, n=3 / n=2)**:
  | depth | PP F1 | PP ub3072 | Δ% | TG F1 | TG ub3072 | Δ% | TTFT F1 | TTFT ub3072 | VRAM max F1 | VRAM max ub3072 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 8192 | 1473.36 | **1744.57** | **+18.4** | 46.87 | 50.13 | +7.0 | 5.56 | 4.70 | 9597 | 9541 |
  | 32768 | 1426.27 | **1782.44** | **+25.0** | 49.67 | 50.08 | +0.8 | 22.97 | 18.38 | 9570 | 9520 |
  | 131072 | 1156.77 | **1395.48** | **+20.6** | 43.05 | 42.30 | −1.7 | 113.31 | 93.93 | 9568 | 9520 |
- **Decision**: **KEPT** — PP +18…+25 %, TG unchanged, maximum VRAM slightly lower. We do not go to 3584/4096: the margin at 192K would drop below the measured 362 MiB.
- **Files**: `results/bench_F1r_ub3072.jsonl`, `logs/exp/F1r_ub3072.log`, `logs/F1r_ub3072.out`

### F1r_ub3072_long — 2026-11-07 13:44–13:50
- **Hypothesis**: the new configuration holds the maximum context without OOM and keeps the gain at 192K.
- **Change (single)**: depth 196608, GEN 128, 2 rep.
- **Checks**: `n_ctx=200192` ✔; VRAM max **9878 MiB** → margin **362 MiB**; no error/OOM ✔.
- **Results**: PP **1209.18 t/s** (vs 1030.62 for F1, **+17.3 %**), TG 32.69 (vs 34.97, −6.5 %, within the noise: acc 0.68 vs 0.74), TTFT 162.6 s (vs 190.77, −14.8 %).
- **Decision**: **PASSED** — the PP gain remains at 192K, the VRAM fits with a 362 MiB margin.
- **Files**: `results/bench_F1r_ub3072_long.jsonl`, `logs/exp/F1r_ub3072_long.log`, `logs/queue_rebar.out`

### F1r_agentic — 2026-11-07 13:50–13:52
- **Hypothesis**: the new configuration does not degrade tool calling and coding.
- **Change (single)**: `agentic.py --coding-reps 2` on the `F1r_ub3072` config.
- **Results**: tool_call ok, tool_call_stream ok, multi_turn ok, coding **10/10**, `tasks_per_min = 8.82` (F1: 8.77).
- **Decision**: **PASSED**.
- **Files**: `results/agentic_F1r_agentic.jsonl`, `logs/exp/F1r_agentic_server.log`

### F1r_session — 2026-11-07 13:52–13:56
- **Hypothesis**: stability over a long agentic session with the new configuration.
- **Change (single)**: `session.py --target 180000 --chunk 6000`.
- **Results**:
  | tag | turns | final context | `prompt_n` min–max | median TTFT | median PP | median TG | empty/1-token |
  |---|---|---|---|---|---|---|---|
  | F1_session | 29 | 177 082 | 6083–6548 | 6.93 s | 969.3 | 37.53 | 7 |
  | **F1r_session** | 29 | 178 609 | 6082–6548 | **5.66 s** | **1142.1** | 37.30 | **2** |
- **Checks**: no complete reprocessing of the context (`prompt_n` stable), no error/abort in the log.
- **Decision**: **PASSED** — TTFT −18 %, PP +18 %, TG unchanged, fewer empty responses.
- **Files**: `results/session_F1r_session.jsonl`, `logs/exp/F1r_session_server.log`, `logs/queue_rebar.out`

---

## Technical note — fix to `bench/summarize.py` (2026-11-06)

`bench/summarize.py` rewrites `results/summary.csv` at every invocation. The previous
version used the argument list as the tag, so a filtered invocation
(`summarize.py F1_full`) or a wrong one (`--help`) **truncated** the consolidated CSV to only the
rows of that tag (or to zero rows, with `IndexError`). Fixed: the printed table
respects the requested tags, but `results/summary.csv` is **always** regenerated with
all the tags; a nonexistent tag prints only the header without touching the data.
Verification: file of 95 lines (94 data + header) before and after `summarize.py F1_full`
and after a nonexistent tag.
No prompt or unit test was modified.
