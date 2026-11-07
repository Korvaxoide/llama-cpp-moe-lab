# METHODOLOGY — Detailed operational plan

> Original document: `PLAN_FOR_AGENT.md`. It is the plan that the executing agent followed,
> preserved in full as the specification of the method.

> Recipient: an AI agent that runs the tests and writes the documentation.
> Follow it **in order**. Do not skip steps. Do not invent results: every number written in the documents
> must come from a file in `results/` or `logs/`. If a step fails, apply the section
> "If it fails" of the step; if there is none, stop and write the problem in `PROGRESS.md`.

Working directory: `$HOME/AI/llama-optimizer-lab` (from now on `$LAB`). All commands start from there.

---

## 0. ABSOLUTE RULES (read them before every phase)

1. **Model**: use only `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`. Do not touch, copy, convert or requantize the file. The directory is mounted `:ro`.
2. **Context**: every configuration must have `--ctx-size 200000` (or higher). The log must show `n_ctx = 200192` (or higher). If `n_ctx` is lower → configuration DISCARDED.
3. **Never `--fit on` with context reduction**: if the log shows "context size reduced" the configuration is invalid.
4. **Network**: publish the ports only on `127.0.0.1` (the script `scripts/run_server.sh` already does this). Never `-p 8080:8080` without an IP.
5. **Docker**: ALWAYS use `bin/dk` instead of `docker` (the OpenCode process does not have the docker group; `bin/dk` adds it with `newgrp`).
6. **One server at a time**: the VRAM (10 GB) is not enough for two instances. Before starting one, check with `bin/dk ps` that no others are running with the model loaded.
7. **Do not delete** the container `qwen36-35b` nor the image `ghcr.io/ggml-org/llama.cpp:server-cuda`. Do not run `docker system prune`, `docker image prune`, `docker rm qwen36-35b`.
8. **Do not modify the operating system**: no sudo, CPU governor, swap/zram, drivers, sysctl. If an idea requires it, write it in `OPTIMIZATION.md` section "Proposals that require authorization" and do not execute it.
9. **Long commands**: benchmarks last from minutes to an hour. ALWAYS launch them detached from the shell, so they survive if the shell is closed (the first baseline attempt died precisely because of this):
   ```bash
   setsid nohup bash -c '<command>' > logs/<name>.out 2>&1 < /dev/null &
   ```
   Then check with `tail -3 logs/<name>.out` at intervals of a few minutes (not every second).
10. **One change per experiment**: every experiment changes ONE variable relative to the current reference configuration (exception: explicit combinations in Phase 6).
11. **Record everything**: after every experiment add an entry to `EXPERIMENTS.md` (template in §9) BEFORE moving to the next one.
12. **When you are unsure**: do not guess a parameter. Check that it exists with
    `bin/dk run --rm ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276 --help | grep -- '<flag>'`.

---

## 1. TECHNICAL CONTEXT ALREADY ACQUIRED (do not redo it, but use it)

### 1.1 Environment
- Details in `HARDWARE.md`. Critical points: Ryzen 7 3700X 8C/16T, 32 GB DDR4, RTX 3080 10 GB, **PCIe Gen3 x16 (limit of the B450 motherboard)**, ~0.9 GB of VRAM used by the desktop.
- Reference image (pinned by digest, the `server-cuda` tag moves):
  `ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276` = b11977, commit 7ac5727d6.
- Most recent published image found: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12124` (commit 8bed7cfb9, 2026-11-04). Includes: #31391 (batched reads into mapped host memory), #30141 (MTP state saving), #30662 (CUDA graph eviction). Does not include #31047 (MoE MMQ layout) nor #31679.

### 1.2 Model
- `qwen35moe`: 40 layers (30 recurrent Gated DeltaNet + 10 full attention) + 1 MTP layer.
- Experts: 13 942 MiB (in RAM with `--cpu-moe`). Non-experts: ~1962 MiB on GPU.
- KV f16: 3910 MiB (target) + 391 MiB (MTP context) at 200K.

### 1.3 Original configuration (reference to preserve)
```
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024
--flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe
```
Values implicit in the logs: `n_batch=2048`, `n_ubatch=512`, `n_threads=8`, `load_mode=mmap`, KV f16.

### 1.4 Hypotheses already formulated from source analysis (to be verified with the experiments)
| ID | Hypothesis | Reason (verified in the code) |
|---|---|---|
| H1 | PP is limited by PCIe | For every ubatch ≥ 32 tokens the scheduler copies to the GPU the experts used by every layer (`llama_context::sched_copy_experts`). With 512 tokens almost all 256 experts are used: ~13.9 GB per ubatch. Measured: 8192 tok in 18.7 s = 16 ubatch × 13.9 GB / 18.7 s ≈ 11.9 GB/s = PCIe Gen3 bandwidth. |
| H2 | A larger `--ubatch-size` multiplies PP | Same transfer amortized over more tokens (1024 → ~2×, 2048 → ~4×, until it becomes compute-bound). Cost: larger compute buffer in VRAM. |
| H3 | `--load-mode none` speeds up PP and the MoE cache misses | With mmap the experts stay in pageable memory (`llama-model-loader.cpp`: "avoid using a host buffer when using mmap"); without mmap they end up in pinned `CUDA_Host` → real DMA. It is the suggestion of the warning "tensor overrides to CPU are used with mmap enabled". Cost: ~14 GB of RAM locked (not swappable). |
| H4 | KV `q8_0` frees ~2.1 GB of VRAM with negligible quality loss | Half the size; CUDA FA supports the q8_0-q8_0 pair (`FA_QUANTS` in the log). The freed VRAM is used for larger ubatches and/or a larger MoE cache. |
| H5 | Larger MoE cache ⇒ faster generation | Generation (ubatch ≤ 8) computes the experts on GPU from the cache; misses are loaded over PCIe at every token. With 1024 MiB, the q6_K/q4_K groups have only 17 slots but the MTP verification (ubatch 3 × 8 experts = 24) requires ≥ 24 → those layers fall out of the cache (`get_slot_map` returns nullptr) and run on the CPU. |
| H5-data | (measured) baseline hit rate 14.5 %, 706 GB loaded during generation | `logs/E00_baseline_server.log`. Generation limited by PCIe: always compare with `--moe-cache-mib 0` (experts computed by the CPU with DDR4 bandwidth) |
| H6 | `--spec-draft-n-max` 3 can beat 2 | Observed acceptance 0.81–0.99. To be measured: actual t/s, not just acceptance. |
| H7 | `GGML_OP_OFFLOAD_MIN_BATCH` (env, default 32) influences short agentic turns | Small incremental prompts pay ~1.2 s of expert transfer; for 32–128 tokens the CPU could be faster. |
| H8 | More recent build (b12124) | Relevant CUDA/MTP commits (see 1.1). |
| H9 | `--threads` | Relevant only for the part of the computation that stays on the CPU. Try 8 (default) vs 7 vs 16. |
| H10 | `--cache-ram` / `--ctx-checkpoints` | They influence prompt reuse in agentic sessions on a hybrid model. To be verified with `bench/session.py`. |

---

## 2. AVAILABLE TOOLS (already written)

| File | Use |
|---|---|
| `bin/dk` | wrapper of `docker` (uses `newgrp docker`) |
| `scripts/run_server.sh NAME [--image IMG] [--port P] [--docker "..."] -- <args>` | starts a container on `127.0.0.1:P` (default 8081) and waits for `/health` |
| `scripts/exp.sh TAG DEPTHS REPS [LONG_REPS] [--image IMG] [--docker "..."] -- <args>` | full experiment: start → `bench.py` → clean stop → log in `logs/exp/TAG.log`, summary in `logs/exp/index.txt`. Adds `-lv 4`. Env variable `GEN` (default 256) = generated tokens |
| `bench/bench.py --tag T --depths 8192,... --reps N --long-reps M --gen G` | exact-length prompt (token IDs from the real corpus), `cache_prompt=false`, random nonce → always cold. Output: `results/bench_T.jsonl` |
| `bench/summarize.py [TAG ...]` | markdown table with medians + `results/summary.csv` |
| `bench/agentic.py --tag T [--coding-reps N]` | tool calling (streaming and non-streaming), multi-turn, 5 coding problems verified by unit tests. Output: `results/agentic_T.jsonl` |
| `bench/session.py --tag T --target 180000 --chunk 6000` | long agentic session OpenCode-style. Output: `results/session_T.jsonl` |

**Warning**: `scripts/exp.sh` always uses the container name `llopt-exp` and removes it at the end. `agentic.py` and `session.py` are not inside `exp.sh`: for them you start the server with `run_server.sh`, launch the script, then stop the server by hand (§3.3).

---

## 3. STANDARD PROCEDURES (reused in all phases)

### 3.1 Check before every experiment
```bash
bin/dk ps --format '{{.Names}} {{.Status}}'      # no llopt-* container should be running
nvidia-smi --query-gpu=memory.used --format=csv   # expect ~900-1000 MiB (desktop only)
free -m                                           # "available" expected > 24000
```
If the VRAM used is > 1500 MiB without active containers, another application (e.g. a Steam game) is using the GPU: note it in `EXPERIMENTS.md` and wait, because the results would not be comparable.

### 3.2 Standard benchmark experiment
```bash
setsid nohup bash -c 'scripts/exp.sh <TAG> <DEPTHS> <REPS> <LONG_REPS> -- <ARGS>' > logs/<TAG>.out 2>&1 < /dev/null &
```
At the end:
```bash
tail -20 logs/exp/index.txt
python3 bench/summarize.py E00_baseline <TAG>
grep -E "n_ctx  |n_ubatch|n_batch|load_mode|KV buffer|MoE cache size|slots|compute buffer|llama_moe_cache:" logs/exp/<TAG>.log | head -40
```
Mandatory checks (if one fails → configuration INVALID, write it down):
- `n_ctx = 200192` (or higher) in the log.
- In `results/bench_<TAG>.jsonl` no line has `"warning"` and `prompt_n` is equal to `depth`.
- No "CUDA error", "out of memory", "GGML_ABORT" in the log: `grep -iE "cuda error|out of memory|abort|failed to allocate" logs/exp/<TAG>.log`.

### 3.3 Manual start for agentic tests
```bash
scripts/run_server.sh llopt-agent -- <ARGS>
python3 bench/agentic.py --tag <TAG> --coding-reps 2
bin/dk stop -t 60 llopt-agent && bin/dk logs llopt-agent > logs/exp/<TAG>_agent.log 2>&1 && bin/dk rm llopt-agent
```

### 3.4 Depth sets
- **SCREEN** (fast, for discarding): `DEPTHS=8192,32768 REPS=3 LONG_REPS=1`, `GEN=256`. Duration ≈ 6–8 min at baseline speed.
- **FULL** (only for the candidate configurations): `DEPTHS=8192,16384,32768,65536,131072,196608 REPS=3 LONG_REPS=2`, `GEN=256`. Duration: 40–90 min.
- **MAX**: a single measurement at `199000` tokens (`DEPTHS=199000 REPS=1 LONG_REPS=1`, `GEN=128`) to demonstrate capacity near the limit.

### 3.5 Decision rule (always use it, write it in EXPERIMENTS.md)
Compare the medians of the new configuration with those of the current reference on the same depths:
- **PP**: real improvement if the median improves ≥ 5 % and min(new) > max(reference) at at least one depth.
- **TG**: TG varies a lot with MTP acceptance (0.8–0.99 with the same prompt). Real improvement only if the median improves ≥ 5 % with at least 3 repetitions per depth. Between 0 and 5 % → "neutral".
- **Regression**: median worse by > 3 % on TG or > 5 % on PP at any depth ≥ 32K → discard, unless the other indicator improves much more (e.g. PP +100 % and TG −3 %: keep but note the trade-off).
- **VRAM**: `vram_max` must stay ≤ 9900 MiB (margin for the desktop). Above → OOM risk → discard or reduce.
- **RAM**: `ram_avail_min` ≥ 3000 MiB and `swap_used_max` must not grow beyond 1000 MiB.

---

## PHASE 0 — Restoring the state and completing the baseline

**Objective**: complete and reproducible baseline of the original configuration.

0.1 Check what is running:
```bash
bin/dk ps -a --format '{{.Names}} {{.Status}}'
```
If `llopt-baseline` is "Up", stop it and save the logs:
```bash
bin/dk stop -t 60 llopt-baseline; bin/dk logs llopt-baseline > logs/E00_baseline_server.log 2>&1; bin/dk rm llopt-baseline
grep "llama_moe_cache:" logs/E00_baseline_server.log   # baseline MoE cache hit rate -> record it
```

0.2 The FULL baseline had been interrupted. The file `results/bench_E00_baseline.jsonl` already contains 8K (2 rep), 16K (2), 32K (2), 64K (1). **Do not delete it.** Rerun the complete baseline with a new tag, so that all depths have the same conditions:
```bash
setsid nohup bash -c 'scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe' > logs/E00b.out 2>&1 < /dev/null &
```
Expected duration: ~70–90 min (192K at ~400 t/s ≈ 8 min per repetition).
If it fails at 192K due to OOM: note it as a limit of the baseline (it is a result!) and continue.

0.3 Baseline agentic test (§3.3, TAG `E00_agentic`, same arguments). Note: tool_call ok?, tool_call_stream ok?, multi_turn ok?, coding_pass x/10, tasks_per_min.
**If `agentic.py` errors out** (it was written but not tested): read the traceback, fix the script (Python bug, not the server), note the fix in `EXPERIMENTS.md`, rerun. Do not modify the prompts or the unit tests after you have started comparing configurations.

0.4 Baseline long session: start the server (§3.3, name `llopt-agent`), then
```bash
setsid nohup bash -c 'python3 bench/session.py --tag E00_session --target 180000 --chunk 6000' > logs/E00_session.out 2>&1 < /dev/null &
```
Duration: ~30 turns, 30–60 min. Note per turn: `ttft`, `prompt_n` (tokens actually reprocessed) vs `cache_n` (reused). **Critical check**: `prompt_n` must stay approximately = chunk (~6000 + a few hundred). If at some turn `prompt_n` becomes almost equal to `ctx_tokens` the server has reprocessed the whole context (cache loss): note it with the turn number; it is an important problem for agentic use.
At the end stop the server and save the log (`logs/exp/E00_session_server.log`).

0.5 Write `BASELINE.md` (template §10.1).

**Phase 0 exit criterion**: `BASELINE.md` complete with FULL table, agentic outcome, session outcome.

---

## PHASE 1 — Prompt processing: ubatch size (H2)

**Bottleneck**: PCIe transfer of the experts for every ubatch.
**Metrics**: PP t/s at 8K and 32K; `vram_max`; TG must not regress.

SCREEN experiments (each changes only `-b`/`-ub`; everything else = original configuration; `ARGS_BASE` = original arguments):
| TAG | added arguments |
|---|---|
| E01_ub1024 | `-b 2048 -ub 1024` |
| E02_ub2048 | `-b 2048 -ub 2048` |
| E03_ub4096 | `-b 4096 -ub 4096` |

Example:
```bash
setsid nohup bash -c 'scripts/exp.sh E02_ub2048 8192,32768 3 1 -- --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe -b 2048 -ub 2048' > logs/E02.out 2>&1 < /dev/null &
```

**Expected** (to be verified, not copied): PP almost doubles at every doubling of the ubatch until the GPU compute becomes dominant.
**Possible failure**: OOM at startup or during the prompt (larger compute buffer; baseline VRAM already at ~9.6 GB). If it happens:
- note the size of the `CUDA0 compute buffer size` from the log (if present),
- do NOT reduce the context; try the same ubatch in Phase 2 after freeing VRAM with KV q8_0.

For every ubatch that starts, also note `CUDA0 compute buffer size` (extra VRAM paid).

**Decision**: choose the ubatch with the highest PP that stays under 9900 MiB of VRAM *at 32K*. Then verify that it holds at 196608 with a single run:
```bash
scripts/exp.sh E0x_ubXXXX_long 196608 1 1 -- <ARGS> 
```
(launch it with setsid/nohup). If it goes OOM at 196K → ubatch too large for now, go back to the lower one.

---

## PHASE 2 — Quantized KV cache (H4)

**Bottleneck**: VRAM occupied by the f16 KV (4.3 GB).
**Metrics**: VRAM (`KV buffer size` in the log, `vram_max`), PP and TG at 8K/32K/128K, **quality** (agentic.py coding pass).

| TAG | arguments added to ARGS_BASE (with the best ubatch of Phase 1, or 512 if none worked) |
|---|---|
| E10_kvq8 | `-ctk q8_0 -ctv q8_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0` |
| E11_kvq8_tgt_only | `-ctk q8_0 -ctv q8_0` (MTP draft KV stays f16) |

Notes:
- The flags `--spec-draft-type-k/-v` exist in this version (`common/arg.cpp` lines 4146–4170). Verify with `--help` if startup fails.
- **Do not use** `q4_0` for the KV as a final choice without very solid quality evidence (it degrades long-context retrieval). You may test it as E12_kvq4 for documentation purposes only.
- In the log check the line `llama_kv_cache: size = ... K (q8_0) ..., V (q8_0)`.

**Mandatory quality test** for every quantized KV you want to keep:
1. `bench/agentic.py --coding-reps 2` → coding_pass must not drop by more than 1 compared with the baseline (10 tests).
2. Long-context retrieval test ("needle"): see §PHASE 2b.

### PHASE 2b — Long-context needle test (write the script `bench/needle.py` yourself)
Simple script, to be created:
- Takes ~150 000 tokens from the corpus (`bench/corpus/code_corpus.tokens.json`), detokenizes them with `/detokenize` in 5 pieces.
- Inserts in 5 positions (10 %, 30 %, 50 %, 70 %, 90 %) a line `// SECRET_CONFIG_<i> = "<parola casuale di 8 lettere>"`.
- Sends via `/v1/chat/completions` a user message: `<testo>\n\nWhat are the exact values of SECRET_CONFIG_1 ... SECRET_CONFIG_5? Answer as JSON.` with `temperature 0`, `max_tokens 2000`, `chat_template_kwargs: {"enable_thinking": false}`.
- Score = number of exact values found in the response (0–5). Save in `results/needle_<TAG>.jsonl`. Use a fixed seed for the random words (`random.seed(42)`), so that every configuration receives the same prompt.
- Run on the baseline (f16) and on every candidate quantized KV. Accept the quantized KV only if the score equals that of f16 (or at most −1 out of 5 by repeating the test a second time for confirmation).

**Phase 2 decision**: if q8_0 passes the quality tests and is neutral or better in speed → it becomes the new reference (the freed VRAM is the real gain, to be spent in Phases 3–4).

---

## PHASE 3 — Re-expanding the ubatch with the freed VRAM

Only if Phase 2 adopted q8_0. Repeat Phase 1 for the ubatch that previously went OOM (e.g. 4096) with `-ctk q8_0 -ctv q8_0 ...`. TAG `E20_kvq8_ub4096`, etc. Same decision as Phase 1, including the 196608 test.

---

## PHASE 4 — Pinned memory: `--load-mode none` (H3)

**Bottleneck**: H2D copies from pageable memory (mmap).
**Metrics**: PP (must increase if H3 is true), TG, load time (`load_s` in `logs/exp/index.txt`), available RAM.

| TAG | arguments added to the current reference |
|---|---|
| E30_loadnone | `--load-mode none` |
| E31_loadnone_mlock | `--load-mode mlock` (if `none` is better; it may require `--docker "--ulimit memlock=-1:-1"` in the script) |

Checks in the log:
- `load_tensors: loading model tensors ... (load_mode = none)`.
- The lines `CPU_Mapped model buffer size` must become `CUDA_Host model buffer size`. If they stay CPU/CPU_Mapped, H3 does not apply: note it.
- The warning "tensor overrides to CPU are used with mmap enabled" must disappear.
- `free -m` after startup (`logs/exp/<TAG>.mem_idle.txt`): with ~14 GB pinned the available RAM will drop. If `available` < 6000 MiB, note the risk for daily use (browser, IDE).
- Stability: after the SCREEN test also run `bench/session.py` for at least 60 000 tokens with this configuration (pinned memory + long session).

**If startup fails** ("failed to allocate CUDA_Host buffer" or similar): try `--load-mode none --no-host` (in this case the experts go into normal non-pinned RAM, useful only for comparison) and note it.

---

## PHASE 5 — Generation: MoE cache (H5) and MTP (H6)

### 5.1 MoE cache size
Current reference = best so far. Change only `--moe-cache-mib`. Use SCREEN but with `REPS=4` (TG is noisy) and `GEN=512`:
| TAG | value |
|---|---|
| E40_mc0 | `0` (cache disabled: experts on CPU during generation) — important to understand whether the cache really helps |
| E41_mc1536 | `1536` |
| E42_mc2048 | `2048` |
| E43_mc3072 | `3072` (only if VRAM allows it) |
| E44_mcMAX | the maximum that leaves `vram_max` ≤ 9900 MiB at 32K |

For each one note from the log:
- the lines `impl: N layers, <tipo>: <slot> slots (x%)` → verify that every group has **≥ 24 slots** (= (n_max+1)×8). With n_max=3 you need ≥ 32.
- at stop: `llama_moe_cache: ubatch <= 8: hits = ..., misses = ..., hit rate = ...%, uploaded = ... MiB`.
Create a table cache-MiB → hit rate → median TG. Expected: TG grows with the hit rate as long as VRAM allows it.

**VRAM warning**: MoE cache + large ubatch compete. If the large ubatch forces a small cache, evaluate the trade-off with the data of both metrics and the agentic session (§PHASE 7). Document the choice.

**Do not use `--no-kv-offload` together with the MoE cache**: it causes the known abort "the experts of layer 0 are selected in the same split as their MoE cache lookup" (see `src/llama.cpp/src/llama-moe-cache.cpp:560`). If you want to document it, a single attempt, then discard.

### 5.2 Alternative: whole expert layers on GPU instead of the cache
Experiment E45: replace `--cpu-moe` with `--n-cpu-moe N` (e.g. N=37 → layers 37–39 have the experts in VRAM) and reduce the cache accordingly (~332 MiB per layer). Compare with the cache of the same total size. Verify with `--help` that `--n-cpu-moe` exists (it does: `common/arg.cpp` line 2779). Note: `--n-cpu-moe N` keeps the *first* N layers on the CPU.

### 5.3 MTP draft length
Reference = best after 5.1. Change only `--spec-draft-n-max`. `REPS=4`, `GEN=512`, depths 8192,32768,131072 (131072 with 1 rep):
| TAG | value |
|---|---|
| E50_nmax1 | 1 |
| E51_nmax3 | 3 |
| E52_nmax4 | 4 |
Main metric: median TG t/s (not acceptance). With n_max 3/4 check that the MoE cache has ≥ (n_max+1)×8 slots for every group, otherwise increase it (and note it).
Optional experiment E53: `--spec-draft-p-min 0.5` (or 0.75) with the best n_max: it interrupts the draft when confidence is low.
Experiment E54: remove `--spec-draft-cpu-moe` (the MTP draft has no experts of its own, so it is probably neutral: verify it and document it; if neutral, keep it as in the original).

---

## PHASE 6 — Secondary parameters

One at a time on the current reference, SCREEN with `REPS=3`:
| TAG | change | what to look at |
|---|---|---|
| E60_t7 | `--threads 7` | TG |
| E61_t16 | `--threads 16 --threads-batch 16` | TG/PP |
| E62_minb64 | `--docker "-e GGML_OP_OFFLOAD_MIN_BATCH=64"` | short prompts only: use `DEPTHS=64,128,256 REPS=5`, compare with the same test on the reference (TAG `E62_ref`) |
| E63_minb128 | `--docker "-e GGML_OP_OFFLOAD_MIN_BATCH=128"` | as above |
| E64_prio | `--prio 2` | TG (it may have no effect in an unprivileged container: note it) |
| E65_poll | `--poll 0` | TG and CPU usage |

For the short-prompt tests (E62/E63), `bench.py` accepts any depth, including small ones.

---

## PHASE 7 — More recent runtime (H8)

7.1 Download the image with an explicit tag (do NOT pull `server-cuda`, it would move the tag of the original):
```bash
setsid nohup bash -c 'bin/dk pull ghcr.io/ggml-org/llama.cpp:server-cuda-b12124' > logs/pull_b12124.out 2>&1 < /dev/null &
```
Note the digest: `bin/dk inspect --format '{{index .RepoDigests 0}} {{index .Config.Labels "org.opencontainers.image.revision"}}' ghcr.io/ggml-org/llama.cpp:server-cuda-b12124`.

7.2 Verify that all the flags of the best configuration exist in the new image:
```bash
bin/dk run --rm ghcr.io/ggml-org/llama.cpp:server-cuda-b12124 --help > logs/help_b12124.txt 2>&1
```
and look for every flag used with `grep -- '--flag' logs/help_b12124.txt`.

7.3 Run SCREEN with the best configuration and `--image ghcr.io/ggml-org/llama.cpp:server-cuda-b12124` (TAG `E70_b12124`), then `agentic.py` (TAG `E70_agentic`).
Decision: adopt the new build only if ≥ 5 % better on PP or TG **and** the agentic tests all pass **and** the long session (Phase 8) shows no regressions. Otherwise stay on b11977 (more proven by the user).

7.4 (Optional, only if everything else is done) Custom build with `-DCMAKE_CUDA_ARCHITECTURES=86`. It requires writing a Dockerfile based on `nvidia/cuda:12.8.1-devel-ubuntu24.04` and on the official Dockerfiles in `src/llama.cpp/.devops/cuda.Dockerfile`. Save the Dockerfile in `docker/`. The expected gain is small; document it as an experiment anyway.

---

## PHASE 8 — Complete validation of the finalists

Select at most 2 finalist configurations (the best one + the most conservative alternative, e.g. without `--load-mode none`).

For each one:
1. **FULL** benchmark (§3.4) — TAG `F1_full`, `F2_full`.
2. **MAX** at 199000 tokens — TAG `F1_max`. Note TTFT and TG.
3. `agentic.py --coding-reps 2` — TAG `F1_agentic`.
4. `needle.py` — TAG `F1_needle`.
5. `session.py --target 180000` — TAG `F1_session`. Check: no complete reprocessing of the context, TTFT per turn, TG in the final turns.
6. **Stability**: repeat `session.py` a second time without restarting the server (extended session, ~1–2 hours total). At the end: `grep -iE "error|abort|oom" logs`, final VRAM and RAM, swap used. No errors allowed.
7. **Reproducible restart**: stop and restart the container 2 times, verify that the first SCREEN test gives results within ±5 %.

Compare F1/F2 with E00b in the table of `RESULTS.md`.

---

## PHASE 9 — Moving to production (replacement of the original configuration)

Execute ONLY after Phase 8 has passed.

9.1 Create `docker/compose.yaml` (or the scripts `scripts/prod_start.sh` / `scripts/prod_stop.sh`) with:
- container name `qwen36-35b-opt`,
- image pinned by digest,
- `-p 127.0.0.1:8080:8080` (loopback only, same port as the original),
- `-v ~/AI/llama-models:/models:ro`,
- `--restart unless-stopped` only if the user wants it (by default: `no`, like the original),
- all the arguments of the final configuration (without `-lv 4`).

9.2 Replacement procedure (the original container must NOT be deleted):
```bash
bin/dk ps -a --format '{{.Names}} {{.Status}}'   # qwen36-35b must be stopped
scripts/prod_start.sh                             # start qwen36-35b-opt on 8080
curl -s http://127.0.0.1:8080/v1/models
python3 bench/agentic.py --url http://127.0.0.1:8080 --tag PROD_check
```
If something fails: `scripts/prod_stop.sh` and restart the original with `bin/dk start qwen36-35b`. Verify `curl http://127.0.0.1:8080/v1/models`.

9.3 Write in `OPTIMIZATION.md` the start/stop/rollback instructions and the values to set in the clients (OpenCode `limit.context: 200000`, adequate `output`; same for Cline).

---

## PHASE 10 — Final documentation (mandatory)

Update/create all the files. Every number must cite the source file (e.g. "source: `results/bench_F1_full.jsonl`").

### 10.1 `BASELINE.md`
- Exact original configuration (copy from `baseline/original/qwen36-35b.inspect.json`: Cmd, image, digest, mount, ports).
- VRAM breakdown from the log (GPU model, KV, MTP KV, RS, MoE cache, compute buffer).
- FULL E00b table (output of `bench/summarize.py E00b_baseline`).
- Agentic and baseline long-session results.
- Observations from the logs of the real original session (`logs/original_full_run.log`): TG ~26 t/s at ~150K, incremental prompts ~350 t/s, TTFT ~7 s per ~2.4K tokens.
- Explanation of the two original warnings:
  - "tensor overrides to CPU are used with mmap enabled": with mmap the experts cannot use the pinned `CUDA_Host` buffer → tested in Phase 4.
  - "failed to fit params ... tensor_buft_overrides already set by user": `--fit` gives up when `--cpu-moe` is used; it is harmless (the context stays the requested one) but it means the placement is manual.

### 10.2 `EXPERIMENTS.md` (chronological log, one entry per experiment, template §9 below)

### 10.3 `RESULTS.md`
- Main table: rows = configurations (E00b, best of every phase, F1, F2), columns = PP and TG at 8K/32K/128K/192K, TTFT at 32K and 128K, VRAM max, RAM available min, coding pass, needle, session outcome.
- MoE cache table (MiB → min slots per group → hit rate → TG).
- Ubatch table (ubatch → compute buffer → PP 8K/32K).
- MTP table (n_max → acceptance → TG).
- `results/summary.csv` updated (`python3 bench/summarize.py`).

### 10.4 `OPTIMIZATION.md`
- Complete final configuration, line by line, with the reason for every flag and the experiment that justifies it.
- Gains compared with the baseline (% for PP, TG, TTFT, agentic work per minute).
- Demonstrated limits (e.g. PCIe Gen3: compute the effective bandwidth from the data = expert bytes transferred / prompt time).
- Things tried and discarded, with reason.
- "Proposals that require authorization" (e.g. `performance` governor, NVIDIA persistence mode, Resizable BAR in the BIOS, upgrade to a motherboard with PCIe Gen4) — NOT executed.
- Start/stop/rollback instructions.

### 10.5 `README.md`
Mission, constraints, file map, how to reproduce a benchmark in 3 commands.

### 10.6 `PROGRESS.md`
Update it at the end of **every phase** (tick what is done, what remains, container status).

---

## 9. TEMPLATE FOR EVERY ENTRY OF `EXPERIMENTS.md`

```markdown
### E02_ub2048 — 2026-11-06 10:15
- **Hypothesis**: H2 — ubatch 2048 cuts PCIe transfers per prompt token by 4×.
- **Bottleneck addressed**: PCIe H2D expert transfers during prompt processing.
- **Change (single)**: `-b 2048 -ub 2048` relative to <reference TAG>.
- **Command**: `scripts/exp.sh E02_ub2048 8192,32768 3 1 -- ...`
- **Checks**: n_ctx=200192 ✔ | prompt_n=depth ✔ | errors in the log: none ✔
- **Memory**: CUDA0 compute buffer = ... MiB (ref. 565.5) | vram_max = ... MiB | RAM avail min = ... MiB
- **Results (medians, n=3)**:
  | depth | PP ref | PP new | Δ% | TG ref | TG new | Δ% | acc |
  |---|---|---|---|---|---|---|---|
- **Decision**: KEPT / DISCARDED / NEUTRAL — reason according to §3.5.
- **File**: results/bench_E02_ub2048.jsonl, logs/exp/E02_ub2048.log
```

---

## 10. FINAL CHECKLIST (copy it at the bottom of PROGRESS.md and tick it)

- [ ] E00b FULL baseline complete (8K→192K)
- [ ] Agentic baseline + long session
- [ ] Phase 1 ubatch (1024/2048/4096)
- [ ] Phase 2 KV q8_0 + quality (coding + needle)
- [ ] Phase 3 ubatch with KV q8_0
- [ ] Phase 4 load-mode none / mlock
- [ ] Phase 5.1 MoE cache sweep with hit rate
- [ ] Phase 5.2 n-cpu-moe vs cache
- [ ] Phase 5.3 MTP n_max 1/3/4 (+ p_min)
- [ ] Phase 6 thread / offload min batch / prio / poll
- [ ] Phase 7 runtime b12124
- [ ] Phase 8 finalists: FULL, MAX 199K, agentic, needle, session ×2, restart
- [ ] Phase 9 production on 127.0.0.1:8080, original preserved, rollback tested
- [ ] Phase 10 all documents: README, HARDWARE, BASELINE, EXPERIMENTS, RESULTS, OPTIMIZATION, PROGRESS
- [ ] No experimental container left running (`bin/dk ps`)

---

## 11. COMMON MISTAKES TO AVOID

1. Comparing runs with different `GEN`, or a depth with 1 rep against one with 3, without declaring it.
2. Declaring a TG improvement from a single repetition (MTP acceptance varies from 0.8 to 0.99).
3. Leaving a `llopt-*` container running and starting another one (OOM or distorted numbers).
4. Forgetting that the desktop uses ~0.9 GB of VRAM: if the user opens a game during the tests, the numbers change.
5. Using `docker` instead of `bin/dk` (permission denied).
6. Running `docker pull ...:server-cuda` (it moves the tag used by the original configuration).
7. Rewriting numbers by hand in the documents: always generate them with `bench/summarize.py` and copy the table.
8. Launching long benchmarks in the foreground or as simple background processes of the shell: use `setsid nohup`.
9. Choosing a configuration that holds at 32K but goes OOM at 192K: the 196608 test is mandatory before adopting it.
10. Treating MTP acceptance as the goal: the goal is tokens/s and useful work per minute.
