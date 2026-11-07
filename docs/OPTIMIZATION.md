# OPTIMIZATION.md — final configuration, why, limits and management

Operational document for the `qwen36-35b-opt` production.
All figures come from `results/*.jsonl`, `logs/exp/*` and `results/summary.csv`
(medians computed by `python3 bench/summarize.py`). No value is estimated by hand
except where a derived calculation is explicitly indicated.

---

## 1. The final configuration (F1r)

Production container: **`qwen36-35b-opt`**, image
`ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276`
(b11977, commit `7ac5727d6`), port **`127.0.0.1:8080`**.

```
-m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto
--n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe
--spec-draft-n-ubatch 512
-ctk q8_0 -ctv q8_0 -b 3072 -ub 3072 --load-mode none
```

Compared to F1 only three values change: `--spec-draft-n-ubatch 512` (new),
`-b 2048 → 3072` and `-ub 2048 → 3072`. The rest is identical.

Start/stop: `scripts/prod_start.sh` / `scripts/prod_stop.sh` (see §7).

> **ReBAR note (2026-11-07).** The BIOS now has `Re-size BAR Support` active
> (BAR1 16384 MiB). ReBAR **gives no measurable gain** (A/B test on E00b:
> PP −2.3…−6.5 %, TG equal or worse, VRAM max +598 MiB) and reduces free VRAM
> at start-up by ~300 MiB. With ReBAR the old F1 no longer starts; F1r starts thanks to
> `--spec-draft-n-ubatch 512`, which reduces the MTP compute buffer from 942 to 235 MiB.
> If ReBAR is not needed for gaming, disabling it is safe: F1r works in both
> cases.

### 1.1 Line by line

| Flag | Value | Why | Experiment |
|---|---|---|---|
| `-m` | `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | the only model allowed by the plan; never copied/converted | — |
| `--host` / `--port` | `0.0.0.0` / `8080` | inside the container; publication is only `-p 127.0.0.1:8080:8080` | — |
| `--ctx-size` | `200000` | requirement: context ≥ 200 000. The runtime rounds it to **`n_ctx = 200192`** (verified in every log) | — |
| `--n-predict` | `-1` | no cap on generation (agents that write long code) | — |
| `--n-gpu-layers` | `auto` | all layers on the GPU; the experts are then moved by `--n-cpu-moe`. The warning `fit params ... tensor_buft_overrides already set` is harmless: the context stays 200192 | BASELINE §7 |
| `--n-cpu-moe` | `37` | experts of layers **0–36 in RAM**, experts of layers **37–39 in VRAM** (+~1.5 GiB). Resident experts pay neither PCIe nor CPU | **E45** (+8 % PP, +7 % TG vs `--cpu-moe`) |
| `--moe-cache-mib` | `0` | **the MoE cache harms generation**: every miss costs a PCIe transfer, while the CPU computes from DDR4 faster. It only helps PP, and marginally | **E40** (TG +31/+59 % vs 1024), E41, E44 |
| `--flash-attn` | `on` | mandatory for quantized KV on CUDA (in the log `FA_QUANTS ... q8_0-q8_0`); it was already active in the baseline | BASELINE |
| `--parallel` | `1` | a single slot, which can use the whole 200K context. One LLM server at a time (10 GB VRAM) | — |
| `--jinja` | — | model chat template and **tool calling** (used by OpenCode/Cline) | — |
| `--spec-type` | `draft-mtp` | speculative decoding with the MTP head (`nextn`) included in the GGUF; already active in the baseline | BASELINE |
| `--spec-draft-n-max` | `3` | 3 draft token per step: best across all depths tested, +10 % TG at 128K | **E51** (E50 n=1, E52 n=4 discarded) |
| `--spec-draft-cpu-moe` | — | the draft's experts stay on CPU. **It is not neutral**: removing it worsens acceptance and TG at long context | **E54** |
| `--spec-draft-n-ubatch` | `512` | **decouples the MTP context ubatch from the target's.** Without it, the draft inherits the target's ubatch and its compute buffer grows with it (942 MiB at ubatch 2048, measured in F1); with 512 the buffer drops to **235.52 MiB**. Cost: −1…−2 % PP | **F1r_dub128** (58.88 MiB with 128), F1r_ub3072 |
| `-ctk` / `-ctv` | `q8_0` / `q8_0` | **only the target's KV** quantized. KV VRAM 3910 → 2077 MiB (−1833 MiB), quality unchanged (needle 5/5, coding 10/10) | **E11** (E10 target+draft discarded) |
| `-b` | `3072` | logical batch; it must be ≥ ubatch. It was 2048 | **F1r_ub3072** |
| `-ub` | `3072` | ubatch 6× the baseline: **6× fewer expert movements per prompt**. PP +18…+25 % over F1 (which was already 4× the baseline) | **F1r_ub3072** (3584/4096 not tried: insufficient VRAM margin) |
| `--load-mode` | `none` | experts in **pinned `CUDA_Host`** RAM instead of pageable mmap → real DMA. PP +6.5 %, TG +12.6 % at 8K. Cost: ~14.3 GB of non-swappable RAM | **E30** (E31 `mlock` neutral) |

---

## 2. Memory footprint of the final configuration

From `logs/exp/F1r_ub3072.log` (start-up, `n_ctx = 200192`, `n_ubatch = 3072`,
draft `n_ubatch = 512`):

| Buffer | MiB | Notes |
|---|---|---|
| `CUDA0 model buffer` | 3545.70 | non-experts + experts of layers 37–39 (baseline: 1961.70) |
| `CUDA0 KV buffer` (target, q8_0) | 2077.19 | baseline f16: 3910.00 |
| `CUDA0 KV buffer` (draft MTP) | 391.00 | f16 |
| `CUDA0 RS buffer` (Gated DeltaNet) | 251.25 | recurrent state |
| `CUDA0 compute buffer` (main context) | 1973.02 | ubatch 3072 (F1: 1437.02 at ubatch 2048) |
| `CUDA0 compute buffer` (MTP context) | **235.52** | `--spec-draft-n-ubatch 512` (F1: 942.06) |
| `CUDA0 MoE cache` | **0** | disabled |
| **Sum of GPU buffers** | **≈ 8473** | `fit` projection 8473 MiB out of 9029 free |
| `CUDA_Host model buffer` | 12755.85 | experts of layers 0–36, pinned |
| Minimum available RAM | **8666 MiB** at 128K | ~14.3 GB pinned; at 192K the measurement is 10466 MiB |

VRAM margin: at 8K `vram_max` 9541 → ~699 MiB; at 192K `vram_max` **9878** →
**362 MiB** (desktop included). The margin at 192K is the constraint that prevents ubatch
3584/4096 and `--n-cpu-moe 36`. With ReBAR the free VRAM at start-up is ~9029 MiB
(previously ~9070): ~300 MiB less that the old F1 could no longer compensate for.

---

## 3. Gains over the baseline

Baseline `E00b_baseline` (original configuration, KV f16, ubatch 512, MoE cache 1024,
mmap, MTP n_max 2) → `F1` → production `F1r` (`results/bench_F1r_ub3072.jsonl`,
`bench_F1r_ub3072_long.jsonl`).

| Metric | E00b | F1 | **F1r** | Δ F1r vs E00b |
|---|---|---|---|---|
| PP 8K | 445.25 t/s | 1473.36 t/s | **1744.57 t/s** | **+292 %** |
| PP 32K | 443.20 t/s | 1426.27 t/s | **1782.44 t/s** | **+302 %** |
| PP 128K | 398.85 t/s | 1156.77 t/s | **1395.48 t/s** | **+250 %** |
| PP 192K | 385.84 t/s | 1030.62 t/s | **1209.18 t/s** | **+213 %** |
| TG 8K | 31.41 t/s | 46.87 t/s | **50.13 t/s** | **+60 %** |
| TG 32K | 28.54 t/s | 49.67 t/s | **50.08 t/s** | **+75 %** |
| TG 128K | 28.15 t/s | 43.05 t/s | **42.30 t/s** | **+50 %** |
| TG 192K | 28.90 t/s | 34.97 t/s | 32.69 t/s | +13 % |
| TTFT 8K | 18.4 s | 5.56 s | **4.70 s** | **−75 %** |
| TTFT 32K | 73.94 s | 22.97 s | **18.38 s** | **−75 %** |
| TTFT 128K | 328.63 s | 113.31 s | **93.93 s** | **−71 %** |
| TTFT 192K | 509.6 s | 190.77 s | **162.60 s** | **−68 %** |
| Agentic work | 6.7 task/min | 8.77 task/min | **8.82 task/min** | **+32 %** |
| Coding | 10/10 | **10/10** | **10/10** | unchanged |
| Needle 5 needles @150K | 5/5 (f16) | **5/5** (q8_0) | not repeated⁷ | — |
| Long session | 29 turns, ok | 29+29 turns, ok | 29 turns, ok | unchanged |
| VRAM max | 9470 MiB | 9597 MiB | 9878 MiB | +408 MiB |
| Minimum available RAM | 18631 MiB | 5517 MiB | 8666 MiB | −9965 MiB (pinned) |

⁷ The needle test was not repeated on F1r: compared to F1 only the ubatch and the MTP buffer
change, not the KV precision nor the weights. The long session was repeated (TTFT 5.66 s,
PP 1142.1, TG 37.30, 2 empty responses out of 29 turns).

F1r is faster than F1 in PP by **+18…+25 %** with TG unchanged (details in
`RESULTS.md` §1.2).

### 3.1 Where the gain comes from (one variable at a time)

| Step | Configuration | PP 8K | TG 8K | TG 32K | Experiment |
|---|---|---|---|---|---|
| 0 | baseline | 445.25 | 31.41 | 28.54 | E00b |
| 1 | + ubatch 1024 | 753.93 | 30.32 | 28.56 | E01 (+69 % PP) |
| 2 | + KV q8_0 (target) | 764.78 | 31.70 | 28.00 | E11 (VRAM −1905) |
| 3 | + ubatch 2048 | 1309.79 | 28.67 | 27.75 | E20 (+66/71 % PP) |
| 4 | + `--load-mode none` | 1395.03 | 32.29 | 28.44 | E30 (+6.5/12.6 %) |
| 5 | + MoE cache 0 | 1362.78 | 42.22 | 45.09 | E40 (TG +31/59 %) |
| 6 | + `--n-cpu-moe 37` | 1468.84 | 48.49 | 48.66 | E45 (+8/7 %) |
| 7 | + MTP n_max 3 | 1473.36 | 46.87 | 49.67 | E51 / F1_full |
| 8 | + `--spec-draft-n-ubatch 128` (VRAM test only) | 1455.47 | 46.45 | 48.63 | F1r_dub128 (−1.2 % PP) |
| 9 | + `--spec-draft-n-ubatch 512` and ubatch 3072 | **1744.57** | **50.13** | **50.08** | F1r_ub3072 (+18/25 % PP vs F1) |

Step 9 changes two values together because they are linked: without `--spec-draft-n-ubatch`
the draft's compute buffer would grow with the ubatch and would not fit in VRAM. Step 8
measures by itself the cost of the decoupling (−1.2 % PP, −883 MiB of draft buffer).

The bulk of the PP comes from **ubatch** (steps 1 and 3): 16 → 4 full expert passes
for an 8K prompt. The bulk of the TG comes from **MoE cache off + experts in VRAM**
(steps 5 and 6). The q8_0 KV is not a speed-up: it is what frees the VRAM needed
for ubatch 2048 and for the 3 resident expert layers, without measured quality loss.

---

## 4. Demonstrated limits

### 4.1 PCIe Gen3 x16 is the baseline's limit

The B450 motherboard exposes the RTX 3080 over **PCIe Gen3 x16** (the GPU is Gen4-capable,
the host is not; `pcie_gen_max = 3` in every benchmark row). Every prompt ubatch must
read from RAM the MoE experts that do not fit in VRAM.

Calculation from the baseline datum (`results/bench_E00b_baseline.jsonl`, 8K, median):
- 8192 token / ubatch 512 = **16 ubatch**;
- total experts = **13 942 MiB = 13.62 GiB**; with `--cpu-moe` no expert is resident,
  so a large ubatch touches practically all experts → ~13.62 GiB per ubatch;
- median `prompt_ms` = **18 398.6 ms = 18.4 s**.

```
16 × 13.62 GiB / 18.4 s ≈ 11.8 GiB/s ≈ 12.7 GB/s
```

It is the practical limit of Gen3 x16 (~12.6–12.7 GB/s effective after encoding).
**The baseline was therefore saturated on the bus, not on the GPU.** This explains why:
- doubling the ubatch gives almost linear gains (half the passes → half the bytes);
- the MoE cache does not help generation (it adds work, it does not reduce the useful bytes);
- no thread/priority tuning could compensate (Phase 6: everything neutral or worse).

With F1 the resident experts (3/40 layers) and ubatch 2048 reduce the bytes per prompt:
at 8K, 4 sweeps of ~12.6 GiB take 5.56 s ≈ 9.1 GiB/s. **F1 is no longer at the bus ceiling**;
the residual margin is CPU/compute, not PCIe. With F1r (ubatch 3072) the PP rises to 1744.57 t/s:
the bytes per prompt drop further and the bus stays below the ceiling.

**Resizable BAR** does not change this picture: it widens the BAR1 window (256 MiB → 16384 MiB)
but **does not increase Gen3 x16 bandwidth**. The A/B test on E00b shows PP −2.3…−6.5 %, TG equal
or worse and maximum VRAM +598 MiB (`RESULTS.md` §1.3). ReBAR is only useful if needed
for gaming; for llama.cpp it is neutral or slightly negative.

### 4.2 VRAM (10 GB) and context

The context cannot be reduced below 200K by requirement. With KV q8_0 the target occupies
2077 MiB; ~6.4 GiB remain for non-expert weights, resident experts, compute buffer and MTP.
`-ub 4096` is **OOM** even with KV q8_0 (`E21`): with the draft decoupling
(`--spec-draft-n-ubatch`) the practical limit measured is **ubatch 3072**, which at 192K
leaves **362 MiB** of margin. There is no room for another 3 expert layers
(`E46`, `--n-cpu-moe 36`: +1–2 % and VRAM max 9829 MiB at 32K) nor for ubatch 3584/4096.

With ReBAR active the free VRAM at start-up is ~9029 MiB (previously ~9070): ~300 MiB less.
The old F1 (draft buffer 942 MiB) no longer fits; F1r fits because the draft buffer
drops to 235.52 MiB. If ReBAR is disabled, F1r remains valid (indeed, with ~300 MiB
more margin).

### 4.3 A single server

A second `llama-server` on the same model does not fit in 10 GB. Production is
`parallel 1`, one container at a time; rollback requires stopping the other one (§7).

---

## 5. Tried and discarded

| Experiment | What | Why discarded |
|---|---|---|
| `E02_ub2048` (f16) | ubatch 2048 with KV f16 | **OOM** at start-up (the MTP context asks for 942 MiB) |
| `E03` / `E21` (ub4096) | ubatch 4096 | **OOM** even with KV q8_0 (`allocating 1884.12 MiB` failed) |
| `E10_kvq8` | KV q8_0 on target **and** draft | TG −4.3 % at 32K, no useful VRAM benefit |
| `E31_loadnone_mlock` | added `--mlock` | neutral; TG 8K −3.3 %, requires special ulimit |
| `E41_mc1536` | MoE cache 1536 | better than 1024 but **worse than 0** (TG 37.5/32.2 vs 42.2/45.1) |
| `E44_mcMAX` | maximum MoE cache (1792) | TG 39.0/36.3, still below cache 0 |
| `E46_ncmoe36` | 4 expert layers in VRAM | +1–2 % (noise), VRAM margin ~45 MiB |
| `E50_nmax1` | MTP 1 token | TG 46.8/44.4/37.4 < n_max 3 |
| `E52_nmax4` | MTP 4 token | acceptance collapses to 0.77–0.78, TG worse at 8K/128K |
| `E53_pmin05` | `--spec-draft-p-min 0.5` | TG 46.2/49.9/43.6, worse |
| `E54_nodraftmoe` | removal of `--spec-draft-cpu-moe` | acceptance and TG at long context worsen |
| `E60_t7` / `E61_t16` | thread 7 / 16 | the CPU part loses throughput; SMT harms the AVX2 experts |
| `E62` / `E63` | `GGML_OP_OFFLOAD_MIN_BATCH` 64 / 128 | −18 % and worse: offloading is worthwhile even for small batches |
| `E64_prio` | `--prio 2` | net negative effect |
| `E65_poll` | `--poll 0` | neutral (<5 %), we stay on the default |
| `E70_b12124` | runtime b12124 | no gain ≥5 %; TG 8K −10.4 %, acc 0.77 vs 0.92 |
| `E00b_rebar` | ReBAR in the BIOS (BAR1 16 GiB) | PP −2.3…−6.5 %, TG equal or worse, VRAM max +598 MiB: it does not raise Gen3 bandwidth. Tested on 2026-11-07, it is not an optimization |
| `F2` (mmap) | F1 without `--load-mode none` | ~5 % slower in PP; **not discarded**: documented as a conservative alternative (min RAM 18.7 GB instead of 5.5 GB) |

---

## 6. Proposals that require authorization (NOT executed)

These changes fall outside the lab's constraints (no sudo, no changes to
governor/swap/driver/BIOS). They are only proposals, with the expected effect inferred from the data.

1. **CPU governor `performance` + EPP `performance`** (now `powersave`/`balance_performance`).
   The CPU part computes the experts of layers 0–36: a higher and more stable clock can
   reduce the CPU queue during generation. Expected effect: small on PP (tied to PCIe),
   possible on TG; to be measured.
2. **NVIDIA persistence mode** (`nvidia-smi -pm 1`). Avoids the driver initialization cost
   after a stop; it does not change steady-state throughput.
3. **Resizable BAR in the BIOS** — **already tested on 2026-11-07**: BAR1 went from 256 MiB to
   16384 MiB. No gain (PP −2.3…−6.5 %, TG equal or worse, VRAM max +598 MiB);
   bandwidth remains Gen3 x16. If it is not needed for gaming, it can be disabled without loss:
   F1r works with ReBAR active and disabled.
4. **PCIe Gen4 platform** (motherboard + compatible AM4/AM5 CPU). It is the only way to
   raise the bus ceiling above ~12.7 GB/s. A theoretical doubling of the bus does not double
   the throughput (with F1 the bus is no longer saturated), but it reduces the residual cost of the
   non-resident experts. Significant hardware cost: evaluate only if the load grows.
5. **CPU with larger L3** (e.g. Ryzen 5000/5800X3D on AM4). The active experts are
   ~10.4 MiB/layer: a larger L3 can retain more of them and reduce DRAM pressure
   on the CPU part. Requires a CPU change and BIOS verification.
6. **Fans/power limit**: not touched. The power limit is already at maximum (370 W) and the GPU
   is not the bottleneck (average utilization 77–94 % during the tests).

None of these was applied: production runs with the configuration §1.

---

## 7. Start, stop, rollback

Prerequisites: one server at a time. `bin/dk` must be used instead of `docker`.

```bash
# Start the optimized production server (waits for /health, up to 600 s)
scripts/prod_start.sh

# Verifica
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models

# Stop (the qwen36-35b-opt container stays, ready to restart)
scripts/prod_stop.sh
```

**Rollback to the original** (container `qwen36-35b`, never deleted):

```bash
scripts/prod_stop.sh                 # ferma l'ottimizzato
bin/dk start qwen36-35b              # start the original on the same 127.0.0.1:8080
curl -s http://127.0.0.1:8080/v1/models   # deve rispondere 200
```

Return to the optimized one:

```bash
bin/dk stop qwen36-35b
scripts/prod_start.sh
```

The rollback was tried in Phase 9: original started, HTTP 200, then stopped and
`qwen36-35b-opt` restarted. Never run the two containers together (insufficient VRAM).

---

## 8. Recommended settings in the clients

OpenAI-compatible endpoint: **`http://127.0.0.1:8080/v1`**
(model reported by `/v1/models`: `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`).

| Client | Setting | Value |
|---|---|---|
| OpenCode | `baseURL` | `http://127.0.0.1:8080/v1` |
| OpenCode | `limit.context` | **200000** |
| OpenCode | `limit.output` | ≥ 8192 (generation not limited on the server side: `--n-predict -1`) |
| Cline / Roo | context window | **200000** |
| Cline / Roo | max output | ≥ 8192 |
| All | tool calling | requires `--jinja` (already active); the model template is Qwen3.6 |
| All | thinking | for coding tasks send `chat_template_kwargs: {"enable_thinking": false}` (as in `bench/agentic.py`) |

Note: the server has a single slot (`--parallel 1`). Two clients generating at the same time
serialize; it is the trade-off for having the entire 200K context on a single GPU.

---

## 9. Coexistence with the desktop and gaming

The machine is also a desktop workstation with Plasma/Wayland and is used for gaming.
With the LLM server active **~280–650 MiB of VRAM** and **~5.5–8.7 GB of RAM** remain
available: they are resources taken from the compositor and the games. It is not a hardware fault, but
a condition of memory exhaustion, and it is documented in the logs of 2026-11-06/07:

| Time (boot) | Event | Evidence |
|---|---|---|
| 13:15:50 | start of `qwen36-35b-opt` (config F1) | `dockerd ... ep=qwen36-35b-opt` |
| 14:08:12 → 19:06:18 | **3087** compositor errors due to exhausted VRAM | `nvidia-drm ... Failed to allocate NVKMS memory for GEM object` |
| 19:06:18 | compositor fallback to system RAM (artifacts/stutter) | `Failed to allocate NVKMS video memory ... trying to fall back to sysmem` |
| 22:59:46 → 23:58:27 | second server session | `prod_start.sh` / `prod_stop.sh` |
| 23:00:47 → 23:59:36 | **161** NVKMS errors, same messages | `journalctl -k` |
| 00:00:55 | `systemd-oomd` kills **plasmashell** (42 processes) | RAM pressure 90.2 % > 80 % for > 30 s |
| 00:01:11 | `systemd-oomd` kills **Brave** (421 processes) | Chromium: peak 10.8 GB RAM, 14.1 GB swap |

The reported symptoms (graphical artifacts in the browser when moving the mouse, warning of
"degraded graphics performance") are consistent with these two mechanisms:

1. **VRAM exhausted** — the GPU has 10 GB and the server occupies 9.5–9.9 of them. KWin cannot
   allocate the buffers for the new frames (NVKMS) and falls back to system RAM: hence
   glitches and stutters.
2. **RAM under pressure** — `--load-mode none` keeps ~14.3 GB pinned and non-swappable.
   With browsers open it reaches 90 % and `systemd-oomd` kills first the desktop shell,
   then the browser.

Hardware damage excluded: no GPU **Xid** errors and no retired pages
(`Retired Pages`/`Remapped Rows` = N/A) in `nvidia-smi -q`.

### 9.1 Practical rules

- **Before gaming or using the GPU for anything else: `scripts/prod_stop.sh`.** It is the
  simplest solution and costs nothing in speed when the server is serving.
- If the server must stay on while using the desktop, **no** dedicated light profile
  **is planned** (user's choice: the server is stopped). The following remain as documented
  options, should they ever be needed: `--load-mode mmap` returns ~13 GB of RAM
  (cost ~5 % PP, measured in `F2`), `--cpu-moe` returns ~1.5 GiB of VRAM (cost
  ~7 % TG, measured in `E45`), a shorter context reduces the KV proportionally
  (2077 MiB at 200K → ~340 MiB at 32K, linear calculation).
- ReBAR **is not the cause** of these symptoms: the A/B test shows that it improves neither
  the bandwidth nor the available VRAM. If it helps in games, it is fine to keep it; F1r works
  identically with ReBAR active or disabled.

