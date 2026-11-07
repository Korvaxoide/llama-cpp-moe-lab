# RESULTS.md — consolidated results

All figures come from `results/bench_*.jsonl`, `results/agentic_*.jsonl`,
`results/session_*.jsonl`, `results/needle_*.jsonl` and from the logs in `logs/exp/`.
The medians are computed by `python3 bench/summarize.py` (full output in
`results/summary.csv`).

## 1. Main table (key configurations)

| Configuration | PP 8K | TG 8K | PP 32K | TG 32K | PP 128K | TG 128K | PP 192K | TG 192K | TTFT 32K | TTFT 128K | VRAM max | RAM min | coding | needle | session |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E00b** baseline (KV f16, ub512, mc1024, mmap, n_max2) | 445.25 | 31.41 | 443.20 | 28.54 | 398.85 | 28.15 | 385.84 | 28.90 | 73.94 | 328.63 | 9470 | 18631 | 10/10 | 5/5 | ok (29 turns) |
| **E01** + ub1024 | 753.93 | 30.32 | 761.32 | 28.56 | — | — | 613.66¹ | 30.81¹ | 43.04 | — | 9879 | 23208 | (10/10) | 5/5² | — |
| **E20** + KV q8 + ub2048 (mc1024, mmap) | 1309.79 | 28.67 | 1267.00 | 27.75 | — | — | 931.92³ | 24.75³ | 25.86 | — | 8939 | 24112 | (9/10)⁴ | 5/5⁴ | — |
| **E30** + `--load-mode none` | 1395.03 | 32.29 | 1351.21 | 28.44 | — | — | — | — | 24.25 | — | 8963 | 9669 | — | — | ok⁵ |
| **E40** MoE cache 0 | 1362.78 | 42.22 | 1323.88 | 45.09 | — | — | — | — | 24.75 | — | 7960 | 8814 | — | — | — |
| **E45** `--n-cpu-moe 37` + cache 0 | 1468.84 | 48.49 | 1423.32 | 48.66 | — | — | — | — | 23.02 | — | 9525 | 10485 | — | — | — |
| **F1** = E45 + n_max 3 (finalist) | 1473.36 | 46.87 | 1426.27 | 49.67 | 1156.77 | 43.05 | 1030.62 | 34.97 | 22.97 | 113.31 | 9597 | **5517** | **10/10** | **5/5** | **ok ×2** |
| **F2** = F1 without `--load-mode none` (mmap) | 1398.43 | 45.43 | 1347.43 | 49.40 | 1115.07 | 42.23 | 1001.32 | 34.04 | 24.32 | 117.55 | 9568 | 18702 | **10/10** | **5/5** | ok but 16 empty responses⁶ |
| **F1r** = F1 + `--spec-draft-n-ubatch 512` + `-b 3072 -ub 3072` | **1744.57** | **50.13** | **1782.44** | **50.08** | **1395.48** | **42.30** | **1209.18** | 32.69 | **18.38** | **93.93** | 9878 | 8666 | **10/10** | —⁷ | ok (TTFT 5.66 s) |

¹ `E01_ub1024_long` (196608, 1 rep). ² `E01_needle_f16`. ³ `E20_kvq8_ub2048_long`.
⁴ `E11_kvq8_tgt_only` (same KV q8, ub1024). ⁵ `E30_session_none` (9 turns, 60K).
⁶ In `F2_session` 16/29 turns produced an empty/1-token response (finish `stop`);
the median TG over valid turns is 42.92 t/s. F1: 7/29 empty turns, TG 37.53 t/s over valid ones.
The phenomenon belongs to the model/prompt, not the server (no error in the log).
⁷ The needle test was not repeated on F1r: the only difference from F1 is the size
of the ubatch and of the MTP buffer, not the KV precision nor the weights. Coding and the long session
were repeated and are unchanged/better.

Sources: `results/bench_E00b_baseline.jsonl`, `bench_E01_ub1024.jsonl`,
`bench_E01_ub1024_long.jsonl`, `bench_E20_kvq8_ub2048.jsonl`, `bench_E20_kvq8_ub2048_long.jsonl`,
`bench_E30_loadnone.jsonl`, `bench_E40_mc0.jsonl`, `bench_E45_ncmoe37.jsonl`,
`bench_F1_full.jsonl`, `bench_F2_full.jsonl`, `bench_F1r_ub3072.jsonl`,
`bench_F1r_ub3072_long.jsonl`.

### 1.1 Final comparison F1 vs baseline E00b

| Metric | E00b | F1 | Δ |
|---|---|---|---|
| PP 8K | 445.25 t/s | 1473.36 t/s | **+231 %** |
| PP 32K | 443.20 t/s | 1426.27 t/s | **+222 %** |
| PP 128K | 398.85 t/s | 1156.77 t/s | **+190 %** |
| PP 192K | 385.84 t/s | 1030.62 t/s | **+167 %** |
| TG 8K | 31.41 t/s | 46.87 t/s | **+49 %** |
| TG 32K | 28.54 t/s | 49.67 t/s | **+74 %** |
| TG 128K | 28.15 t/s | 43.05 t/s | **+53 %** |
| TG 192K | 28.90 t/s | 34.97 t/s | **+21 %** |
| TTFT 32K | 73.94 s | 22.97 s | **−69 %** |
| TTFT 128K | 328.63 s | 113.31 s | **−66 %** |
| TTFT ~2.4K (real session) | ~7.0 s | ~2.5 s (6.45 s per 6K) | **−64 %** |
| Agentic work | 6.7 task/min | 8.77 task/min | **+31 %** |
| VRAM max | 9470 MiB | 9597 MiB | +127 MiB |
| Minimum available RAM | 18631 MiB | 5517 MiB | −13114 MiB (pinned) |

### 1.2 F1r configuration (ReBAR active): F1 → F1r

ReBAR was enabled in the BIOS on 2026-11-07 (BAR1 256 → 16384 MiB). The
original F1 configuration **no longer starts** with ReBAR active: free VRAM
at start-up drops from ~9070 to ~8763 MiB and the allocation of the compute buffer of the MTP
context fails (`failed to allocate compute pp buffers`). The adopted solution reduces
`--spec-draft-n-ubatch` (MTP buffer 942 → 235 MiB) and reinvests the VRAM in a
larger ubatch.

| depth | PP F1 | PP F1r | Δ% | TG F1 | TG F1r | Δ% | TTFT F1 | TTFT F1r | VRAM max F1 | VRAM max F1r |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1473.36 | **1744.57** | **+18.4** | 46.87 | 50.13 | +7.0 | 5.56 s | **4.70 s** | 9597 | 9541 |
| 32768 | 1426.27 | **1782.44** | **+25.0** | 49.67 | 50.08 | +0.8 | 22.97 s | **18.38 s** | 9570 | 9520 |
| 131072 | 1156.77 | **1395.48** | **+20.6** | 43.05 | 42.30 | −1.7 | 113.31 s | **93.93 s** | 9568 | 9520 |
| 196608 | 1030.62 | **1209.18** | **+17.3** | 34.97 | 32.69 | −6.5 | 190.77 s | **162.60 s** | 9595 | 9878 |

At 192K the remaining VRAM margin is **362 MiB** (10240 − 9878). For this reason we do not go
up to ubatch 3584/4096 nor to `--n-cpu-moe 36`.

Quality: `F1r_agentic` tool calling ok and coding **10/10** (8.82 task/min vs 8.77);
`F1r_session` 29 turns, `prompt_n` 6082–6548, median TTFT 5.66 s (F1: 6.93 s),
median PP 1142.1 (F1: 969.3), TG 37.30 (F1: 37.53), 2 empty responses (F1: 7).

### 1.3 Effect of ReBAR alone (original configuration E00b)

A/B test with the baseline configuration, with ReBAR as the only variable:

| depth | PP E00b | PP rebar | Δ% | TG E00b | TG rebar | Δ% | VRAM max E00b | VRAM max rebar |
|---|---|---|---|---|---|---|---|---|
| 8192 | 445.25 | 416.47 | −6.5 | 31.41 | 26.91 | −14.3 | 9306 | 9904 |
| 32768 | 443.20 | 431.93 | −2.5 | 28.54 | 27.35 | −4.2 | 9294 | 9831 |
| 65536 | 424.63 | 415.06 | −2.3 | 28.48 | 25.30 | −11.2 | 9285 | 9401 |
| 131072 | 398.85 | 388.70 | −2.5 | 28.15 | 27.73 | −1.5 | 9444 | 9863 |

ReBAR **does not raise Gen3 x16 bandwidth** and brings no gains: PP −2.3…−6.5 %, TG equal
or worse, maximum VRAM up to +598 MiB. The cost is free VRAM at start-up lower
by ~300 MiB. Sources: `results/bench_E00b_rebar.jsonl`, `logs/exp/E00b_rebar.log`.

## 2. ubatch table

| ubatch | KV | CUDA0 compute buffer | PP 8K | PP 32K | Outcome |
|---|---|---|---|---|---|
| 512 (E00b) | f16 | 565.50 MiB | 445.25 | 443.20 | reference |
| 1024 (E01) | f16 | 785.01 MiB | 753.93 | 761.32 | +69/+72 % PP; VRAM 9879 |
| 2048 (E02) | f16 | 1352.02 MiB | — | — | **OOM at start-up** (MTP asks for 942 MiB) |
| 2048 (E20) | q8_0 | 1437.02 MiB | 1309.79 | 1267.00 | +66/71 % over E11; VRAM 8939 |
| 4096 (E03) | f16 | — | — | — | OOM |
| 4096 (E21) | q8_0 | — | — | — | OOM (`allocating 1884.12 MiB` failed) |
| 3072 (F1r) | q8_0 | 1973.02 (target) + 235.52 (MTP, `--spec-draft-n-ubatch 512`) | **1744.57** | **1782.44** | **KEPT**; VRAM max 9541 |

Sources: `logs/exp/E00b_baseline.log`, `E01_ub1024.log`, `E02_ub2048.log`,
`E20_kvq8_ub2048.log`, `E21_kvq8_ub4096.log`, `F1r_ub3072.log`.

The PP gain per doubling of ubatch: 512→1024 = +69 %, 1024→2048 = +71 %.
It does not saturate: the limit becomes VRAM (compute buffer + MTP context).
With ReBAR active the MTP context at ubatch 2048 asks for 942 MiB and no longer fits;
`--spec-draft-n-ubatch 512` brings that buffer down to 235.52 MiB and allows ubatch 3072
(PP +18…+25 % over F1). Beyond 3072 the margin at 192K drops below the 362 MiB measured.

## 3. MoE cache table (n_max 2, GEN 512, 4 rep)

| `--moe-cache-mib` | minimum slots per group | hit rate (ubatch ≤ 8) | TG 8K | TG 32K | VRAM max 32K |
|---|---|---|---|---|---|
| 0 (E40) | no cache | n/a | **42.22** | **45.09** | 7960 |
| 1024 (E30) | **17** (q6_K/q4_K) | 15.21 % | 32.29 | 28.44 | 8946 |
| 1536 (E41) | 27 | 29.57 % | 37.48 | 32.24 | 9481 |
| 1792 (E44) | 31 | 35.37 % | 39.02 | 36.34 | 9758 |
| 2048 (E42) | — | — | — | — | **OOM** |

With 1024 MiB the q6_K/q4_K groups have 17 slots < 24 required by the MTP check
((n_max+1)×8 = 24 with n_max 2): those layers leave the cache. Raising the cache to 1536
solves the problem (27 slots) and raises the hit rate, **but TG remains worse than cache 0**:
generation is dominated by the cost of the per-layer lookup/upload, not by the hit rate.

## 4. Table `--n-cpu-moe` vs cache (n_max 2, cache 0 unless stated)

| Configuration | VRAM max 32K | PP 8K | PP 32K | TG 8K | TG 32K |
|---|---|---|---|---|---|
| cache 1024, `--cpu-moe` (E30) | 8946 | 1395.03 | 1351.21 | 32.29 | 28.44 |
| cache 1536, `--cpu-moe` (E41) | 9481 | 1408.88 | 1367.48 | 37.48 | 32.24 |
| cache 0, `--cpu-moe` (E40) | 7960 | 1362.78 | 1323.88 | 42.22 | 45.09 |
| **cache 0, `--n-cpu-moe 37` (E45)** | **9525** | **1468.84** | **1423.32** | **48.49** | **48.66** |
| cache 0, `--n-cpu-moe 36` (E46) | 9829 | 1501.94 | 1450.45 | 48.02 | 49.33 |

`--n-cpu-moe 37` = layers 37–39 with experts in VRAM (`CUDA0 model buffer` 3545.70 MiB vs
1961.70 of the baseline). It is the best solution: resident experts pay neither PCIe
nor the CPU. `--n-cpu-moe 36` gains only 1–2 % with 300 MiB less VRAM
margin → not adopted.

## 5. MTP table (`--spec-draft-n-max`)

| n_max | acc 8K | acc 32K | acc 128K | TG 8K | TG 32K | TG 128K | Outcome |
|---|---|---|---|---|---|---|---|
| 1 (E50) | 0.96 | 0.99 | 0.98 | 46.78 | 44.42 | 37.37 | DISCARDED |
| 2 (E47) | 0.91 | 0.98 | 0.94 | 48.08 | 48.73 | 41.01 | reference |
| **3 (E51)** | 0.92 | 0.97 | 0.97 | **49.74** | **49.91** | **45.18** | **KEPT** |
| 4 (E52) | 0.78 | 0.96 | 0.77 | 45.36 | 51.39 | 39.66 | DISCARDED (acc collapses) |

`--spec-draft-p-min 0.5` (E53): TG 46.24/49.89/43.56 → worse.
Removal of `--spec-draft-cpu-moe` (E54): TG 47.43/49.77/41.83 → worse at 128K.

## 6. Quality

| Test | Baseline | F1 | F2 | F1r | Notes |
|---|---|---|---|---|---|
| `agentic.py` tool_call | ok | ok | ok | ok | |
| `agentic.py` tool_call_stream | ok | ok | ok | ok | |
| `agentic.py` multi_turn | ok | ok | ok | ok | |
| coding (10 checks) | 10/10 | **10/10** | **10/10** | **10/10** | F1: 8.77 task/min; F2: 8.65; F1r: 8.82 |
| needle (5 needles, 150 128 token) | 5/5 (f16) | **5/5** | **5/5** | — | F1r not repeated: only the ubatch changes |

The target's q8_0 KV was promoted after: needle 5/5 (same as f16) and coding 9/10
(vs 10/10, a loss of 1 = maximum limit allowed by the plan) in the intermediate
configuration `E11`.

## 7. Long sessions

| Tag | Turns | Final context | `prompt_n` min–max | Median TTFT | Median PP | Median TG | Errors |
|---|---|---|---|---|---|---|---|
| E00_session (baseline) | 29 | 178 647 | 6082–6548 | 16.86 s | 364.6 | 27.59 | none |
| E30_session_none | 9 | 55 854 | 6082–6548 | 5.54 s | 1111.1 | 30.04 | none |
| **F1_session** | 29 | 177 082 | 6083–6548 | 6.45 s | 969.3 | 36.81 | none |
| **F1_session2** (same server) | 29 | 179 246 | 6082–6100 | 6.45 s | 967.8 | 37.09 | none |
| F2_session | 29 | 177 682 | 6083–6548 | 6.82 s (valid turns) | 925.5 | 42.92 (valid) | none |
| **F1r_session** | 29 | 178 609 | 6082–6548 | **5.66 s** | **1142.1** | 37.30 | none |

In all sessions `prompt_n` stays ≈ 6 085 token: **no full re-processing
of the context** (no cache loss). F1 passes two consecutive sessions without
restarting the server, without errors and without swap growth. `F1r_session` (29 turns)
confirms the stability and improves TTFT by 18 % and PP by 18 %.

## 8. Reproducible restart (F1, two consecutive stop/start)

| Tag | PP 8K | TG 8K | PP 32K | TG 32K | VRAM max |
|---|---|---|---|---|---|
| F1_full (reference) | 1473.36 | 46.87 | 1426.27 | 49.67 | 9597 |
| F1_restart1 | 1473.86 (+0.03 %) | 44.58 (−4.9 %) | 1424.37 (−0.1 %) | 49.09 (−1.2 %) | 9594 |
| F1_restart2 | 1470.11 (−0.2 %) | 46.92 (+0.1 %) | 1425.49 (−0.1 %) | 49.00 (−1.3 %) | 9567 |

All differences are within ±5 % → restart reproducible.

## 9. `results/summary.csv`

Updated with `python3 bench/summarize.py` (all tags, **94 data rows** + header = 95 rows): it contains medians,
TG min/max, MTP acceptance, TTFT, VRAM max, minimum RAM and `n_ctx` for each depth.
