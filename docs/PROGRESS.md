# PROGRESS.md — work status

Last update: 2026-11-07 ~14:10.

## Completed

- [x] **Phase 0** — full baseline: `E00b_baseline` FULL (8K→192K), `E00_agentic` (10/10), `E00_session` (29 turns). `BASELINE.md` written.
- [x] **Phase 1** — ubatch: `E01_ub1024` PP +69 %, `E02_ub2048`/`E03_ub4096` OOM with KV f16. `E01_ub1024_long` holds at 196608.
- [x] **Phase 2** — KV q8_0: `E10_kvq8` (target+draft) DISCARDED for TG; `E11_kvq8_tgt_only` promoted (VRAM −1905 MiB, needle 5/5, coding 9/10).
- [x] **Phase 3** — `E20_kvq8_ub2048` (PP +66/71 %), `E21_kvq8_ub4096` OOM; `E20_..._long` at 196608.
- [x] **Phase 4** — `E30_loadnone`: PP +6.5 %, TG +12.6 % at 8K, pinned RAM ~14.3 GB; `E31_loadnone_mlock` NEUTRAL; `E30_session_none` stable.
- [x] **Phase 5** — key discovery: **the MoE cache hurts generation** (`E40_mc0` TG +31/+59 %; `E41_mc1536`, `E44_mcMAX` slower; `E42_mc2048` OOM). `E45_ncmoe37` (3 expert layers in VRAM) beats everything: TG 48.5/48.7. `E46_ncmoe36` NEUTRAL and too close to the VRAM limit. MTP: `E51_nmax3` KEPT; `E50_nmax1`, `E52_nmax4`, `E53_pmin`, `E54_nodraftmoe` DISCARDED.
- [x] **Phase 6** — `E60_t7`, `E61_t16`, `E64_prio` get worse; `E65_poll` NEUTRAL; `E62/E63` (min batch offload) get worse. All DISCARDED.
- [x] **Phase 7** — b12124 pull successful (digest `821d0ae4…`, commit 8bed7cfb9, all flags present). `E70_b12124` does not beat b11977 (TG 8K −10.4 %): **b11977 stays**.
- [x] **Phase 8 (F1)** — `F1_full` complete (PP 1473→1031 t/s, TG 46.9→35.0 t/s), `F1_max` at 199000 tokens (PP 1021.7, TG 36.7, TTFT 194.8 s), `F1_agentic` **10/10** (8.77 task/min), `F1_needle` **5/5**, `F1_session` + `F1_session2` on the same server (29+29 turns, `prompt_n` stable, no errors), `F1_restart1/2` within ±5 %.
- [x] **Phase 8 (F2)** — conservative finalist (mmap, without `--load-mode none`): FULL, agentic 10/10, needle 5/5, session completed. ~5 % slower in PP but with ~18.7 GB of free RAM: documented as an alternative.
- [x] **Phase 9** — production `qwen36-35b-opt` on `127.0.0.1:8080` with the F1 config; `PROD_check` 5/5 with tool call; rollback tried (original started, HTTP 200, stopped, optimized restarted).
- [x] **Phase 10** — final documents written: `README.md`, `HARDWARE.md` (updated measured bandwidth), `BASELINE.md`, `EXPERIMENTS.md`, `RESULTS.md`, `OPTIMIZATION.md`, `PROGRESS.md`.

## Extension 2026-11-07 — ReBAR and new ubatch (F1r)

- [x] **`E00b_rebar`** — ReBAR enabled in the BIOS (BAR1 256 → 16384 MiB), A/B test on the
      baseline configuration. Result: **no gain** (PP −6.5 % at 8K, −2.5 % at 32K,
      −2.3 % at 64K, −1.8 % at 128K; TG equal or worse; VRAM max +598 MiB). The bandwidth stays
      Gen3 x16. ReBAR costs ~300–450 MiB of free VRAM at startup: with ReBAR the old
      F1 **no longer starts** (`failed to allocate compute pp buffers`).
- [x] **`F1r_dub128`** — `--spec-draft-n-ubatch 128`: draft compute buffer 942.06 →
      **58.88 MiB**. Cost −1.2 % PP. Evidence that the MTP buffer scales with the draft ubatch.
- [x] **`F1r_ub3072`** — `--spec-draft-n-ubatch 512` + `-b 3072 -ub 3072`: PP **1744.57**
      (8K, +18.4 %), **1782.44** (32K, +25.0 %), **1395.48** (128K, +20.6 %) vs F1; TG
      unchanged; VRAM max 9541 MiB; TTFT 4.70 / 18.38 / 93.93 s.
- [x] **`F1r_ub3072_long`** — 196608: PP **1209.18** (+17.3 %), TG 32.69, TTFT 162.6 s,
      VRAM max **9878 MiB** → margin 362 MiB. No OOM.
- [x] **`F1r_agentic`** — tool calling stream/non-stream and multi-turn ok, coding **10/10**,
      8.82 task/min.
- [x] **`F1r_session`** — 29 turns, `prompt_n` 6082–6548, median TTFT **5.66 s** (F1: 6.93),
      median PP **1142.1** (F1: 969.3), TG 37.30 (F1: 37.53), 2 empty replies (F1: 7).
- [x] **Production updated** — `scripts/prod_start.sh` now uses F1r; `qwen36-35b-opt`
      started and verified: `/health` ok, `/v1/models` ok, `n_ctx = 200192`,
      `PROD_f1r_check` coding **10/10**, 8.79 task/min, TG 45.8–52.0 t/s.
- [x] **Documents updated** — `EXPERIMENTS.md` (6 new entries), `RESULTS.md` (§1.2 F1r,
      §1.3 ReBAR-only, ubatch table, quality, sessions), `OPTIMIZATION.md` (F1r config
      line by line, memory footprint, gains, limits, ReBAR), `HARDWARE.md`
      (BAR1 16384, real VRAM), `README.md` (commands), `results/summary.csv` regenerated
      (106 rows).

### Conclusion on ReBAR

ReBAR **is not needed** by llama.cpp: it does not increase Gen3 x16 bandwidth and reduces
available VRAM. The F1r configuration works both with ReBAR active and disabled; if
ReBAR is not needed for gaming, it can be disabled without losses. The real gain of
this round comes from decoupling the MTP ubatch (`--spec-draft-n-ubatch 512`)
and from ubatch 3072: **PP +17…+25 % on F1, TG unchanged**.

## Service status (final)

- **`qwen36-35b-opt`**: **running and healthy** on `127.0.0.1:8080` (config **F1r**,
  `scripts/prod_start.sh`).
- **`qwen36-35b`** (original): **stopped and intact**, never deleted. Rollback: `scripts/prod_stop.sh && bin/dk start qwen36-35b`.
- Image b11977: intact and pinned by digest. Image b12124: downloaded, **not used** in production.
- No `llopt-*` (experimental) container left, neither running nor present.
- A single LLM server at a time: never start `qwen36-35b` and `qwen36-35b-opt` together.

## Current reference configuration (F1r)

```
-m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto
--n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe
--spec-draft-n-ubatch 512
-ctk q8_0 -ctv q8_0 -b 3072 -ub 3072 --load-mode none
```

Image: `ghcr.io/ggml-org/llama.cpp@sha256:0283744a…` (b11977). `n_ctx = 200192`.
Compared to F1, `--spec-draft-n-ubatch 512`, `-b 3072`, `-ub 3072` change.

## Final checklist (§10 of the plan)

- [x] E00b baseline FULL complete (8K→192K)
- [x] Agentic baseline + long session
- [x] Phase 1 ubatch (1024/2048/4096)
- [x] Phase 2 KV q8_0 + quality (coding + needle)
- [x] Phase 3 ubatch with KV q8_0
- [x] Phase 4 load-mode none / mlock
- [x] Phase 5.1 MoE cache sweep with hit rate
- [x] Phase 5.2 n-cpu-moe vs cache
- [x] Phase 5.3 MTP n_max 1/3/4 (+ p_min)
- [x] Phase 6 threads / min batch offload / prio / poll
- [x] Phase 7 runtime b12124
- [x] Phase 8 finalists: FULL, MAX 199K, agentic, needle, session ×2, restart
- [x] Phase 9 production on 127.0.0.1:8080, original preserved, rollback tried
- [x] Phase 10 all documents: README, HARDWARE, BASELINE, EXPERIMENTS, RESULTS, OPTIMIZATION, PROGRESS
- [x] No experimental container left running (`bin/dk ps`)
- [x] ReBAR extension: A/B test, `--spec-draft-n-ubatch`, ubatch 3072, production F1r
      started and verified (2026-11-07)

**Work completed.**
