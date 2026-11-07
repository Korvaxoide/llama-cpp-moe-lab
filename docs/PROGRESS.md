# PROGRESS.md — stato del lavoro

Ultimo aggiornamento: 2026-11-07 ~14:10.

## Completato

- [x] **Fase 0** — baseline completa: `E00b_baseline` FULL (8K→192K), `E00_agentic` (10/10), `E00_session` (29 turni). `BASELINE.md` scritto.
- [x] **Fase 1** — ubatch: `E01_ub1024` PP +69 %, `E02_ub2048`/`E03_ub4096` OOM con KV f16. `E01_ub1024_long` regge a 196608.
- [x] **Fase 2** — KV q8_0: `E10_kvq8` (target+draft) scartato per TG; `E11_kvq8_tgt_only` promosso (VRAM −1905 MiB, needle 5/5, coding 9/10).
- [x] **Fase 3** — `E20_kvq8_ub2048` (PP +66/71 %), `E21_kvq8_ub4096` OOM; `E20_..._long` a 196608.
- [x] **Fase 4** — `E30_loadnone`: PP +6.5 %, TG +12.6 % a 8K, RAM pinnata ~14.3 GB; `E31_loadnone_mlock` neutro; `E30_session_none` stabile.
- [x] **Fase 5** — scoperta chiave: **la MoE cache danneggia la generazione** (`E40_mc0` TG +31/+59 %; `E41_mc1536`, `E44_mcMAX` più lente; `E42_mc2048` OOM). `E45_ncmoe37` (3 layer esperti in VRAM) batte tutto: TG 48.5/48.7. `E46_ncmoe36` neutro e troppo vicino al limite VRAM. MTP: `E51_nmax3` tenuto; `E50_nmax1`, `E52_nmax4`, `E53_pmin`, `E54_nodraftmoe` scartati.
- [x] **Fase 6** — `E60_t7`, `E61_t16`, `E64_prio` peggiorano; `E65_poll` neutro; `E62/E63` (offload min batch) peggiorano. Tutti scartati.
- [x] **Fase 7** — pull b12124 riuscito (digest `821d0ae4…`, commit 8bed7cfb9, tutte le flag presenti). `E70_b12124` non batte b11977 (TG 8K −10.4 %): **resta b11977**.
- [x] **Fase 8 (F1)** — `F1_full` completo (PP 1473→1031 t/s, TG 46.9→35.0 t/s), `F1_max` a 199000 token (PP 1021.7, TG 36.7, TTFT 194.8 s), `F1_agentic` **10/10** (8.77 task/min), `F1_needle` **5/5**, `F1_session` + `F1_session2` sullo stesso server (29+29 turni, `prompt_n` stabile, nessun errore), `F1_restart1/2` entro ±5 %.
- [x] **Fase 8 (F2)** — finalista conservativo (mmap, senza `--load-mode none`): FULL, agentic 10/10, needle 5/5, sessione completata. ~5 % più lento in PP ma con ~18.7 GB di RAM liberi: documentato come alternativa.
- [x] **Fase 9** — produzione `qwen36-35b-opt` su `127.0.0.1:8080` con la config F1; `PROD_check` 5/5 con tool call; rollback provato (originale avviato, HTTP 200, fermato, ottimizzato riavviato).
- [x] **Fase 10** — documenti finali scritti: `README.md`, `HARDWARE.md` (banda misurata aggiornata), `BASELINE.md`, `EXPERIMENTS.md`, `RESULTS.md`, `OPTIMIZATION.md`, `PROGRESS.md`.

## Estensione 2026-11-07 — ReBAR e nuovo ubatch (F1r)

- [x] **`E00b_rebar`** — ReBAR abilitato nel BIOS (BAR1 256 → 16384 MiB), test A/B sulla
      configurazione di baseline. Risultato: **nessun guadagno** (PP −6.5 % a 8K, −2.5 % a 32K,
      −2.3 % a 64K, −1.8 % a 128K; TG uguale o peggiore; VRAM max +598 MiB). La banda resta
      Gen3 x16. Il ReBAR costa ~300–450 MiB di VRAM libera all'avvio: con ReBAR la vecchia
      F1 **non parte più** (`failed to allocate compute pp buffers`).
- [x] **`F1r_dub128`** — `--spec-draft-n-ubatch 128`: buffer compute del draft 942.06 →
      **58.88 MiB**. Costo −1.2 % PP. Prova che il buffer MTP scala con l'ubatch del draft.
- [x] **`F1r_ub3072`** — `--spec-draft-n-ubatch 512` + `-b 3072 -ub 3072`: PP **1744.57**
      (8K, +18.4 %), **1782.44** (32K, +25.0 %), **1395.48** (128K, +20.6 %) vs F1; TG
      invariata; VRAM max 9541 MiB; TTFT 4.70 / 18.38 / 93.93 s.
- [x] **`F1r_ub3072_long`** — 196608: PP **1209.18** (+17.3 %), TG 32.69, TTFT 162.6 s,
      VRAM max **9878 MiB** → margine 362 MiB. Nessun OOM.
- [x] **`F1r_agentic`** — tool calling stream/non-stream e multi-turno ok, coding **10/10**,
      8.82 task/min.
- [x] **`F1r_session`** — 29 turni, `prompt_n` 6082–6548, TTFT mediano **5.66 s** (F1: 6.93),
      PP mediano **1142.1** (F1: 969.3), TG 37.30 (F1: 37.53), 2 risposte vuote (F1: 7).
- [x] **Produzione aggiornata** — `scripts/prod_start.sh` ora usa F1r; `qwen36-35b-opt`
      avviato e verificato: `/health` ok, `/v1/models` ok, `n_ctx = 200192`,
      `PROD_f1r_check` coding **10/10**, 8.79 task/min, TG 45.8–52.0 t/s.
- [x] **Documenti aggiornati** — `EXPERIMENTS.md` (6 nuove voci), `RESULTS.md` (§1.2 F1r,
      §1.3 solo-ReBAR, tabella ubatch, qualità, sessioni), `OPTIMIZATION.md` (config F1r
      riga per riga, impronta di memoria, guadagni, limiti, ReBAR), `HARDWARE.md`
      (BAR1 16384, VRAM reale), `README.md` (comandi), `results/summary.csv` rigenerato
      (106 righe).

### Conclusione sul ReBAR

Il ReBAR **non serve** a llama.cpp: non aumenta la banda Gen3 x16 e riduce la VRAM
disponibile. La configurazione F1r funziona sia con ReBAR attivo sia disattivo; se il
ReBAR non serve per il gaming, si può disattivare senza perdite. Il guadagno reale di
questa tornata viene dal disaccoppiamento dell'ubatch MTP (`--spec-draft-n-ubatch 512`)
e dall'ubatch 3072: **PP +17…+25 % su F1, TG invariata**.

## Stato dei servizi (finale)

- **`qwen36-35b-opt`**: **in esecuzione e healthy** su `127.0.0.1:8080` (config **F1r**,
  `scripts/prod_start.sh`).
- **`qwen36-35b`** (originale): **fermo e intatto**, mai cancellato. Rollback: `scripts/prod_stop.sh && bin/dk start qwen36-35b`.
- Immagine b11977: intatta e pinnata per digest. Immagine b12124: scaricata, **non usata** in produzione.
- Nessun container `llopt-*` (sperimentale) rimasto, né in esecuzione né presente.
- Un solo server LLM alla volta: mai avviare `qwen36-35b` e `qwen36-35b-opt` insieme.

## Configurazione di riferimento attuale (F1r)

```
-m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto
--n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe
--spec-draft-n-ubatch 512
-ctk q8_0 -ctv q8_0 -b 3072 -ub 3072 --load-mode none
```

Immagine: `ghcr.io/ggml-org/llama.cpp@sha256:0283744a…` (b11977). `n_ctx = 200192`.
Rispetto a F1 cambiano `--spec-draft-n-ubatch 512`, `-b 3072`, `-ub 3072`.

## Checklist finale (§10 del piano)

- [x] E00b baseline FULL completa (8K→192K)
- [x] Baseline agentica + sessione lunga
- [x] Fase 1 ubatch (1024/2048/4096)
- [x] Fase 2 KV q8_0 + qualità (coding + needle)
- [x] Fase 3 ubatch con KV q8_0
- [x] Fase 4 load-mode none / mlock
- [x] Fase 5.1 MoE cache sweep con hit rate
- [x] Fase 5.2 n-cpu-moe vs cache
- [x] Fase 5.3 MTP n_max 1/3/4 (+ p_min)
- [x] Fase 6 thread / offload min batch / prio / poll
- [x] Fase 7 runtime b12124
- [x] Fase 8 finalisti: FULL, MAX 199K, agentic, needle, sessione ×2, riavvio
- [x] Fase 9 produzione su 127.0.0.1:8080, originale preservato, rollback provato
- [x] Fase 10 tutti i documenti: README, HARDWARE, BASELINE, EXPERIMENTS, RESULTS, OPTIMIZATION, PROGRESS
- [x] Nessun container sperimentale lasciato in esecuzione (`bin/dk ps`)
- [x] Estensione ReBAR: test A/B, `--spec-draft-n-ubatch`, ubatch 3072, produzione F1r
      avviata e verificata (2026-11-07)

**Lavoro completato.**
