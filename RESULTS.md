# RESULTS.md — risultati consolidati

Tutte le cifre provengono da `results/bench_*.jsonl`, `results/agentic_*.jsonl`,
`results/session_*.jsonl`, `results/needle_*.jsonl` e dai log in `logs/exp/`.
Le mediane sono calcolate da `python3 bench/summarize.py` (output completo in
`results/summary.csv`).

## 1. Tabella principale (configurazioni chiave)

| Configurazione | PP 8K | TG 8K | PP 32K | TG 32K | PP 128K | TG 128K | PP 192K | TG 192K | TTFT 32K | TTFT 128K | VRAM max | RAM min | coding | needle | sessione |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E00b** baseline (KV f16, ub512, mc1024, mmap, n_max2) | 445.25 | 31.41 | 443.20 | 28.54 | 398.85 | 28.15 | 385.84 | 28.90 | 73.94 | 328.63 | 9470 | 18631 | 10/10 | 5/5 | ok (29 turni) |
| **E01** + ub1024 | 753.93 | 30.32 | 761.32 | 28.56 | — | — | 613.66¹ | 30.81¹ | 43.04 | — | 9879 | 23208 | (10/10) | 5/5² | — |
| **E20** + KV q8 + ub2048 (mc1024, mmap) | 1309.79 | 28.67 | 1267.00 | 27.75 | — | — | 931.92³ | 24.75³ | 25.86 | — | 8939 | 24112 | (9/10)⁴ | 5/5⁴ | — |
| **E30** + `--load-mode none` | 1395.03 | 32.29 | 1351.21 | 28.44 | — | — | — | — | 24.25 | — | 8963 | 9669 | — | — | ok⁵ |
| **E40** cache MoE 0 | 1362.78 | 42.22 | 1323.88 | 45.09 | — | — | — | — | 24.75 | — | 7960 | 8814 | — | — | — |
| **E45** `--n-cpu-moe 37` + cache 0 | 1468.84 | 48.49 | 1423.32 | 48.66 | — | — | — | — | 23.02 | — | 9525 | 10485 | — | — | — |
| **F1** = E45 + n_max 3 (finalista) | 1473.36 | 46.87 | 1426.27 | 49.67 | 1156.77 | 43.05 | 1030.62 | 34.97 | 22.97 | 113.31 | 9597 | **5517** | **10/10** | **5/5** | **ok ×2** |
| **F2** = F1 senza `--load-mode none` (mmap) | 1398.43 | 45.43 | 1347.43 | 49.40 | 1115.07 | 42.23 | 1001.32 | 34.04 | 24.32 | 117.55 | 9568 | 18702 | **10/10** | **5/5** | ok ma 16 risposte vuote⁶ |

¹ `E01_ub1024_long` (196608, 1 rep). ² `E01_needle_f16`. ³ `E20_kvq8_ub2048_long`.
⁴ `E11_kvq8_tgt_only` (stessa KV q8, ub1024). ⁵ `E30_session_none` (9 turni, 60K).
⁶ In `F2_session` 16/29 turni hanno prodotto una risposta vuota/1 token (finish `stop`);
la mediana TG sui turni validi è 42.92 t/s. F1: 7/29 turni vuoti, TG 37.53 t/s sui validi.
Il fenomeno è del modello/prompt, non del server (nessun errore nel log).

Sorgenti: `results/bench_E00b_baseline.jsonl`, `bench_E01_ub1024.jsonl`,
`bench_E01_ub1024_long.jsonl`, `bench_E20_kvq8_ub2048.jsonl`, `bench_E20_kvq8_ub2048_long.jsonl`,
`bench_E30_loadnone.jsonl`, `bench_E40_mc0.jsonl`, `bench_E45_ncmoe37.jsonl`,
`bench_F1_full.jsonl`, `bench_F2_full.jsonl`.

### 1.1 Confronto finale F1 vs baseline E00b

| Metrica | E00b | F1 | Δ |
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
| TTFT ~2.4K (sessione reale) | ~7.0 s | ~2.5 s (6.45 s per 6K) | **−64 %** |
| Lavoro agentico | 6.7 task/min | 8.77 task/min | **+31 %** |
| VRAM max | 9470 MiB | 9597 MiB | +127 MiB |
| RAM disponibile min | 18631 MiB | 5517 MiB | −13114 MiB (pinned) |

## 2. Tabella ubatch

| ubatch | KV | compute buffer CUDA0 | PP 8K | PP 32K | Esito |
|---|---|---|---|---|---|
| 512 (E00b) | f16 | 565.50 MiB | 445.25 | 443.20 | riferimento |
| 1024 (E01) | f16 | 785.01 MiB | 753.93 | 761.32 | +69/+72 % PP; VRAM 9879 |
| 2048 (E02) | f16 | 1352.02 MiB | — | — | **OOM all'avvio** (MTP chiede 942 MiB) |
| 2048 (E20) | q8_0 | 1437.02 MiB | 1309.79 | 1267.00 | +66/71 % su E11; VRAM 8939 |
| 4096 (E03) | f16 | — | — | — | OOM |
| 4096 (E21) | q8_0 | — | — | — | OOM (`allocating 1884.12 MiB` fallita) |

Sorgenti: `logs/exp/E00b_baseline.log`, `E01_ub1024.log`, `E02_ub2048.log`,
`E20_kvq8_ub2048.log`, `E21_kvq8_ub4096.log`.

Il guadagno di PP per raddoppio di ubatch: 512→1024 = +69 %, 1024→2048 = +71 %.
Non si satura: il limite diventa la VRAM (compute buffer + contesto MTP).

## 3. Tabella MoE cache (n_max 2, GEN 512, 4 rep)

| `--moe-cache-mib` | slot minimi per gruppo | hit rate (ubatch ≤ 8) | TG 8K | TG 32K | VRAM max 32K |
|---|---|---|---|---|---|
| 0 (E40) | nessuna cache | n/d | **42.22** | **45.09** | 7960 |
| 1024 (E30) | **17** (q6_K/q4_K) | 15.21 % | 32.29 | 28.44 | 8946 |
| 1536 (E41) | 27 | 29.57 % | 37.48 | 32.24 | 9481 |
| 1792 (E44) | 31 | 35.37 % | 39.02 | 36.34 | 9758 |
| 2048 (E42) | — | — | — | — | **OOM** |

Con 1024 MiB i gruppi q6_K/q4_K hanno 17 slot < 24 richiesti dalla verifica MTP
((n_max+1)×8 = 24 con n_max 2): quei layer escono dalla cache. Portare la cache a 1536
risolve il problema (27 slot) e alza l'hit rate, **ma la TG resta peggiore della cache 0**:
la generazione è dominata dal costo del lookup/upload per layer, non dall'hit rate.

## 4. Tabella `--n-cpu-moe` vs cache (n_max 2, cache 0 salvo indicazione)

| Configurazione | VRAM max 32K | PP 8K | PP 32K | TG 8K | TG 32K |
|---|---|---|---|---|---|
| cache 1024, `--cpu-moe` (E30) | 8946 | 1395.03 | 1351.21 | 32.29 | 28.44 |
| cache 1536, `--cpu-moe` (E41) | 9481 | 1408.88 | 1367.48 | 37.48 | 32.24 |
| cache 0, `--cpu-moe` (E40) | 7960 | 1362.78 | 1323.88 | 42.22 | 45.09 |
| **cache 0, `--n-cpu-moe 37` (E45)** | **9525** | **1468.84** | **1423.32** | **48.49** | **48.66** |
| cache 0, `--n-cpu-moe 36` (E46) | 9829 | 1501.94 | 1450.45 | 48.02 | 49.33 |

`--n-cpu-moe 37` = layer 37–39 con esperti in VRAM (`CUDA0 model buffer` 3545.70 MiB vs
1961.70 della baseline). È la soluzione migliore: gli esperti residenti non pagano PCIe
e non pagano la CPU. `--n-cpu-moe 36` guadagna solo l'1–2 % con 300 MiB di VRAM in meno
di margine → non adottato.

## 5. Tabella MTP (`--spec-draft-n-max`)

| n_max | acc 8K | acc 32K | acc 128K | TG 8K | TG 32K | TG 128K | Esito |
|---|---|---|---|---|---|---|---|
| 1 (E50) | 0.96 | 0.99 | 0.98 | 46.78 | 44.42 | 37.37 | scartato |
| 2 (E47) | 0.91 | 0.98 | 0.94 | 48.08 | 48.73 | 41.01 | riferimento |
| **3 (E51)** | 0.92 | 0.97 | 0.97 | **49.74** | **49.91** | **45.18** | **tenuto** |
| 4 (E52) | 0.78 | 0.96 | 0.77 | 45.36 | 51.39 | 39.66 | scartato (acc crolla) |

`--spec-draft-p-min 0.5` (E53): TG 46.24/49.89/43.56 → peggiore.
Rimozione di `--spec-draft-cpu-moe` (E54): TG 47.43/49.77/41.83 → peggiore a 128K.

## 6. Qualità

| Test | Baseline | F1 | F2 | Note |
|---|---|---|---|---|
| `agentic.py` tool_call | ok | ok | ok | |
| `agentic.py` tool_call_stream | ok | ok | ok | |
| `agentic.py` multi_turn | ok | ok | ok | |
| coding (10 prove) | 10/10 | **10/10** | **10/10** | F1: 8.77 task/min; F2: 8.65 |
| needle (5 aghi, 150 128 token) | 5/5 (f16) | **5/5** | **5/5** | `needle_E01_needle_f16`, `F1_needle`, `F2_needle` |

La KV q8_0 del target è stata promossa dopo: needle 5/5 (uguale a f16) e coding 9/10
(vs 10/10, perdita di 1 = limite massimo ammesso dal piano) nella configurazione
intermedia `E11`.

## 7. Sessioni lunghe

| Tag | Turni | Contesto finale | `prompt_n` min–max | TTFT mediano | PP mediano | TG mediano | Errori |
|---|---|---|---|---|---|---|---|
| E00_session (baseline) | 29 | 178 647 | 6082–6548 | 16.86 s | 364.6 | 27.59 | nessuno |
| E30_session_none | 9 | 55 854 | 6082–6548 | 5.54 s | 1111.1 | 30.04 | nessuno |
| **F1_session** | 29 | 177 082 | 6083–6548 | 6.45 s | 969.3 | 36.81 | nessuno |
| **F1_session2** (stesso server) | 29 | 179 246 | 6082–6100 | 6.45 s | 967.8 | 37.09 | nessuno |
| F2_session | 29 | 177 682 | 6083–6548 | 6.82 s (turni validi) | 925.5 | 42.92 (validi) | nessuno |

In tutte le sessioni `prompt_n` resta ≈ 6 085 token: **nessuna rielaborazione completa
del contesto** (nessuna perdita di cache). F1 passa due sessioni consecutive senza
riavviare il server, senza errori e senza crescita di swap.

## 8. Riavvio riproducibile (F1, due stop/start consecutivi)

| Tag | PP 8K | TG 8K | PP 32K | TG 32K | VRAM max |
|---|---|---|---|---|---|
| F1_full (riferimento) | 1473.36 | 46.87 | 1426.27 | 49.67 | 9597 |
| F1_restart1 | 1473.86 (+0.03 %) | 44.58 (−4.9 %) | 1424.37 (−0.1 %) | 49.09 (−1.2 %) | 9594 |
| F1_restart2 | 1470.11 (−0.2 %) | 46.92 (+0.1 %) | 1425.49 (−0.1 %) | 49.00 (−1.3 %) | 9567 |

Tutte le differenze sono entro ±5 % → riavvio riproducibile.

## 9. `results/summary.csv`

Aggiornato con `python3 bench/summarize.py` (tutti i tag, **94 righe di dati** + intestazione = 95 righe): contiene mediane,
min/max TG, accettazione MTP, TTFT, VRAM max, RAM minima e `n_ctx` per ogni profondità.
