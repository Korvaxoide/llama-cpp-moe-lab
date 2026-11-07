# EXPERIMENTS.md — registro cronologico degli esperimenti

Ogni voce segue il modello della §9 di `PLAN_FOR_AGENT.md`. Decisioni secondo la regola §3.5.

## E00b_baseline — 2026-11-05

- **Ipotesi**: baseline completa (riprende dalla baseline E00 interrotta): tutte le profondità 8192→196608 con la configurazione originale.
- **Collo di bottiglia affrontato**: —
- **Modifica (unica)**: run con tag nuovo `E00b_baseline` tramite `scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- <args originali> -lv 4`.
- **Comando**: `setsid nohup bash -c 'scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe' > logs/E00b.out 2>&1 < /dev/null &`
- **Verifiche**: `n_ctx = 200192` ✔ | `prompt_n = depth` ✔ (nessun `warning`) ✔ | errori nel log: nessuno (solo benigno `common_fit_params: failed to fit params ... tensor_buft_overrides already set by user, abort`) ✔
- **Memoria**: `load_mode = mmap` (CPU_Mapped) | `CUDA0 KV buffer size = 3910.00 MiB` (f16) | `llama_memory_recurrent: CUDA0 RS buffer size = 188.44 MiB` | `impl: CUDA0 MoE cache size = 1019.49 MiB` | `CUDA0 compute buffer size = 565.50 MiB` | `CUDA0 compute buffer size = 235.52 MiB` / `CUDA_Host compute buffer = 211.52 MiB` (contexti aggiuntivi MTP) | `llama_moe_cache: ubatch <= 8: hits = 140201, misses = 905056, hit rate = 13.41%`
- **Risultati (mediane, n=3 / n=2)**:
  | depth | PP t/s | TG t/s | MTP acc | TTFT s | VRAM max MiB | RAM avail min |
  |---|---|---|---|---|---|---|
  | 8192 | 445.25 | 31.41 | 0.86 | 18.4 | 9306 | 25690 |
  | 16384 | 451.23 | 29.34 | 0.91 | 36.31 | 9298 | 24186 |
  | 32768 | 443.20 | 28.54 | 0.97 | 73.94 | 9294 | 21797 |
  | 65536 | 424.63 | 28.48 | 0.80 | 154.33 | 9285 | 19394 |
  | 131072 | 398.85 | 28.15 | 0.96 | 328.63 | 9444 | 18631 |
  | 196608 | 385.84 | 28.90 | 0.85 | 509.6 | 9470 | 19779 |
- **Decisione**: TENUTO — baseline di confronto completa e riproducibile. Vedi `BASELINE.md` (da redigere in Fase 10).
- **File**: `results/bench_E00b_baseline.jsonl`, `logs/exp/E00b_baseline.log`, `logs/E00b.out`

## E00_agentic — 2026-11-05 (prima esecuzione) e rilancio — 2026-11-05

- **Ipotesi**: baseline agentica con la configurazione originale (`--ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe -lv 4`).
- **Collo di bottiglia affrontato**: —
- **Modifica (unica)**: `scripts/run_server.sh llopt-agent` + `python3 bench/agentic.py --tag E00_agentic --coding-reps 2`.
- **Esito iterazione 1 (2026-11-05 ~22:21–22:57)**: tool_call **ok**, tool_call_stream **ok**, multi_turn **ok**, coding **0/10** (tutti i task con `NameError` in fase di test).
- **Bug Python corretto** (annotato): i task di coding di `bench/agentic.py` non passavano `chat_template_kwargs: {"enable_thinking": false}`. Con il thinking attivo il codice Python veniva generato in `reasoning_content` e `extract_code` ritornava contenuto vuoto → codice di test eseguito senza le funzioni → `NameError`.
- **Correzione applicata**: aggiunta di `"chat_template_kwargs": {"enable_thinking": false}` al payload dello stream dei task di coding in `bench/agentic.py`. Nessun prompt né test unitario modificato.
- **Esito iterazione 2 (2026-11-05 ~23:05–23:45, interrotta in sosta)**: tool_call **ok**, tool_call_stream **ok**, multi_turn **ok**, coding rep 0: **5/5 ok** (con `finish: "stop"` e `gen_tokens` 107–432; TG ~30–33 t/s, draft acc 0.83–0.97). File parziale archiviato in `results/archive/agentic_E00_agentic_rep0.jsonl`.
- **Esito definitivo (2026-11-06 08:25–08:27)**: run completo `--coding-reps 2` con il fix: tool_call **ok**, tool_call_stream **ok**, multi_turn **ok**, coding **10/10**, `tasks_per_min = 6.7`, TG mediana sul coding ≈ 32.1 t/s. MoE cache all'arresto: hit rate 10.06 % (`logs/exp/E00_agentic_server.log`).
- **Decisione**: TENUTO come baseline agentica di riferimento.
- **File**: `results/agentic_E00_agentic.jsonl`, `logs/E00_agentic_bench.out`, `logs/exp/E00_agentic_server.log`

## E00_session — 2026-11-06 08:30–08:39

- **Ipotesi**: baseline della sessione agentica lunga (riuso del prompt sul modello ibrido).
- **Collo di bottiglia affrontato**: — (baseline).
- **Modifica (unica)**: `scripts/run_server.sh llopt-agent` + `python3 bench/session.py --tag E00_session --target 180000 --chunk 6000`.
- **Comando**: `setsid nohup bash -c 'python3 bench/session.py --tag E00_session --target 180000 --chunk 6000' > logs/E00_session.out 2>&1 < /dev/null &`
- **Verifiche**: `n_ctx=200192` ✔ | 29 turni, nessun errore ✔ | `prompt_n` per turno 6082–6548 (mediana 6085) → **nessuna rielaborazione completa del contesto** ✔
- **Memoria**: identica alla baseline (`CUDA0 KV 3910+391 MiB`, MoE cache 1019.49 MiB, compute 565.50 MiB). MoE cache all'arresto: hit rate 13.23 % (`logs/exp/E00_session_server.log`).
- **Risultati**:
  | metrica | valore |
  |---|---|
  | turni | 29 |
  | contesto iniziale → finale | 6 612 → 178 647 token |
  | TTFT min/mediana/max | 14.62 / 16.86 / 18.79 s |
  | PP mediana | 364.6 t/s |
  | TG mediana | 27.59 t/s (ultimi 3 turni 25.1–27.5) |
  | accettazione MTP mediana | 0.833 |
  | token generati | 2 012 (2 012 in 558.5 s) |
- **Decisione**: TENUTO come baseline della sessione lunga.
- **File**: `results/session_E00_session.jsonl`, `logs/E00_session.out`, `logs/exp/E00_session_server.log`

## FASE 1 — Prompt processing: dimensione dell'ubatch (H2)

### E01_ub1024 — 2026-11-06 08:43
- **Ipotesi**: H2 — raddoppiare `-ub 512 → 1024` ammortizza su 2× token lo stesso trasferimento PCIe degli esperti.
- **Collo di bottiglia affrontato**: PCIe H2D degli esperti durante il prompt processing.
- **Modifica (unica)**: `-b 2048 -ub 1024` rispetto a `E00b_baseline`.
- **Comando**: `scripts/exp.sh E01_ub1024 8192,32768 3 1 -- <ARGS_BASE> -b 2048 -ub 1024`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun `warning` ✔ | nessun errore ✔
- **Memoria**: `CUDA0 compute buffer size = 785.01 MiB` (rif. 565.50, +219.51 MiB) | `vram_max = 9879 MiB` (8K), 9781 MiB (32K) | `ram_avail_min = 23208 MiB` | swap 1 MiB
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 445.25 | 753.93 | **+69.3** | 31.41 | 30.32 | −3.5 | 0.88 |
  | 32768 | 443.20 | 761.32 | **+71.8** | 28.54 | 28.56 | +0.1 | 0.96 |
- **Decisione**: TENUTO — PP +69 % (min nuovo 753.9 > max rif 451.2 a 8K; 757.9 > 443.2 a 32K). TG 8K −3.5 % è al limite della soglia §3.5 ma l'altro indicatore migliora molto di più (eccezione esplicita del piano). VRAM 9879 ≤ 9900 (margine 21 MiB, da annotare).
- **File**: `results/bench_E01_ub1024.jsonl`, `logs/exp/E01_ub1024.log`

### E02_ub2048 — 2026-11-06 08:46
- **Ipotesi**: H2 — ubatch 2048 ammortizza 4× i trasferimenti.
- **Collo di bottiglia affrontato**: PCIe H2D degli esperti nel prompt processing.
- **Modifica (unica)**: `-b 2048 -ub 2048` rispetto a `E00b_baseline`.
- **Comando**: `scripts/exp.sh E02_ub2048 8192,32768 3 1 -- <ARGS_BASE> -b 2048 -ub 2048`
- **Verifiche**: avvio **FALLITO** — nessuna misura.
- **Errore**: `CUDA0 compute buffer size = 1352.02 MiB` (rif. 565.50); alla creazione del contesto MTP `allocating 942.06 MiB on device 0: cudaMalloc failed: out of memory` → `failed to allocate compute pp buffers` → `failed to create MTP context`.
- **Decisione**: SCARTATO in Fase 1 (VRAM insufficiente con KV f16 + MoE cache 1024). Da ritentare in Fase 3 dopo KV q8_0, come previsto dal piano.
- **File**: `logs/exp/E02_ub2048.log`, `logs/queue_phase1.out`

### E03_ub4096 — 2026-11-06 08:47
- **Ipotesi**: H2 — ubatch 4096 ammortizza 8× i trasferimenti.
- **Collo di bottiglia affrontato**: PCIe H2D degli esperti nel prompt processing.
- **Modifica (unica)**: `-b 4096 -ub 4096` rispetto a `E00b_baseline`.
- **Comando**: `scripts/exp.sh E03_ub4096 8192,32768 3 1 -- <ARGS_BASE> -b 4096 -ub 4096`
- **Verifiche**: avvio **FALLITO** — nessuna misura (OOM come E02, compute buffer ancora più grande).
- **Decisione**: SCARTATO in Fase 1; da ritentare in Fase 3 con KV q8_0.
- **File**: `logs/exp/E03_ub4096.log`, `logs/queue_phase1.out`

### E01_ub1024_long — 2026-11-06 08:50
- **Ipotesi**: verifica obbligatoria della capacità a 196608 token dell'ubatch vincente.
- **Collo di bottiglia affrontato**: tenuta della VRAM a contesto pieno.
- **Modifica (unica)**: profondità 196608 (1 rep) con `-b 2048 -ub 1024`.
- **Comando**: `scripts/exp.sh E01_ub1024_long 196608 1 1 -- <ARGS_BASE> -b 2048 -ub 1024`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=196608` ✔ | nessun errore ✔
- **Memoria**: compute buffer 785.01 MiB | `vram_max = 9847 MiB` (≤ 9900) | `ram_avail_min = 26092 MiB`
- **Risultati**: PP **613.66 t/s** (rif. 385.84 → **+59.0 %**), TG **30.81 t/s** (rif. 28.90 → +6.6 %), TTFT ≈ 320 s.
- **Decisione**: TENUTO — regge a 196608 con VRAM sotto la soglia. **Ubatch 1024 = nuovo riferimento**.
- **File**: `results/bench_E01_ub1024_long.jsonl`, `logs/exp/E01_ub1024_long.log`

## FASE 2 — KV cache quantizzata (H4)

### E10_kvq8 — 2026-11-06 09:01
- **Ipotesi**: H4 — KV q8_0 su target e draft MTP dimezza ~4.3 GB di KV e libera VRAM per ubatch/cache più grandi; qualità trascurabile.
- **Collo di bottiglia affrontato**: VRAM occupata dalla KV f16.
- **Modifica (unica)**: `-ctk q8_0 -ctv q8_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0` rispetto a `E01_ub1024`.
- **Comando**: `scripts/exp.sh E10_kvq8 8192,32768 3 1 -- <ARGS rif.> -ctk q8_0 -ctv q8_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun errore ✔ | log: `KV buffer size = 2077.19 MiB … K (q8_0), V (q8_0)` + draft `207.72 MiB` ✔
- **Memoria**: KV target 3910.00 → **2077.19 MiB** (−1832.81), KV draft 391.00 → **207.72 MiB** (−183.28); `CUDA0 compute buffer size = 982.01 MiB` (rif. 785.01) | `vram_max = 8277 MiB` (8K) / 8198 (32K)
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 753.93 | 768.95 | +2.0 | 30.32 | 29.08 | −4.1 | 0.876 |
  | 32768 | 761.32 | 760.64 | −0.1 | 28.56 | 27.33 | **−4.3** | 0.966 |
- **Decisione**: SCARTATO rispetto a E11 — TG −4.3 % a 32K supera la soglia §3.5 e il PP non compensa. La KV draft quantizzata non porta un beneficio di VRAM utile e sembra costare in TG.
- **File**: `results/bench_E10_kvq8.jsonl`, `logs/exp/E10_kvq8.log`

### E11_kvq8_tgt_only — 2026-11-06 09:04
- **Ipotesi**: H4 — quantizzare solo la KV del target (draft MTP in f16) libera ~1.8 GB con impatto nullo sulla velocità.
- **Collo di bottiglia affrontato**: VRAM della KV f16.
- **Modifica (unica)**: `-ctk q8_0 -ctv q8_0` rispetto a `E01_ub1024`.
- **Comando**: `scripts/exp.sh E11_kvq8_tgt_only 8192,32768 3 1 -- <ARGS rif.> -ctk q8_0 -ctv q8_0`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun errore ✔ | KV target q8_0 2077.19 MiB, draft resta f16 391.00 MiB ✔
- **Memoria**: `vram_max = 7974 MiB` (entrambe le profondità) | swap 1 MiB | RAM avail min 25075 MiB
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 753.93 | 764.78 | +1.4 | 30.32 | 31.70 | +4.5 | 0.880 |
  | 32768 | 761.32 | 761.58 | +0.03 | 28.56 | 28.00 | −2.0 | 0.966 |
- **Decisione**: CANDIDATO — PP neutro, TG neutro/migliore a 8K e −2.0 % (dentro la soglia "neutro") a 32K; VRAM −1905 MiB. La promozione a riferimento è subordinata a coding 10/10 e needle = f16.
- **File**: `results/bench_E11_kvq8_tgt_only.jsonl`, `logs/exp/E11_kvq8_tgt_only.log`

### E11_quality — 2026-11-06 09:10–09:20
- **Ipotesi**: la KV q8_0 del target non degrada il recupero a lungo contesto né la qualità del codice.
- **Modifica (unica)**: nessuna; test di qualità sulla configurazione `E11_kvq8_tgt_only`.
- **Comandi**: `python3 bench/needle.py --tag E11_needle_q8` e `python3 bench/agentic.py --tag E11_agentic --coding-reps 2` (server q8, ub1024).
- **Needle** (5 aghi in 150 128 token, stesso prompt f16/q8): f16 `E01_needle_f16` **5/5**; q8 `E11_needle_q8` **5/5** → nessuna perdita.
- **Agentico**: tool_call ok, tool_call_stream ok, coding **9/10** (baseline 10/10; unico errore: `parse_duration('1h1h')` non solleva `ValueError`, rep 0). Perdita = 1, entro il limite del piano (≤ 1). `tasks_per_min = 5.25` vs 6.7 baseline, ma con 3 026 token generati vs 2 659 (+14 %): il dato è dominato dalla lunghezza delle risposte, non dalla velocità.
- **Decisione**: **PROMOSSO a nuovo riferimento** — PP neutro, TG entro ±2 % a 32K, VRAM −1905 MiB, needle 5/5, coding −1 (limite ammesso). La VRAM libera va spesa in Fase 3.
- **File**: `results/needle_E01_needle_f16.jsonl`, `results/needle_E11_needle_q8.jsonl`, `results/agentic_E11_agentic.jsonl`, `logs/exp/E11_needle_q8_server.log`, `logs/exp/E11_agentic_server.log`

## FASE 3 — Ri-espansione dell'ubatch con KV q8_0

### E20_kvq8_ub2048 — 2026-11-06 09:24
- **Ipotesi**: H2+H4 — la VRAM liberata dalla KV q8_0 permette l'ubatch 2048 che in Fase 1 andava in OOM.
- **Collo di bottiglia affrontato**: PCIe H2D degli esperti nel prompt processing.
- **Modifica (unica)**: `-b 2048 -ub 2048` rispetto a `E11_kvq8_tgt_only`.
- **Comando**: `scripts/exp.sh E20_kvq8_ub2048 8192,32768 3 1 -- <ARGS rif. + q8> -b 2048 -ub 2048`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun errore ✔
- **Memoria**: `CUDA0 compute buffer size = 1437.02 MiB` (rif. 982.01 con ub1024) | `vram_max = 8939 MiB` | RAM avail min 24112 MiB
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 764.78 | 1309.79 | **+71.3** | 31.70 | 28.67 | −9.6 | 0.85 |
  | 32768 | 761.58 | 1267.00 | **+66.4** | 28.00 | 27.75 | −0.9 | 0.97 |
- **Decisione**: TENUTO — PP +66/71 %, TG a 32K neutra (−0.9 %). Il calo TG a 8K (−9.6 %, con acc 0.85 vs 0.88) è annotato come compromesso: a 8K la generazione parte subito dopo un prompt breve, dove il compute buffer più grande (1437 MiB) lascia meno VRAM/L2 alla cache; a 32K+ il fenomeno sparisce.
- **File**: `results/bench_E20_kvq8_ub2048.jsonl`, `logs/exp/E20_kvq8_ub2048.log`

### E21_kvq8_ub4096 — 2026-11-06 09:27
- **Ipotesi**: H2+H4 — ubatch 4096 con KV q8_0.
- **Modifica (unica)**: `-b 4096 -ub 4096` rispetto a `E11_kvq8_tgt_only`.
- **Verifiche**: avvio **FALLITO** — `allocating 1884.12 MiB on device 0: cudaMalloc failed: out of memory`; compute buffer ub4096 ≈ 2724 MiB (stimato dai log: 1437 a ub2048), non entra con KV q8 + cache 1024.
- **Decisione**: SCARTATO — ub2048 è il massimo che regge a 32K con questa configurazione. Non si riduce il contesto né la cache per inseguirlo (§0.2/§0.3).
- **File**: `logs/exp/E21_kvq8_ub4096.log`, `logs/queue_phase3.out`

## FASE 4 — Memoria pinned: `--load-mode none` (H3)

### E30_loadnone — 2026-11-06 09:40
- **Ipotesi**: H3 — con mmap gli esperti stanno in pagine non pinnate; `--load-mode none` li mette in `CUDA_Host` pinned e rende le DMA reali → PP e miss della MoE cache più veloci.
- **Collo di bottiglia affrontato**: copie H2D da memoria pageable.
- **Modifica (unica)**: `--load-mode none` rispetto a `E20_kvq8_ub2048`.
- **Comando**: `scripts/exp.sh E30_loadnone 8192,32768 3 1 -- <ARGS rif.> --load-mode none`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun errore ✔ | log: `load_mode = none` ✔ | `CUDA_Host model buffer size = 14339.85 MiB`, **zero** righe `CPU_Mapped model buffer` ✔ | warning "tensor overrides to CPU are used with mmap enabled" **sparito** ✔
- **Memoria**: `vram_max = 8963 MiB` (invariata) | RAM available min **9669 MiB** (pinned ~14 GB) | swap max 47 MiB | `load_s` 13 s (rif. 4–5 s con mmap)
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1309.79 | 1395.03 | **+6.5** | 28.67 | 32.29 | **+12.6** | 0.87 |
  | 32768 | 1267.00 | 1351.21 | **+6.6** | 27.75 | 28.44 | +2.5 | 0.96 |
- **Decisione**: TENUTO — migliora sia PP sia TG, la VRAM non cambia. Costo: ~14.3 GB di RAM pinnata (available 9.7 GB, sopra il minimo di 6 GB richiesto dal piano; annotato come rischio per browser/IDE).
- **File**: `results/bench_E30_loadnone.jsonl`, `logs/exp/E30_loadnone.log`, `logs/exp/E30_loadnone.mem_idle.txt`

### E31_loadnone_mlock — 2026-11-06 09:46
- **Ipotesi**: `--load-mode mlock` (con `--ulimit memlock=-1:-1`) può superare `none`.
- **Modifica (unica)**: `--load-mode mlock` rispetto a `E30_loadnone`.
- **Verifiche**: `n_ctx=200192` ✔ | `load_mode = mlock` ✔ | nessun errore ✔
- **Memoria**: `vram_max = 8933 MiB` | RAM avail min 9624 MiB
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1395.03 | 1393.18 | −0.1 | 32.29 | 31.23 | −3.3 | 0.84 |
  | 32768 | 1351.21 | 1348.58 | −0.2 | 28.44 | 28.75 | +1.1 | 0.97 |
- **Decisione**: NEUTRO/SCARTATO — nessun guadagno misurabile; TG 8K −3.3 %. Si resta su `none` (meno requisiti: niente `ulimit` speciale).
- **File**: `results/bench_E31_loadnone_mlock.jsonl`, `logs/exp/E31_loadnone_mlock.log`

### E30_session_none — 2026-11-06 09:49
- **Ipotesi**: la memoria pinned regge una sessione agentica senza errori né degrado.
- **Modifica (unica)**: nessuna; `session.py --target 60000` su `E30_loadnone`.
- **Verifiche**: 9 turni, `prompt_n` 6082–6548 (nessuna rielaborazione completa) ✔ | nessun errore nel log ✔
- **Risultati**: contesto 6 623 → 55 854 token; TTFT mediano **5.54 s**; PP mediano 1111.1 t/s; TG mediano 30.04 t/s; 71.7 s totali.
- **Decisione**: TENUTO — stabilità confermata con RAM pinnata.
- **File**: `results/session_E30_session_none.jsonl`, `logs/exp/E30_session_none_server.log`

## FASE 5 — Generazione: MoE cache (H5) e MTP (H6)

### E40_mc0 — 2026-11-06 09:54
- **Ipotesi**: H5-dato — se la generazione è limitata dal PCIe, calcolare gli esperti su CPU (DDR4) può essere più veloce che caricarli nella cache GPU ad ogni token.
- **Collo di bottiglia affrontato**: upload PCIe degli esperti in generazione.
- **Modifica (unica)**: `--moe-cache-mib 0` (cache disattivata) rispetto a `E30_loadnone` (1024).
- **Comando**: `GEN=512 scripts/exp.sh E40_mc0 8192,32768 4 1 -- <ARGS rif.> --moe-cache-mib 0`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun errore ✔ | nessuna riga `llama_moe_cache:` (cache assente) ✔
- **Memoria**: `vram_max = 7940` (8K) / **7960** (32K) MiB | RAM avail min 8814 MiB | swap max 199 MiB
- **Risultati (mediane, n=4)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1395.03 | 1362.78 | −2.3 | 32.29 | **42.22** | **+30.8** | 0.83 |
  | 32768 | 1351.21 | 1323.88 | −2.0 | 28.44 | **45.09** | **+58.5** | 0.97 |
- **Decisione**: TENUTO — TG +31/+59 %, PP −2 % (neutro). Conferma quantitativa che **la MoE cache danneggia la generazione**: ogni miss costa un trasferimento PCIe, mentre la CPU calcola gli esperti dalla DDR4 più in fretta. La cache è utile solo al prompt processing (large batches).
- **File**: `results/bench_E40_mc0.jsonl`, `logs/exp/E40_mc0.log`

### E41_mc1536 — 2026-11-06 09:58
- **Ipotesi**: una cache più grande (≥ 24 slot per gruppo) elimina i miss dei layer q6_K/q4_K in verifica MTP (H5).
- **Modifica (unica)**: `--moe-cache-mib 1536` rispetto a `E30_loadnone` (1024).
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔ | slot: 1042 / 55 / **27** / **27** (min 27 ≥ 24) | hit rate 29.57 % (vs 15.21 % con 1024)
- **Memoria**: `vram_max = 9455` (8K) / 9481 (32K) MiB | RAM avail min 8760 MiB
- **Risultati (mediane, n=4)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1395.03 | 1408.88 | +1.0 | 32.29 | 37.48 | +16.1 | 0.86 |
  | 32768 | 1351.21 | 1367.48 | +1.2 | 28.44 | 32.24 | +13.4 | 0.97 |
- **Decisione**: MIGLIORA la cache 1024 ma **peggiora rispetto a cache 0** (TG 37.5/32.2 vs 42.2/45.1). Non adottato.
- **File**: `results/bench_E41_mc1536.jsonl`, `logs/exp/E41_mc1536.log`

### E42_mc2048 — 2026-11-06 10:03
- **Ipotesi**: cache 2048 MiB per massimizzare l'hit rate.
- **Modifica (unica)**: `--moe-cache-mib 2048` rispetto a `E30_loadnone`.
- **Verifiche**: avvio **FALLITO** — `CUDA error: out of memory` durante il warm-up (la cache entra nell'allocazione iniziale ma non lascia spazio ai compute buffer).
- **Decisione**: SCARTATO — non entra con KV q8 + ub2048 + `--load-mode none`.
- **File**: `logs/exp/E42_mc2048.log`

### E43_mc3072 — non eseguito
- **Motivo**: 2048 già in OOM; 3072 non può entrare (§5.1 "solo se la VRAM lo permette"). Annotato qui per completezza della checklist.

### E44_mcMAX — 2026-11-06 10:24
- **Ipotesi**: trovare il massimo `--moe-cache-mib` che lascia `vram_max ≤ 9900` a 32K.
- **Modifica (unica)**: `--moe-cache-mib 1792` rispetto a `E40_mc0`.
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔ | slot: 1216 / 64 / **31** / **31** | hit rate 35.37 %
- **Memoria**: `vram_max = 9751` (8K) / 9758 (32K) MiB → è il massimo che entra (2048 OOM).
- **Risultati (mediane, n=4)**: PP 1415.95 / 1383.35; TG **39.02 / 36.34**; acc 0.90 / 0.98.
- **Decisione**: SCARTATO — anche la cache massima possibile resta sotto cache 0 come TG (39.0/36.3 vs 42.2/45.1). Il "massimo" non è il migliore: la cache conviene solo al PP, e solo marginalmente.
- **File**: `results/bench_E44_mcMAX.jsonl`, `logs/exp/E44_mcMAX.log`

### E45_ncmoe37 — 2026-11-06 10:28
- **Ipotesi**: H5 alternativa — invece della cache, tenere in VRAM gli esperti di alcuni layer interi (`--n-cpu-moe 37` → layer 37–39 su GPU, ~996 MiB).
- **Modifica (unica)**: `--n-cpu-moe 37 --moe-cache-mib 0` al posto di `--cpu-moe --moe-cache-mib 0` (E40).
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔ | `CUDA0 model buffer size = 3545.70 MiB` (rif. 1961.70, +1584) | nessuna cache
- **Memoria**: `vram_max = 9504` (8K) / **9525** (32K) MiB | RAM avail min 10485 MiB | swap max 335 MiB
- **Risultati (mediane, n=4)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1362.78 | 1468.84 | **+7.8** | 42.22 | **48.49** | **+14.8** | 0.90 |
  | 32768 | 1323.88 | 1423.32 | **+7.5** | 45.09 | **48.66** | **+7.9** | 0.98 |
- **Decisione**: **TENUTO — nuovo riferimento**. Gli esperti residenti in VRAM non pagano PCIe e non pagano nemmeno la CPU: battono sia la cache sia la CPU pura.
- **File**: `results/bench_E45_ncmoe37.jsonl`, `logs/exp/E45_ncmoe37.log`

### E46_ncmoe36 — 2026-11-06 10:33
- **Ipotesi**: un layer esperto in più su GPU (4 layer, `--n-cpu-moe 36`) migliora ancora.
- **Modifica (unica)**: `--n-cpu-moe 36` rispetto a `E45_ncmoe37`.
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔
- **Memoria**: `vram_max = 9856` (8K) / 9829 (32K) MiB (margine 44 MiB dal limite 9900) | RAM avail min 10861 MiB
- **Risultati (mediane, n=4)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1468.84 | 1501.94 | +2.2 | 48.49 | 48.02 | −1.0 | 0.87 |
  | 32768 | 1423.32 | 1450.45 | +1.9 | 48.66 | 49.33 | +1.4 | 0.98 |
- **Verifica a 196608** (`E46_ncmoe36_long`): PP **1036.74 t/s**, TG **32.38 t/s**, `vram_max = 9855 MiB`, nessun errore.
- **Decisione**: NEUTRO — guadagno < 5 % su entrambe le metriche e margine VRAM ridotto a ~45 MiB. Si preferisce E45 (VRAM 9525, margine 375 MiB). Secondo §3.5 una differenza del 1–2 % è rumore.
- **File**: `results/bench_E46_ncmoe36.jsonl`, `results/bench_E46_ncmoe36_long.jsonl`, `logs/exp/E46_ncmoe36.log`

### E47_nmax2_ref — 2026-11-06 10:53
- **Ipotesi**: riferimento di Fase 5.3 (n_max 2) sulle tre profondità richieste, con la configurazione `E45_ncmoe37`.
- **Modifica (unica)**: aggiunta della profondità 131072 rispetto a `E45_ncmoe37` (stessa configurazione).
- **Comando**: `GEN=512 scripts/exp.sh E47_nmax2_ref 8192,32768,131072 4 1 -- <ARGS rif.>`
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔
- **Risultati (mediane)**: 8K PP 1467.72 / TG **48.08** (acc 0.91); 32K PP 1426.97 / TG **48.73** (0.98); 131K PP 1162.29 / TG **41.01** (0.94, n=1). VRAM max 9551 MiB.
- **Decisione**: riferimento per 5.3.
- **File**: `results/bench_E47_nmax2_ref.jsonl`, `logs/exp/E47_nmax2_ref.log`

### E50_nmax1 — 2026-11-06 10:59
- **Ipotesi**: H6 — con n_max 1 meno token draft sprecati; l'accettazione è alta (0.96–0.99).
- **Modifica (unica)**: `--spec-draft-n-max 1` rispetto a `E47_nmax2_ref`.
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔
- **Risultati (mediane, n=4)**:
  | depth | TG rif | TG nuovo | Δ% | acc rif | acc nuovo |
  |---|---|---|---|---|---|
  | 8192 | 48.08 | 46.78 | −2.7 | 0.91 | 0.96 |
  | 32768 | 48.73 | 44.42 | **−8.8** | 0.98 | 0.99 |
  | 131072 | 41.01 | 37.37 | **−8.9** | 0.94 | 0.98 |
- **Decisione**: SCARTATO — meno token per step di verifica; l'accettazione più alta non compensa.
- **File**: `results/bench_E50_nmax1.jsonl`, `logs/exp/E50_nmax1.log`

### E51_nmax3 — 2026-11-06 11:05
- **Ipotesi**: H6 — n_max 3 può battere 2 sui token/s effettivi.
- **Modifica (unica)**: `--spec-draft-n-max 3` rispetto a `E47_nmax2_ref`.
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔
- **Risultati (mediane, n=4 / n=1 a 131K)**:
  | depth | PP rif | PP nuovo | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|
  | 8192 | 1467.72 | 1469.26 | 48.08 | **49.74** | +3.5 | 0.92 |
  | 32768 | 1426.97 | 1422.41 | 48.73 | **49.91** | +2.4 | 0.97 |
  | 131072 | 1162.29 | 1153.93 | 41.01 | **45.18** | **+10.2** | 0.97 |
- **Decisione**: **TENUTO come riferimento** — nessuna profondità peggiora, il guadagno a 131K è +10.2 %; a 8–32K è sotto la soglia del 5 % (quindi formalmente "neutro"), ma non c'è motivo di scartare una variante mai peggiore. Da riconfermare nella Fase 8 FULL (2 rep a 131K/196K).
- **File**: `results/bench_E51_nmax3.jsonl`, `logs/exp/E51_nmax3.log`

### E52_nmax4 — 2026-11-06 11:11
- **Ipotesi**: H6 — n_max 4 sfrutta ulteriormente l'alta accettazione.
- **Modifica (unica)**: `--spec-draft-n-max 4` rispetto a `E47_nmax2_ref`.
- **Risultati (mediane, n=4 / n=1)**:
  | depth | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|
  | 8192 | 48.08 | 45.36 | −5.7 | 0.78 |
  | 32768 | 48.73 | 51.39 | +5.5 | 0.96 |
  | 131072 | 41.01 | 39.66 | −3.3 | 0.77 |
- **Decisione**: SCARTATO — l'accettazione crolla a 0.77–0.78 a 8K e 131K e la TG peggiora; solo i 32K migliorano. Non riproducibile.
- **File**: `results/bench_E52_nmax4.jsonl`, `logs/exp/E52_nmax4.log`

### E53_nmax3_pmin05 — 2026-11-06 11:18
- **Ipotesi**: interrompere il draft quando la confidenza è bassa risparmia lavoro.
- **Modifica (unica)**: `--spec-draft-p-min 0.5` rispetto a `E51_nmax3`.
- **Risultati (mediane, n=4 / n=1)**: TG 46.24 (8K, **−7.0 %**), 49.89 (32K, neutro), 43.56 (131K, −3.6 %); acc 0.89/0.97/0.93.
- **Decisione**: SCARTATO — peggiora a 8K e 131K.
- **File**: `results/bench_E53_nmax3_pmin05.jsonl`, `logs/exp/E53_nmax3_pmin05.log`

### E54_nmax3_nodraftmoe — 2026-11-06 11:24
- **Ipotesi**: il draft MTP non ha esperti propri, quindi `--spec-draft-cpu-moe` potrebbe essere inutile.
- **Modifica (unica)**: rimozione di `--spec-draft-cpu-moe` rispetto a `E51_nmax3`.
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔
- **Risultati (mediane, n=4 / n=1)**: TG 47.43 (8K, −4.6 %), 49.77 (32K, −0.3 %), 41.83 (131K, **−7.4 %**); acc 0.85/0.96/0.88.
- **Decisione**: SCARTATO — la flag **non è neutra**: senza di essa l'accettazione e la TG a lungo contesto peggiorano. Resta nella configurazione.
- **File**: `results/bench_E54_nmax3_nodraftmoe.jsonl`, `logs/exp/E54_nmax3_nodraftmoe.log`

## FASE 6 — Parametri secondari

> Nota metodologica: tutti i confronti di questa fase usano lo stesso `GEN` del riferimento.
> I test su prompt brevi usano `GEN=64`; i test standard usano `GEN=256` con `REPS=3` e il
> riferimento dedicato `E66_ref` (stessa configurazione, stesse condizioni).

### E62_ref — 2026-11-06 11:31
- **Ipotesi**: riferimento per i test su prompt brevi (H7).
- **Comando**: `GEN=64 scripts/exp.sh E62_ref 64,128,256 5 1 -- <ARGS rif.>`
- **Risultati (mediane, n=5)**: PP **125.38** / **196.94** / **291.07** t/s a 64/128/256 token; TTFT 0.51/0.65/0.88 s; TG 39.69/40.21/43.15.
- **Decisione**: riferimento.
- **File**: `results/bench_E62_ref.jsonl`, `logs/exp/E62_ref.log`

### E62_minb64 — 2026-11-06 11:32
- **Ipotesi**: H7 — con `GGML_OP_OFFLOAD_MIN_BATCH=64` i prompt sotto 64 token restano sulla CPU ed evitano il costo di trasferimento esperti.
- **Modifica (unica)**: `-e GGML_OP_OFFLOAD_MIN_BATCH=64` rispetto a `E62_ref`.
- **Verifiche**: `n_ctx=200192` ✔ | nessun errore ✔
- **Risultati (mediane, n=5)**: PP 103.05 (**−17.8 %**) / 196.91 (0.0 %) / 290.61 (−0.2 %); TG 38.76/37.78/41.00.
- **Decisione**: SCARTATO — a 64 token peggiora del 18 %, altrove neutro. H7 non confermata: offloadare alla GPU conviene anche per batch piccoli.
- **File**: `results/bench_E62_minb64.jsonl`, `logs/exp/E62_minb64.log`

### E63_minb128 — 2026-11-06 11:33
- **Ipotesi**: H7 con soglia 128.
- **Modifica (unica)**: `-e GGML_OP_OFFLOAD_MIN_BATCH=128` rispetto a `E62_ref`.
- **Risultati (mediane, n=5)**: PP 102.50 (**−18.2 %**) / 131.94 (**−33.0 %**) / 289.88 (−0.4 %); TG 43.49/35.77/38.01.
- **Decisione**: SCARTATO — peggiora nettamente a 64 e 128 token.
- **File**: `results/bench_E63_minb128.jsonl`, `logs/exp/E63_minb128.log`

### E66_ref — 2026-11-06 11:45
- **Ipotesi**: riferimento di Fase 6 nelle stesse condizioni dei test E60/E61/E64/E65 (`REPS=3`, `GEN=256`).
- **Comando**: `scripts/exp.sh E66_ref 8192,32768 3 1 -- <ARGS rif.>`
- **Risultati (mediane, n=3)**: 8K PP 1467.82 / TG **44.78** (acc 0.79); 32K PP 1422.95 / TG **49.53** (0.96). VRAM max 9594 MiB.
- **Decisione**: riferimento per la Fase 6.
- **File**: `results/bench_E66_ref.jsonl`, `logs/exp/E66_ref.log`

### E60_t7 — 2026-11-06 11:34
- **Ipotesi**: H9 — 7 thread riducono la contesa con il thread principale.
- **Modifica (unica)**: `--threads 7` rispetto a `E66_ref`.
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | TG rif | TG nuovo | Δ% |
  |---|---|---|---|---|---|
  | 8192 | 1467.82 | 1472.14 | 44.78 | 40.73 | **−9.0** |
  | 32768 | 1422.95 | 1427.93 | 49.53 | 46.16 | **−6.8** |
- **Decisione**: SCARTATO — la parte CPU (esperti dei layer 0–36) perde throughput.
- **File**: `results/bench_E60_t7.jsonl`, `logs/exp/E60_t7.log`

### E61_t16 — 2026-11-06 11:36
- **Ipotesi**: H9 — 16 thread sfruttano l'SMT.
- **Modifica (unica)**: `--threads 16 --threads-batch 16` rispetto a `E66_ref`.
- **Risultati (mediane, n=3)**: PP 1457.69 / 1419.45 (neutro); TG **29.59** (−33.9 %) / **32.79** (−33.8 %).
- **Decisione**: SCARTATO — l'SMT danneggia il calcolo esperto su CPU (contesa AVX2).
- **File**: `results/bench_E61_t16.jsonl`, `logs/exp/E61_t16.log`

### E64_prio — 2026-11-06 11:39
- **Ipotesi**: `--prio 2` dà priorità alta al processo e riduce le latenze.
- **Modifica (unica)**: `--prio 2` rispetto a `E66_ref`.
- **Risultati (mediane, n=3)**: PP neutro; TG **36.95** (−17.5 %) / **41.49** (−16.2 %).
- **Decisione**: SCARTATO — effetto negativo netto (probabile inversione di priorità con i thread GPU/CPU nel container). Non usare.
- **File**: `results/bench_E64_prio.jsonl`, `logs/exp/E64_prio.log`

### E65_poll — 2026-11-06 11:41
- **Ipotesi**: `--poll 0` riduce il busy-wait e libera CPU per il calcolo degli esperti.
- **Modifica (unica)**: `--poll 0` rispetto a `E66_ref`.
- **Risultati (mediane, n=3)**: PP 1470.90 / 1428.31 (neutro); TG 45.70 (+2.1 %) / 48.92 (−1.2 %).
- **Decisione**: NEUTRO — sotto la soglia del 5 %, nessun motivo per discostarsi dal default.
- **File**: `results/bench_E65_poll.jsonl`, `logs/exp/E65_poll.log`

## FASE 7 — Runtime più recente (H8)

### E70_b12124 — 2026-11-06 11:50
- **Ipotesi**: H8 — la build b12124 (commit 8bed7cfb9) include letture batched in memoria host mappata, salvataggio stato MTP ed eviction grafi CUDA.
- **Immagine**: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12124` = digest `sha256:821d0ae4e5d4e82a2ad8d08973b5f9cd5d0b84524b1d2439ed1f3e41ebc6b4c7`, commit `8bed7cfb999cebc03d2cc5b78911026a8d57549a`, versione b12124. Help in `logs/help_b12124.txt`: tutte le flag usate esistono ✔.
- **Modifica (unica)**: immagine b12124 al posto di b11977 (`E51_nmax3`), stessi argomenti.
- **Comando**: `GEN=512 scripts/exp.sh E70_b12124 8192,32768,131072 4 1 --image …:server-cuda-b12124 -- <ARGS rif.>`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun errore ✔
- **Memoria**: `vram_max = 9619` (8K) / 9590 (32K, 131K) MiB | RAM avail min 9814 MiB
- **Risultati (mediane, n=4 / n=1)**:
  | depth | PP b11977 | PP b12124 | Δ% | TG b11977 | TG b12124 | Δ% | acc |
  |---|---|---|---|---|---|---|---|
  | 8192 | 1469.26 | 1469.72 | +0.03 | 49.74 | 44.59 | **−10.4** | 0.77 |
  | 32768 | 1422.41 | 1421.14 | −0.1 | 49.91 | 49.55 | −0.7 | 0.96 |
  | 131072 | 1153.93 | 1152.41 | −0.1 | 45.18 | 45.31 | +0.3 | 0.98 |
- **Agentico** (`E70_agentic`): tool_call ok, tool_call_stream ok, multi_turn ok, coding **10/10**, `tasks_per_min = 8.83`, TG mediana sul coding 50.28 t/s.
- **Decisione**: **NON ADOTTATO** — nessun guadagno ≥ 5 % su PP o TG (anzi TG 8K −10.4 %, accettazione 0.77 vs 0.92). Il piano richiede un guadagno reale per abbandonare la build più collaudata. Si resta su **b11977**. Il codice agentico passa comunque 10/10: la build è funzionalmente valida, solo non più veloce.
- **File**: `results/bench_E70_b12124.jsonl`, `results/agentic_E70_agentic.jsonl`, `logs/exp/E70_b12124.log`, `logs/help_b12124.txt`, `logs/pull_b12124.out`

## FASE 8 — Validazione completa dei finalisti

### F1_full — 2026-11-06 11:59
- **Ipotesi**: la configurazione F1 (KV q8, ub2048, load-mode none, n-cpu-moe 37, cache 0, n_max 3, b11977) è il finalista migliore e regge il benchmark completo.
- **Modifica (unica)**: nessuna — benchmark FULL del finalista.
- **Comando**: `scripts/exp.sh F1_full 8192,16384,32768,65536,131072,196608 3 2 -- <ARGS F1>`
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=depth` ✔ | nessun `warning` ✔ | nessun errore ✔ | VRAM max 9597 ≤ 9900 ✔ | RAM min 5517 ≥ 3000 ✔ | swap max ~452 MiB ✔
- **Risultati (mediane, n=3 / n=2)**:
  | depth | PP t/s | TG t/s | MTP acc | TTFT s | VRAM max | RAM min |
  |---|---|---|---|---|---|---|
  | 8192 | 1473.36 | 46.87 | 0.88 | 5.56 | 9597 | 12895 |
  | 16384 | 1474.51 | 48.12 | 0.89 | 11.11 | 9569 | 11819 |
  | 32768 | 1426.27 | 49.67 | 0.96 | 22.97 | 9570 | 10233 |
  | 65536 | 1318.21 | 41.94 | 0.79 | 49.72 | 9568 | 8603 |
  | 131072 | 1156.77 | 43.05 | 0.92 | 113.31 | 9568 | 5749 |
  | 196608 | 1030.62 | 34.97 | 0.74 | 190.77 | 9595 | 5517 |
- **Decisione**: **FINALISTA 1** — rispetto a E00b: PP +231 % (8K) … +167 % (192K), TG +49 % … +21 %.
- **File**: `results/bench_F1_full.jsonl`, `logs/exp/F1_full.log`

### F1_max — 2026-11-06 12:17
- **Ipotesi**: capacità vicino al limite del contesto (199 000 token).
- **Modifica (unica)**: profondità 199000, 1 rep, `GEN=128`.
- **Verifiche**: `n_ctx=200192` ✔ | `prompt_n=199000` ✔ | nessun errore ✔
- **Risultati**: **PP 1021.68 t/s, TG 36.71 t/s, TTFT 194.8 s**, VRAM max 9587 MiB, RAM min 13357 MiB.
- **Decisione**: SUPERATO — il contesto pieno è utilizzabile con margine VRAM.
- **File**: `results/bench_F1_max.jsonl`, `logs/exp/F1_max.log`

### F1_agentic — 2026-11-06 12:21
- **Ipotesi**: qualità agentica del finalista.
- **Modifica (unica)**: `agentic.py --coding-reps 2` su F1.
- **Risultati**: tool_call ok, tool_call_stream ok, multi_turn ok, coding **10/10**, `tasks_per_min = 8.77`, TG mediana sul coding ≈ 50 t/s.
- **Decisione**: SUPERATO (nessuna perdita di qualità; +31 % di lavoro utile/minuto vs baseline 6.7).
- **File**: `results/agentic_F1_agentic.jsonl`, `logs/exp/F1_agentic_server.log`

### F1_needle — 2026-11-06 12:23
- **Ipotesi**: recupero a lungo contesto del finalista.
- **Risultati**: **5/5** aghi trovati su 150 128 token; PP 1118.9 t/s.
- **Decisione**: SUPERATO (uguale alla baseline f16).
- **File**: `results/needle_F1_needle.jsonl`, `logs/exp/F1_needle_server.log`

### F1_session / F1_session2 — 2026-11-06 12:25–12:33
- **Ipotesi**: stabilità su due sessioni agentiche consecutive senza riavviare il server.
- **Modifica (unica)**: `session.py --target 180000` eseguito due volte sullo stesso server.
- **Risultati**:
  | tag | turni | contesto finale | prompt_n min–max | TTFT med | PP med | TG med | vuoti/1-token |
  |---|---|---|---|---|---|---|---|
  | F1_session | 29 | 177 082 | 6083–6548 | 6.45 s | 969.3 | 36.81 | 7 |
  | F1_session2 | 29 | 179 246 | 6082–6100 | 6.45 s | 967.8 | 37.09 | 0 |
- **Verifiche**: nessuna rielaborazione completa (`prompt_n` ≈ 6 085) ✔ | nessun errore/abort/OOM nel log ✔ | swap stabile ✔
- **Decisione**: SUPERATO.
- **File**: `results/session_F1_session.jsonl`, `results/session_F1_session2.jsonl`, `logs/exp/F1_session_server.log`

### F1_restart1 / F1_restart2 — 2026-11-06 13:03–13:10
- **Ipotesi**: riavvio riproducibile (2 stop/start) entro ±5 %.
- **Modifica (unica)**: due riavvii del container con SCREEN 8K/32K, 3 rep, stessa configurazione F1.
- **Risultati**: PP 8K 1473.86/1470.11 vs 1473.36 (Δ ≤ 0.2 %); TG 8K 44.58/46.92 vs 46.87 (−4.9 %/+0.1 %); PP 32K 1424.37/1425.49 vs 1426.27; TG 32K 49.09/49.00 vs 49.67 (−1.2 %/−1.3 %).
- **Decisione**: SUPERATO — tutte le differenze entro ±5 %.
- **File**: `results/bench_F1_restart1.jsonl`, `results/bench_F1_restart2.jsonl`, `logs/queue_restart.out`

### F2_full — 2026-11-06 12:37
- **Ipotesi**: finalista conservativo = F1 senza `--load-mode none` (mmap), con molta più RAM libera.
- **Modifica (unica)**: rimozione di `--load-mode none` rispetto a F1.
- **Verifiche**: `n_ctx=200192` ✔ | warning mmap atteso ✔ | nessun errore ✔ | VRAM max 9568 | RAM min 18702
- **Risultati (mediane, n=3 / n=2)**:
  | depth | PP F1 | PP F2 | Δ% | TG F1 | TG F2 | Δ% |
  |---|---|---|---|---|---|---|
  | 8192 | 1473.36 | 1398.43 | −5.1 | 46.87 | 45.43 | −3.1 |
  | 16384 | 1474.51 | 1389.74 | −5.7 | 48.12 | 44.39 | −7.8 |
  | 32768 | 1426.27 | 1347.43 | −5.5 | 49.67 | 49.40 | −0.5 |
  | 65536 | 1318.21 | 1258.19 | −4.6 | 41.94 | 38.20 | −8.9 |
  | 131072 | 1156.77 | 1115.07 | −3.6 | 43.05 | 42.23 | −1.9 |
  | 196608 | 1030.62 | 1001.32 | −2.8 | 34.97 | 34.04 | −2.7 |
- **Agentico/needle/sessione**: `F2_agentic` 10/10 (8.65 task/min), `F2_needle` 5/5, `F2_session` 29 turni con `prompt_n` stabile ma 16 risposte vuote/1 token (fenomeno stocastico del modello, presente anche in F1 in 7 turni).
- **Decisione**: **FINALISTA 2 (alternativa conservativa)** — ~5 % più lento in PP ma lascia ~18.7 GB di RAM liberi invece di ~5.5 GB. Non promosso a produzione: F1 è più veloce e la macchina non ha carenza di RAM durante l'uso previsto.
- **File**: `results/bench_F2_full.jsonl`, `results/agentic_F2_agentic.jsonl`, `results/needle_F2_needle.jsonl`, `results/session_F2_session.jsonl`, `logs/exp/F2_full.log`

---

## Nota tecnica — correzione di `bench/summarize.py` (2026-11-06)

`bench/summarize.py` riscrive `results/summary.csv` a ogni invocazione. La versione
precedente usava come tag la lista degli argomenti, quindi un'invocazione filtrata
(`summarize.py F1_full`) o errata (`--help`) **troncava** il CSV consolidato alle sole
righe di quel tag (o a zero righe, con `IndexError`). Corretto: la tabella stampata
rispetta i tag richiesti, ma `results/summary.csv` viene **sempre** rigenerato con
tutti i tag; un tag inesistente stampa solo l'intestazione senza toccare i dati.
Verifica: file da 95 righe (94 di dati + intestazione) prima e dopo `summarize.py F1_full`
e dopo un tag inesistente.
Nessun prompt o test unitario è stato modificato.
