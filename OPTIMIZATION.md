# OPTIMIZATION.md — configurazione finale, perché, limiti e gestione

Documento operativo della produzione `qwen36-35b-opt`.
Tutte le cifre provengono da `results/*.jsonl`, `logs/exp/*` e `results/summary.csv`
(mediane calcolate da `python3 bench/summarize.py`). Nessun valore è stimato a mano
salvo dove è indicato esplicitamente un calcolo derivato.

---

## 1. La configurazione finale (F1)

Container di produzione: **`qwen36-35b-opt`**, immagine
`ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276`
(b11977, commit `7ac5727d6`), porta **`127.0.0.1:8080`**.

```
-m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto
--n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe
-ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --load-mode none
```

Avvio/arresto: `scripts/prod_start.sh` / `scripts/prod_stop.sh` (vedi §7).

### 1.1 Riga per riga

| Flag | Valore | Perché | Esperimento |
|---|---|---|---|
| `-m` | `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | unico modello ammesso dal piano; mai copiato/convertito | — |
| `--host` / `--port` | `0.0.0.0` / `8080` | dentro il container; la pubblicazione è solo `-p 127.0.0.1:8080:8080` | — |
| `--ctx-size` | `200000` | requisito: contesto ≥ 200 000. Il runtime arrotonda a **`n_ctx = 200192`** (verificato in ogni log) | — |
| `--n-predict` | `-1` | nessun tetto alla generazione (agenti che scrivono codice lungo) | — |
| `--n-gpu-layers` | `auto` | tutti i layer sulla GPU; gli esperti sono poi spostati da `--n-cpu-moe`. Il warning `fit params ... tensor_buft_overrides already set` è innocuo: il contesto resta 200192 | BASELINE §7 |
| `--n-cpu-moe` | `37` | esperti dei layer **0–36 in RAM**, esperti dei layer **37–39 in VRAM** (+~1.5 GiB). Gli esperti residenti non pagano né PCIe né CPU | **E45** (+8 % PP, +7 % TG vs `--cpu-moe`) |
| `--moe-cache-mib` | `0` | **la cache MoE danneggia la generazione**: ogni miss costa un trasferimento PCIe, mentre la CPU calcola dalla DDR4 più in fretta. Utile solo al PP, e marginalmente | **E40** (TG +31/+59 % vs 1024), E41, E44 |
| `--flash-attn` | `on` | obbligatorio per la KV quantizzata su CUDA (in log `FA_QUANTS ... q8_0-q8_0`); era già attivo nella baseline | BASELINE |
| `--parallel` | `1` | un solo slot, che può usare l'intero contesto da 200K. Un LLM server alla volta (10 GB VRAM) | — |
| `--jinja` | — | chat template del modello e **tool calling** (usato da OpenCode/Cline) | — |
| `--spec-type` | `draft-mtp` | speculative decoding con la testa MTP (`nextn`) inclusa nel GGUF; già attivo nella baseline | BASELINE |
| `--spec-draft-n-max` | `3` | 3 token draft per step: migliore su tutte le profondità testate, +10 % TG a 128K | **E51** (E50 n=1, E52 n=4 scartati) |
| `--spec-draft-cpu-moe` | — | gli esperti del draft restano su CPU. **Non è neutra**: rimuoverla peggiora accettazione e TG a contesto lungo | **E54** |
| `-ctk` / `-ctv` | `q8_0` / `q8_0` | **solo KV del target** quantizzata. VRAM KV 3910 → 2077 MiB (−1833 MiB), qualità invariata (needle 5/5, coding 10/10) | **E11** (E10 target+draft scartato) |
| `-b` | `2048` | batch logico, invariato dalla baseline; deve essere ≥ ubatch | BASELINE |
| `-ub` | `2048` | ubatch 4× la baseline: **4× meno spostamenti di esperti per prompt**. PP +66/71 % sul riferimento precedente | **E20** (E02/E03/E21 a 4096 OOM) |
| `--load-mode` | `none` | esperti in RAM **pinned `CUDA_Host`** invece che mmap pageable → DMA reale. PP +6.5 %, TG +12.6 % a 8K. Costo: ~14.3 GB di RAM non swappabile | **E30** (E31 `mlock` neutro) |

---

## 2. Impronta di memoria della configurazione finale

Da `logs/exp/F1_full.log` (avvio, `n_ctx = 200192`):

| Buffer | MiB | Note |
|---|---|---|
| `CUDA0 model buffer` | 3545.70 | non-esperti + esperti layer 37–39 (baseline: 1961.70) |
| `CUDA0 KV buffer` (target, q8_0) | 2077.19 | baseline f16: 3910.00 |
| `CUDA0 KV buffer` (draft MTP) | 391.00 | f16 |
| `CUDA0 RS buffer` (Gated DeltaNet) | 251.25 | stato ricorrente |
| `CUDA0 compute buffer` (contesto principale) | 1437.02 | ubatch 2048 (baseline: 565.50 a ubatch 512) |
| `CUDA0 compute buffer` (contesto MTP) | 942.06 | |
| `CUDA0 MoE cache` | **0** | disattivata |
| **Somma buffer GPU** | **≈ 8644** | + ~0.9 GB desktop → `vram_max` osservato **9597 MiB** |
| `CUDA_Host model buffer` | 12755.85 | esperti layer 0–36, pinned |
| RAM disponibile minima | **5517 MiB** a 196K | pinnata ~14.3 GB; resta sopra il minimo di 3 GB del piano |

Margine VRAM: 10240 − 9597 ≈ **643 MiB** (desktop incluso). `E46` (`--n-cpu-moe 36`)
scendeva a ~45 MiB di margine per un guadagno dell'1–2 %: rifiutato.

---

## 3. Guadagni rispetto alla baseline

Baseline `E00b_baseline` (configurazione originale, KV f16, ubatch 512, MoE cache 1024,
mmap, MTP n_max 2) → produzione `F1` (`results/bench_F1_full.jsonl`).

| Metrica | E00b | F1 | Δ |
|---|---|---|---|
| PP 8K | 445.25 t/s | **1473.36 t/s** | **+231 %** |
| PP 16K | 451.23 t/s | 1474.51 t/s | +227 % |
| PP 32K | 443.20 t/s | 1426.27 t/s | **+222 %** |
| PP 64K | 424.63 t/s | 1318.21 t/s | +210 % |
| PP 128K | 398.85 t/s | 1156.77 t/s | **+190 %** |
| PP 192K | 385.84 t/s | 1030.62 t/s | **+167 %** |
| TG 8K | 31.41 t/s | **46.87 t/s** | **+49 %** |
| TG 32K | 28.54 t/s | 49.67 t/s | **+74 %** |
| TG 128K | 28.15 t/s | 43.05 t/s | **+53 %** |
| TG 192K | 28.90 t/s | 34.97 t/s | **+21 %** |
| TTFT 8K | 18.4 s | 5.56 s | −70 % |
| TTFT 32K | 73.94 s | **22.97 s** | **−69 %** |
| TTFT 128K | 328.63 s | **113.31 s** | **−66 %** |
| TTFT 192K | 509.6 s | 190.77 s | −63 % |
| Lavoro agentico | 6.7 task/min | **8.77 task/min** | **+31 %** |
| Coding | 10/10 | **10/10** | invariato |
| Needle 5 aghi @150K | 5/5 (f16) | **5/5** (q8_0) | invariato |
| Sessione lunga | 29 turni, ok | 29+29 turni, ok | invariato |
| VRAM max | 9470 MiB | 9597 MiB | +127 MiB |
| RAM disponibile min | 18631 MiB | 5517 MiB | −13114 MiB (pinned) |

### 3.1 Da dove viene il guadagno (una variabile per volta)

| Passo | Configurazione | PP 8K | TG 8K | TG 32K | Esperimento |
|---|---|---|---|---|---|
| 0 | baseline | 445.25 | 31.41 | 28.54 | E00b |
| 1 | + ubatch 1024 | 753.93 | 30.32 | 28.56 | E01 (+69 % PP) |
| 2 | + KV q8_0 (target) | 764.78 | 31.70 | 28.00 | E11 (VRAM −1905) |
| 3 | + ubatch 2048 | 1309.79 | 28.67 | 27.75 | E20 (+66/71 % PP) |
| 4 | + `--load-mode none` | 1395.03 | 32.29 | 28.44 | E30 (+6.5/12.6 %) |
| 5 | + MoE cache 0 | 1362.78 | 42.22 | 45.09 | E40 (TG +31/59 %) |
| 6 | + `--n-cpu-moe 37` | 1468.84 | 48.49 | 48.66 | E45 (+8/7 %) |
| 7 | + MTP n_max 3 | **1473.36** | **46.87** | **49.67** | E51 / F1_full |

Il grosso del PP viene da **ubatch** (passi 1 e 3): 16 → 4 passaggi completi degli esperti
per un prompt da 8K. Il grosso della TG viene da **cache MoE spenta + esperti in VRAM**
(passi 5 e 6). La KV q8_0 non è una velocizzazione: è ciò che libera la VRAM necessaria
per l'ubatch 2048 e per i 3 layer di esperti residenti, senza perdita di qualità misurata.

---

## 4. Limiti dimostrati

### 4.1 PCIe Gen3 x16 è il limite della baseline

La scheda madre B450 espone la RTX 3080 su **PCIe Gen3 x16** (la GPU è Gen4-capable,
l'host no; `pcie_gen_max = 3` in ogni riga di benchmark). Ogni ubatch di prompt deve
leggere dalla RAM gli esperti MoE che non stanno in VRAM.

Calcolo dal dato baseline (`results/bench_E00b_baseline.jsonl`, 8K, mediana):
- 8192 token / ubatch 512 = **16 ubatch**;
- esperti totali = **13 942 MiB = 13.62 GiB**; con `--cpu-moe` nessun esperto è residente,
  quindi un ubatch grande tocca praticamente tutti gli esperti → ~13.62 GiB per ubatch;
- `prompt_ms` mediano = **18 398.6 ms = 18.4 s**.

```
16 × 13.62 GiB / 18.4 s ≈ 11.8 GiB/s ≈ 12.7 GB/s
```

È il limite pratico di Gen3 x16 (~12.6–12.7 GB/s effettivi dopo l'encoding).
**La baseline era quindi satura sul bus, non sulla GPU.** Questo spiega perché:
- raddoppiare l'ubatch dà guadagni quasi lineari (metà passaggi → metà byte);
- la MoE cache non aiuta la generazione (aggiunge lavoro, non riduce i byte utili);
- nessun tuning di thread/priorità poteva compensare (Fase 6: tutto neutro o peggio).

Con F1 gli esperti residenti (3/40 layer) e l'ubatch 2048 riducono i byte per prompt:
al 8K servono 4 sweep di ~12.6 GiB in 5.56 s ≈ 9.1 GiB/s. **F1 non è più al tetto del bus**;
il margine residuo è CPU/compute, non PCIe.

### 4.2 VRAM (10 GB) e contesto

Il contesto non è riducibile sotto 200K per requisito. Con KV q8_0 il target occupa
2077 MiB; restano ~6.5 GiB per pesi non-esperti, esperti residenti, compute buffer e MTP.
`-ub 4096` è **OOM** anche con KV q8_0 (`E21`): ubatch 2048 è il massimo stabile.
La configurazione lavora a ~643 MiB dal limite: non c'è spazio per altri 3 layer di esperti
(`--n-cpu-moe 36` → ~45 MiB di margine).

### 4.3 Un solo server

Un secondo `llama-server` sullo stesso modello non entra in 10 GB. La produzione è
`parallel 1`, un container alla volta; il rollback richiede di fermare l'altro (§7).

---

## 5. Provato e scartato

| Esperimento | Cosa | Perché scartato |
|---|---|---|
| `E02_ub2048` (f16) | ubatch 2048 con KV f16 | **OOM** all'avvio (il contesto MTP chiede 942 MiB) |
| `E03` / `E21` (ub4096) | ubatch 4096 | **OOM** anche con KV q8_0 (`allocating 1884.12 MiB` fallita) |
| `E10_kvq8` | KV q8_0 su target **e** draft | TG −4.3 % a 32K, nessun beneficio di VRAM utile |
| `E31_loadnone_mlock` | aggiunta `--mlock` | neutro; TG 8K −3.3 %, richiede ulimit speciale |
| `E41_mc1536` | MoE cache 1536 | meglio di 1024 ma **peggio di 0** (TG 37.5/32.2 vs 42.2/45.1) |
| `E44_mcMAX` | MoE cache massima (1792) | TG 39.0/36.3, ancora sotto cache 0 |
| `E46_ncmoe36` | 4 layer di esperti in VRAM | +1–2 % (rumore), margine VRAM ~45 MiB |
| `E50_nmax1` | MTP 1 token | TG 46.8/44.4/37.4 < n_max 3 |
| `E52_nmax4` | MTP 4 token | accettazione crolla a 0.77–0.78, TG peggiore a 8K/128K |
| `E53_pmin05` | `--spec-draft-p-min 0.5` | TG 46.2/49.9/43.6, peggiore |
| `E54_nodraftmoe` | rimozione `--spec-draft-cpu-moe` | accettazione e TG a lungo contesto peggiorano |
| `E60_t7` / `E61_t16` | thread 7 / 16 | la parte CPU perde throughput; SMT danneggia gli esperti AVX2 |
| `E62` / `E63` | `GGML_OP_OFFLOAD_MIN_BATCH` 64 / 128 | −18 % e peggio: offloadare conviene anche per batch piccoli |
| `E64_prio` | `--prio 2` | effetto negativo netto |
| `E65_poll` | `--poll 0` | neutro (<5 %), si resta sul default |
| `E70_b12124` | runtime b12124 | nessun guadagno ≥5 %; TG 8K −10.4 %, acc 0.77 vs 0.92 |
| `F2` (mmap) | F1 senza `--load-mode none` | ~5 % più lento in PP; **non scartato**: documentato come alternativa conservativa (RAM min 18.7 GB invece di 5.5 GB) |

---

## 6. Proposte che richiedono autorizzazione (NON eseguite)

Queste modifiche esulano dai vincoli del laboratorio (niente sudo, niente modifiche a
governor/swap/driver/BIOS). Sono solo proposte, con l'effetto atteso dedotto dai dati.

1. **Governor CPU `performance` + EPP `performance`** (ora `powersave`/`balance_performance`).
   La parte CPU calcola gli esperti dei layer 0–36: un clock più alto e più stabile può
   ridurre la coda CPU durante la generazione. Effetto atteso: piccolo su PP (legato a PCIe),
   possibile su TG; da misurare.
2. **NVIDIA persistence mode** (`nvidia-smi -pm 1`). Evita il costo di inizializzazione
   del driver dopo un arresto; non cambia il throughput a regime.
3. **Resizable BAR nel BIOS** (oggi BAR1 = 256 MiB). Con ReBAR la GPU può mappare più
   memoria host: non aumenta la banda Gen3, ma può ridurre le latenze di accesso agli
   esperti. Da verificare; richiede modifica BIOS.
4. **Piattaforma PCIe Gen4** (scheda madre + CPU AM4/AM5 compatibili). È l'unico modo per
   alzare il tetto del bus oltre ~12.7 GB/s. Un raddoppio teorico del bus non raddoppia
   il throughput (a F1 il bus non è più saturo), ma riduce il costo residuo degli esperti
   non residenti. Costo hardware significativo: valutare solo se il carico cresce.
5. **CPU con L3 più grande** (es. Ryzen 5000/5800X3D su AM4). Gli esperti attivi sono
   ~10.4 MiB/layer: una L3 più grande può trattenerne di più e ridurre la pressione DRAM
   sulla parte CPU. Richiede cambio CPU e verifica BIOS.
6. **Ventole/power limit**: non toccati. Il power limit è già al massimo (370 W) e la GPU
   non è il collo di bottiglia (utilizzazione media 77–94 % durante i test).

Nessuna di queste è stata applicata: la produzione gira con la configurazione §1.

---

## 7. Avvio, arresto, rollback

Prerequisiti: un solo server alla volta. `bin/dk` va usato al posto di `docker`.

```bash
# Avvio produzione ottimizzata (attende /health, fino a 600 s)
scripts/prod_start.sh

# Verifica
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models

# Arresto (il container qwen36-35b-opt resta, pronto al riavvio)
scripts/prod_stop.sh
```

**Rollback all'originale** (container `qwen36-35b`, mai cancellato):

```bash
scripts/prod_stop.sh                 # ferma l'ottimizzato
bin/dk start qwen36-35b              # avvia l'originale sulla stessa 127.0.0.1:8080
curl -s http://127.0.0.1:8080/v1/models   # deve rispondere 200
```

Ritorno all'ottimizzato:

```bash
bin/dk stop qwen36-35b
scripts/prod_start.sh
```

Il rollback è stato provato in Fase 9: originale avviato, HTTP 200, poi fermato e
`qwen36-35b-opt` riavviato. Non eseguire mai i due container insieme (VRAM insufficiente).

---

## 8. Impostazioni consigliate nei client

Endpoint OpenAI-compatibile: **`http://127.0.0.1:8080/v1`**
(modello riportato da `/v1/models`: `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`).

| Client | Impostazione | Valore |
|---|---|---|
| OpenCode | `baseURL` | `http://127.0.0.1:8080/v1` |
| OpenCode | `limit.context` | **200000** |
| OpenCode | `limit.output` | ≥ 8192 (generazione non limitata lato server: `--n-predict -1`) |
| Cline / Roo | context window | **200000** |
| Cline / Roo | max output | ≥ 8192 |
| Tutti | tool calling | richiede `--jinja` (già attivo); il template del modello è Qwen3.6 |
| Tutti | thinking | per i task di codice inviare `chat_template_kwargs: {"enable_thinking": false}` (come in `bench/agentic.py`) |

Nota: il server ha un solo slot (`--parallel 1`). Due client che generano insieme si
serializzano; è il compromesso per avere l'intero contesto da 200K su una sola GPU.
