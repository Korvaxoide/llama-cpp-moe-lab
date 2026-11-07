# PLAN_FOR_AGENT.md — Piano operativo dettagliato per l'agente esecutore

> Destinatario: un agente AI che esegue i test e scrive la documentazione.
> Seguilo **in ordine**. Non saltare passi. Non inventare risultati: ogni numero scritto nei documenti
> deve provenire da un file in `results/` o `logs/`. Se un passo fallisce, applica la sezione
> "Se fallisce" del passo; se non c'è, fermati e scrivi il problema in `PROGRESS.md`.

Directory di lavoro: `$HOME/AI/llama-optimizer-lab` (d'ora in poi `$LAB`). Tutti i comandi partono da lì.

---

## 0. REGOLE ASSOLUTE (leggile prima di ogni fase)

1. **Modello**: usa solo `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`. Non toccare, copiare, convertire o riquantizzare il file. La directory è montata `:ro`.
2. **Contesto**: ogni configurazione deve avere `--ctx-size 200000` (o superiore). Nel log deve comparire `n_ctx = 200192` (o superiore). Se `n_ctx` è inferiore → configurazione SCARTATA.
3. **Mai `--fit on` con riduzione del contesto**: se nel log compare "context size reduced" la configurazione è invalida.
4. **Rete**: pubblica le porte solo su `127.0.0.1` (lo script `scripts/run_server.sh` lo fa già). Mai `-p 8080:8080` senza IP.
5. **Docker**: usa SEMPRE `bin/dk` al posto di `docker` (il processo di OpenCode non ha il gruppo docker; `bin/dk` lo aggiunge con `newgrp`).
6. **Un solo server alla volta**: la VRAM (10 GB) non basta per due istanze. Prima di avviarne uno, verifica con `bin/dk ps` che non ce ne siano altri in esecuzione con il modello caricato.
7. **Non cancellare** il container `qwen36-35b` né l'immagine `ghcr.io/ggml-org/llama.cpp:server-cuda`. Non fare `docker system prune`, `docker image prune`, `docker rm qwen36-35b`.
8. **Non modificare il sistema operativo**: niente sudo, governor CPU, swap/zram, driver, sysctl. Se un'idea lo richiede, scrivila in `OPTIMIZATION.md` sezione "Proposte che richiedono autorizzazione" e non eseguirla.
9. **Comandi lunghi**: i benchmark durano da minuti a un'ora. Lanciali SEMPRE staccati dalla shell, così sopravvivono se la shell viene chiusa (il primo tentativo di baseline è morto proprio per questo):
   ```bash
   setsid nohup bash -c '<comando>' > logs/<nome>.out 2>&1 < /dev/null &
   ```
   Poi controlla con `tail -3 logs/<nome>.out` a intervalli di qualche minuto (non ogni secondo).
10. **Un cambiamento per esperimento**: ogni esperimento cambia UNA variabile rispetto alla configurazione di riferimento corrente (eccezione: combinazioni esplicite nella Fase 6).
11. **Registra tutto**: dopo ogni esperimento aggiungi una voce a `EXPERIMENTS.md` (modello nella §9) PRIMA di passare al successivo.
12. **Quando non sei sicuro**: non indovinare un parametro. Verifica che esista con
    `bin/dk run --rm ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276 --help | grep -- '<flag>'`.

---

## 1. CONTESTO TECNICO GIÀ ACQUISITO (non rifarlo, ma usalo)

### 1.1 Ambiente
- Dettagli in `HARDWARE.md`. Punti critici: Ryzen 7 3700X 8C/16T, 32 GB DDR4, RTX 3080 10 GB, **PCIe Gen3 x16 (limite della scheda madre B450)**, ~0.9 GB di VRAM usati dal desktop.
- Immagine di riferimento (fissata per digest, il tag `server-cuda` si sposta):
  `ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276` = b11977, commit 7ac5727d6.
- Immagine più recente pubblicata trovata: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12124` (commit 8bed7cfb9, 2026-11-04). Include: #31391 (letture batched in memoria host mappata), #30141 (salvataggio stato MTP), #30662 (eviction grafi CUDA). Non include #31047 (layout MoE MMQ) né #31679.

### 1.2 Modello
- `qwen35moe`: 40 layer (30 Gated DeltaNet ricorrenti + 10 full attention) + 1 layer MTP.
- Esperti: 13 942 MiB (in RAM con `--cpu-moe`). Non-esperti: ~1962 MiB su GPU.
- KV f16: 3910 MiB (target) + 391 MiB (contesto MTP) a 200K.

### 1.3 Configurazione originale (riferimento da preservare)
```
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024
--flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe
```
Valori impliciti dai log: `n_batch=2048`, `n_ubatch=512`, `n_threads=8`, `load_mode=mmap`, KV f16.

### 1.4 Ipotesi già formulate dall'analisi del sorgente (da verificare con gli esperimenti)
| ID | Ipotesi | Motivo (verificato nel codice) |
|---|---|---|
| H1 | Il PP è limitato dal PCIe | Per ogni ubatch ≥ 32 token lo scheduler copia sulla GPU gli esperti usati da ogni layer (`llama_context::sched_copy_experts`). Con 512 token quasi tutti i 256 esperti sono usati: ~13.9 GB per ubatch. Misurato: 8192 tok in 18.7 s = 16 ubatch × 13.9 GB / 18.7 s ≈ 11.9 GB/s = banda PCIe Gen3. |
| H2 | Un `--ubatch-size` più grande moltiplica il PP | Stesso trasferimento ammortizzato su più token (1024 → ~2×, 2048 → ~4×, finché non diventa compute-bound). Costo: compute buffer più grande in VRAM. |
| H3 | `--load-mode none` velocizza PP e i miss della MoE cache | Con mmap gli esperti restano in memoria pageable (`llama-model-loader.cpp`: "avoid using a host buffer when using mmap"); senza mmap finiscono in `CUDA_Host` pinned → DMA reale. È il suggerimento del warning "tensor overrides to CPU are used with mmap enabled". Costo: ~14 GB di RAM bloccata (non swappabile). |
| H4 | KV `q8_0` libera ~2.1 GB di VRAM con perdita di qualità trascurabile | Dimensione a metà; FA CUDA supporta la coppia q8_0-q8_0 (`FA_QUANTS` nel log). La VRAM liberata serve per ubatch più grandi e/o MoE cache più grande. |
| H5 | MoE cache più grande ⇒ generazione più veloce | Generazione (ubatch ≤ 8) calcola gli esperti su GPU dalla cache; i miss vengono caricati via PCIe ad ogni token. Con 1024 MiB, i gruppi q6_K/q4_K hanno solo 17 slot ma la verifica MTP (ubatch 3 × 8 esperti = 24) ne richiede ≥ 24 → quei layer escono dalla cache (`get_slot_map` restituisce nullptr) e girano su CPU. |
| H5-dato | (misurato) hit rate baseline 14.5 %, 706 GB caricati in generazione | `logs/E00_baseline_server.log`. Generazione limitata dal PCIe: confronta sempre con `--moe-cache-mib 0` (esperti calcolati dalla CPU con banda DDR4) |
| H6 | `--spec-draft-n-max` 3 può battere 2 | Accettazione osservata 0.81–0.99. Da misurare: t/s effettivi, non solo accettazione. |
| H7 | `GGML_OP_OFFLOAD_MIN_BATCH` (env, default 32) influenza i turni agentici brevi | Prompt incrementali piccoli pagano ~1.2 s di trasferimento esperti; per 32–128 token la CPU potrebbe essere più veloce. |
| H8 | Build più recente (b12124) | Commit CUDA/MTP rilevanti (vedi 1.1). |
| H9 | `--threads` | Rilevante solo per la parte di calcolo che resta su CPU. Provare 8 (default) vs 7 vs 16. |
| H10 | `--cache-ram` / `--ctx-checkpoints` | Influenzano il riuso del prompt nelle sessioni agentiche su modello ibrido. Da verificare con `bench/session.py`. |

---

## 2. STRUMENTI DISPONIBILI (già scritti)

| File | Uso |
|---|---|
| `bin/dk` | wrapper di `docker` (usa `newgrp docker`) |
| `scripts/run_server.sh NAME [--image IMG] [--port P] [--docker "..."] -- <args>` | avvia un container su `127.0.0.1:P` (default 8081) e attende `/health` |
| `scripts/exp.sh TAG DEPTHS REPS [LONG_REPS] [--image IMG] [--docker "..."] -- <args>` | esperimento completo: avvio → `bench.py` → stop pulito → log in `logs/exp/TAG.log`, riepilogo in `logs/exp/index.txt`. Aggiunge `-lv 4`. Variabile env `GEN` (default 256) = token generati |
| `bench/bench.py --tag T --depths 8192,... --reps N --long-reps M --gen G` | prompt di lunghezza esatta (ID token dal corpus reale), `cache_prompt=false`, nonce casuale → sempre a freddo. Output: `results/bench_T.jsonl` |
| `bench/summarize.py [TAG ...]` | tabella markdown con mediane + `results/summary.csv` |
| `bench/agentic.py --tag T [--coding-reps N]` | tool calling (stream e non), multi-turno, 5 problemi di coding verificati da unit test. Output: `results/agentic_T.jsonl` |
| `bench/session.py --tag T --target 180000 --chunk 6000` | sessione agentica lunga stile OpenCode. Output: `results/session_T.jsonl` |

**Attenzione**: `scripts/exp.sh` usa sempre il nome container `llopt-exp` e lo rimuove alla fine. `agentic.py` e `session.py` non sono dentro `exp.sh`: per loro avvii il server con `run_server.sh`, lanci lo script, poi fermi il server a mano (§3.3).

---

## 3. PROCEDURE STANDARD (riusate in tutte le fasi)

### 3.1 Controllo prima di ogni esperimento
```bash
bin/dk ps --format '{{.Names}} {{.Status}}'      # nessun llopt-* deve essere in esecuzione
nvidia-smi --query-gpu=memory.used --format=csv   # atteso ~900-1000 MiB (solo desktop)
free -m                                           # "available" atteso > 24000
```
Se la VRAM usata è > 1500 MiB senza container attivi, un'altra applicazione (es. un gioco Steam) sta usando la GPU: annotalo in `EXPERIMENTS.md` e aspetta, perché i risultati non sarebbero confrontabili.

### 3.2 Esperimento di benchmark standard
```bash
setsid nohup bash -c 'scripts/exp.sh <TAG> <DEPTHS> <REPS> <LONG_REPS> -- <ARGS>' > logs/<TAG>.out 2>&1 < /dev/null &
```
Al termine:
```bash
tail -20 logs/exp/index.txt
python3 bench/summarize.py E00_baseline <TAG>
grep -E "n_ctx  |n_ubatch|n_batch|load_mode|KV buffer|MoE cache size|slots|compute buffer|llama_moe_cache:" logs/exp/<TAG>.log | head -40
```
Verifiche obbligatorie (se una fallisce → configurazione INVALIDA, scrivilo):
- `n_ctx = 200192` (o superiore) nel log.
- In `results/bench_<TAG>.jsonl` nessuna riga ha `"warning"` e `prompt_n` è uguale a `depth`.
- Nessun "CUDA error", "out of memory", "GGML_ABORT" nel log: `grep -iE "cuda error|out of memory|abort|failed to allocate" logs/exp/<TAG>.log`.

### 3.3 Avvio manuale per test agentici
```bash
scripts/run_server.sh llopt-agent -- <ARGS>
python3 bench/agentic.py --tag <TAG> --coding-reps 2
bin/dk stop -t 60 llopt-agent && bin/dk logs llopt-agent > logs/exp/<TAG>_agent.log 2>&1 && bin/dk rm llopt-agent
```

### 3.4 Set di profondità
- **SCREEN** (veloce, per scartare): `DEPTHS=8192,32768 REPS=3 LONG_REPS=1`, `GEN=256`. Durata ≈ 6–8 min a velocità baseline.
- **FULL** (solo per le configurazioni candidate): `DEPTHS=8192,16384,32768,65536,131072,196608 REPS=3 LONG_REPS=2`, `GEN=256`. Durata: 40–90 min.
- **MAX**: una singola misura a `199000` token (`DEPTHS=199000 REPS=1 LONG_REPS=1`, `GEN=128`) per dimostrare la capacità vicino al limite.

### 3.5 Regola di decisione (usala sempre, scrivila in EXPERIMENTS.md)
Confronta mediane della configurazione nuova con quelle del riferimento corrente sulle stesse profondità:
- **PP**: miglioramento reale se la mediana migliora ≥ 5 % e min(nuovo) > max(riferimento) ad almeno una profondità.
- **TG**: la TG varia molto con l'accettazione MTP (0.8–0.99 con lo stesso prompt). Miglioramento reale solo se la mediana migliora ≥ 5 % con almeno 3 ripetizioni per profondità. Tra 0 e 5 % → "neutro".
- **Peggioramento**: mediana peggiore di > 3 % su TG o > 5 % su PP a qualsiasi profondità ≥ 32K → scarta, a meno che l'altro indicatore migliori molto di più (es. PP +100 % e TG −3 %: tieni ma annota il compromesso).
- **VRAM**: `vram_max` deve restare ≤ 9900 MiB (margine per il desktop). Oltre → rischio OOM → scarta o riduci.
- **RAM**: `ram_avail_min` ≥ 3000 MiB e `swap_used_max` non deve crescere oltre 1000 MiB.

---

## FASE 0 — Ripristino dello stato e completamento della baseline

**Obiettivo**: baseline completa e riproducibile della configurazione originale.

0.1 Verifica cosa è in esecuzione:
```bash
bin/dk ps -a --format '{{.Names}} {{.Status}}'
```
Se `llopt-baseline` è "Up", fermalo e salva i log:
```bash
bin/dk stop -t 60 llopt-baseline; bin/dk logs llopt-baseline > logs/E00_baseline_server.log 2>&1; bin/dk rm llopt-baseline
grep "llama_moe_cache:" logs/E00_baseline_server.log   # hit rate MoE cache della baseline -> annotala
```

0.2 La baseline FULL era stata interrotta. Il file `results/bench_E00_baseline.jsonl` contiene già 8K (2 rep), 16K (2), 32K (2), 64K (1). **Non cancellarlo.** Rilancia la baseline completa con un tag nuovo, così tutte le profondità hanno le stesse condizioni:
```bash
setsid nohup bash -c 'scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe' > logs/E00b.out 2>&1 < /dev/null &
```
Durata attesa: ~70–90 min (192K a ~400 t/s ≈ 8 min per ripetizione).
Se fallisce a 192K per OOM: annotalo come limite della baseline (è un risultato!) e prosegui.

0.3 Test agentico baseline (§3.3, TAG `E00_agentic`, stessi argomenti). Annota: tool_call ok?, tool_call_stream ok?, multi_turn ok?, coding_pass x/10, tasks_per_min.
**Se `agentic.py` va in errore** (è stato scritto ma non collaudato): leggi il traceback, correggi lo script (bug Python, non il server), annota la correzione in `EXPERIMENTS.md`, rilancia. Non modificare i prompt né i test unitari dopo aver iniziato a confrontare configurazioni.

0.4 Sessione lunga baseline: avvia il server (§3.3, nome `llopt-agent`), poi
```bash
setsid nohup bash -c 'python3 bench/session.py --tag E00_session --target 180000 --chunk 6000' > logs/E00_session.out 2>&1 < /dev/null &
```
Durata: ~30 turni, 30–60 min. Annota per turno: `ttft`, `prompt_n` (token realmente rielaborati) vs `cache_n` (riusati). **Controllo critico**: `prompt_n` deve restare circa = chunk (~6000 + poche centinaia). Se a un certo turno `prompt_n` diventa quasi uguale a `ctx_tokens` il server ha riprocessato tutto il contesto (perdita cache): annotalo con il numero di turno; è un problema importante per l'uso agentico.
Al termine ferma il server e salva il log (`logs/exp/E00_session_server.log`).

0.5 Scrivi `BASELINE.md` (modello §10.1).

**Criterio di uscita Fase 0**: `BASELINE.md` completo con tabella FULL, esito agentico, esito sessione.

---

## FASE 1 — Prompt processing: dimensione dell'ubatch (H2)

**Collo di bottiglia**: trasferimento PCIe degli esperti per ogni ubatch.
**Metriche**: PP t/s a 8K e 32K; `vram_max`; TG non deve peggiorare.

Esperimenti SCREEN (ognuno cambia solo `-b`/`-ub`; tutto il resto = configurazione originale; `ARGS_BASE` = argomenti originali):
| TAG | argomenti aggiunti |
|---|---|
| E01_ub1024 | `-b 2048 -ub 1024` |
| E02_ub2048 | `-b 2048 -ub 2048` |
| E03_ub4096 | `-b 4096 -ub 4096` |

Esempio:
```bash
setsid nohup bash -c 'scripts/exp.sh E02_ub2048 8192,32768 3 1 -- --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-cpu-moe -b 2048 -ub 2048' > logs/E02.out 2>&1 < /dev/null &
```

**Atteso** (da verificare, non da copiare): PP quasi raddoppia a ogni raddoppio di ubatch finché il calcolo GPU non diventa dominante.
**Possibile fallimento**: OOM all'avvio o durante il prompt (compute buffer più grande; VRAM baseline già a ~9.6 GB). Se succede:
- annota la dimensione del `CUDA0 compute buffer size` dal log (se presente),
- NON ridurre il contesto; prova lo stesso ubatch nella Fase 2 dopo aver liberato VRAM con KV q8_0.

Per ogni ubatch che parte, annota anche `CUDA0 compute buffer size` (VRAM extra pagata).

**Decisione**: scegli l'ubatch con il PP più alto che resta sotto 9900 MiB di VRAM *a 32K*. Poi verifica che regga a 196608 con un singolo run:
```bash
scripts/exp.sh E0x_ubXXXX_long 196608 1 1 -- <ARGS> 
```
(lancialo con setsid/nohup). Se va in OOM a 196K → ubatch troppo grande per ora, torna a quello inferiore.

---

## FASE 2 — KV cache quantizzata (H4)

**Collo di bottiglia**: VRAM occupata dalla KV f16 (4.3 GB).
**Metriche**: VRAM (`KV buffer size` nel log, `vram_max`), PP e TG a 8K/32K/128K, **qualità** (agentic.py coding pass).

| TAG | argomenti aggiunti a ARGS_BASE (con l'ubatch migliore della Fase 1, oppure 512 se nessuno funzionava) |
|---|---|
| E10_kvq8 | `-ctk q8_0 -ctv q8_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0` |
| E11_kvq8_tgt_only | `-ctk q8_0 -ctv q8_0` (KV del draft MTP resta f16) |

Note:
- Le flag `--spec-draft-type-k/-v` esistono in questa versione (`common/arg.cpp` righe 4146–4170). Verifica con `--help` se l'avvio fallisce.
- **Non usare** `q4_0` per la KV come scelta finale senza prove di qualità molto solide (degrada il recupero a lungo contesto). Puoi testarlo come E12_kvq4 solo a fini documentali.
- Nel log verifica la riga `llama_kv_cache: size = ... K (q8_0) ..., V (q8_0)`.

**Test di qualità obbligatorio** per ogni KV quantizzata che si vuole tenere:
1. `bench/agentic.py --coding-reps 2` → coding_pass non deve scendere di più di 1 rispetto alla baseline (10 prove).
2. Test di recupero a lungo contesto ("needle"): vedi §FASE 2b.

### FASE 2b — Test needle a lungo contesto (scrivi tu lo script `bench/needle.py`)
Script semplice, da creare:
- Prende ~150 000 token dal corpus (`bench/corpus/code_corpus.tokens.json`), li detokenizza con `/detokenize` in 5 pezzi.
- Inserisce in 5 posizioni (10 %, 30 %, 50 %, 70 %, 90 %) una riga `// SECRET_CONFIG_<i> = "<parola casuale di 8 lettere>"`.
- Invia via `/v1/chat/completions` un messaggio user: `<testo>\n\nWhat are the exact values of SECRET_CONFIG_1 ... SECRET_CONFIG_5? Answer as JSON.` con `temperature 0`, `max_tokens 2000`, `chat_template_kwargs: {"enable_thinking": false}`.
- Punteggio = numero di valori esatti trovati nella risposta (0–5). Salva in `results/needle_<TAG>.jsonl`. Usa un seed fisso per le parole casuali (`random.seed(42)`), così ogni configurazione riceve lo stesso prompt.
- Esegui su baseline (f16) e su ogni KV quantizzata candidata. Accetta la KV quantizzata solo se il punteggio è uguale a quello f16 (o al massimo −1 su 5 ripetendo il test una seconda volta per conferma).

**Decisione Fase 2**: se q8_0 supera i test di qualità ed è neutra o migliore in velocità → diventa il nuovo riferimento (la VRAM liberata è il vero guadagno, da spendere nelle Fasi 3–4).

---

## FASE 3 — Ri-espansione dell'ubatch con la VRAM liberata

Solo se la Fase 2 ha adottato q8_0. Ripeti la Fase 1 per l'ubatch che prima andava in OOM (es. 4096) con `-ctk q8_0 -ctv q8_0 ...`. TAG `E20_kvq8_ub4096`, ecc. Stessa decisione della Fase 1, inclusa la prova a 196608.

---

## FASE 4 — Memoria pinned: `--load-mode none` (H3)

**Collo di bottiglia**: copie H2D da memoria pageable (mmap).
**Metriche**: PP (deve aumentare se H3 è vera), TG, tempo di caricamento (`load_s` in `logs/exp/index.txt`), RAM disponibile.

| TAG | argomenti aggiunti al riferimento corrente |
|---|---|
| E30_loadnone | `--load-mode none` |
| E31_loadnone_mlock | `--load-mode mlock` (se `none` è migliore; potrebbe richiedere `--docker "--ulimit memlock=-1:-1"` nello script) |

Verifiche nel log:
- `load_tensors: loading model tensors ... (load_mode = none)`.
- Le righe `CPU_Mapped model buffer size` devono diventare `CUDA_Host model buffer size`. Se restano CPU/CPU_Mapped, H3 non si applica: annotalo.
- Il warning "tensor overrides to CPU are used with mmap enabled" deve sparire.
- `free -m` dopo l'avvio (`logs/exp/<TAG>.mem_idle.txt`): con ~14 GB pinned la RAM disponibile scenderà. Se `available` < 6000 MiB, annota il rischio per l'uso quotidiano (browser, IDE).
- Stabilità: dopo la prova SCREEN esegui anche `bench/session.py` per almeno 60 000 token con questa configurazione (memoria pinned + lunga sessione).

**Se fallisce l'avvio** ("failed to allocate CUDA_Host buffer" o simile): prova `--load-mode none --no-host` (in questo caso gli esperti vanno in RAM normale non pinned, utile solo per confronto) e annota.

---

## FASE 5 — Generazione: MoE cache (H5) e MTP (H6)

### 5.1 Dimensione MoE cache
Riferimento corrente = migliore finora. Cambia solo `--moe-cache-mib`. Usa SCREEN ma con `REPS=4` (la TG è rumorosa) e `GEN=512`:
| TAG | valore |
|---|---|
| E40_mc0 | `0` (cache disattivata: esperti su CPU in generazione) — importante per capire se la cache aiuta davvero |
| E41_mc1536 | `1536` |
| E42_mc2048 | `2048` |
| E43_mc3072 | `3072` (solo se la VRAM lo permette) |
| E44_mcMAX | il massimo che lascia `vram_max` ≤ 9900 MiB a 32K |

Per ognuno annota dal log:
- le righe `impl: N layers, <tipo>: <slot> slots (x%)` → verifica che ogni gruppo abbia **≥ 24 slot** (= (n_max+1)×8). Con n_max=3 servono ≥ 32.
- allo stop: `llama_moe_cache: ubatch <= 8: hits = ..., misses = ..., hit rate = ...%, uploaded = ... MiB`.
Crea una tabella cache-MiB → hit rate → TG mediana. Atteso: TG cresce con l'hit rate finché la VRAM lo consente.

**Attenzione VRAM**: MoE cache + ubatch grande competono. Se l'ubatch grande costringe a una cache piccola, valuta il compromesso con i dati di entrambe le metriche e la sessione agentica (§FASE 7). Documenta la scelta.

**Non usare `--no-kv-offload` insieme alla MoE cache**: causa l'abort noto "the experts of layer 0 are selected in the same split as their MoE cache lookup" (vedi `src/llama.cpp/src/llama-moe-cache.cpp:560`). Se vuoi documentarlo, un solo tentativo, poi scarta.

### 5.2 Alternativa: layer di esperti interi su GPU invece della cache
Esperimento E45: sostituisci `--cpu-moe` con `--n-cpu-moe N` (es. N=37 → i layer 37–39 hanno gli esperti in VRAM) e riduci la cache di conseguenza (~332 MiB per layer). Confronta con la cache della stessa dimensione totale. Verifica con `--help` che `--n-cpu-moe` esista (esiste: `common/arg.cpp` riga 2779). Nota: `--n-cpu-moe N` tiene su CPU i *primi* N layer.

### 5.3 MTP draft length
Riferimento = migliore dopo 5.1. Cambia solo `--spec-draft-n-max`. `REPS=4`, `GEN=512`, profondità 8192,32768,131072 (131072 con 1 rep):
| TAG | valore |
|---|---|
| E50_nmax1 | 1 |
| E51_nmax3 | 3 |
| E52_nmax4 | 4 |
Metrica principale: TG t/s mediana (non l'accettazione). Con n_max 3/4 controlla che la MoE cache abbia ≥ (n_max+1)×8 slot per ogni gruppo, altrimenti aumentala (e annotalo).
Esperimento opzionale E53: `--spec-draft-p-min 0.5` (o 0.75) con il migliore n_max: interrompe il draft quando la confidenza è bassa.
Esperimento E54: rimuovere `--spec-draft-cpu-moe` (il draft MTP non ha esperti propri, quindi probabilmente è neutro: verificalo e documentalo; se neutro, tienilo come nell'originale).

---

## FASE 6 — Parametri secondari

Uno alla volta sul riferimento corrente, SCREEN con `REPS=3`:
| TAG | modifica | cosa guardare |
|---|---|---|
| E60_t7 | `--threads 7` | TG |
| E61_t16 | `--threads 16 --threads-batch 16` | TG/PP |
| E62_minb64 | `--docker "-e GGML_OP_OFFLOAD_MIN_BATCH=64"` | solo prompt brevi: usa `DEPTHS=64,128,256 REPS=5`, confronta con lo stesso test sul riferimento (TAG `E62_ref`) |
| E63_minb128 | `--docker "-e GGML_OP_OFFLOAD_MIN_BATCH=128"` | come sopra |
| E64_prio | `--prio 2` | TG (può non avere effetto in container senza privilegi: annota) |
| E65_poll | `--poll 0` | TG e uso CPU |

Per i test su prompt brevi (E62/E63), `bench.py` accetta qualsiasi profondità, incluse quelle piccole.

---

## FASE 7 — Runtime più recente (H8)

7.1 Scarica l'immagine con tag esplicito (NON fare pull di `server-cuda`, sposterebbe il tag dell'originale):
```bash
setsid nohup bash -c 'bin/dk pull ghcr.io/ggml-org/llama.cpp:server-cuda-b12124' > logs/pull_b12124.out 2>&1 < /dev/null &
```
Annota il digest: `bin/dk inspect --format '{{index .RepoDigests 0}} {{index .Config.Labels "org.opencontainers.image.revision"}}' ghcr.io/ggml-org/llama.cpp:server-cuda-b12124`.

7.2 Verifica che tutte le flag della configurazione migliore esistano nella nuova immagine:
```bash
bin/dk run --rm ghcr.io/ggml-org/llama.cpp:server-cuda-b12124 --help > logs/help_b12124.txt 2>&1
```
e cerca ogni flag usata con `grep -- '--flag' logs/help_b12124.txt`.

7.3 Esegui SCREEN con la configurazione migliore e `--image ghcr.io/ggml-org/llama.cpp:server-cuda-b12124` (TAG `E70_b12124`), poi `agentic.py` (TAG `E70_agentic`).
Decisione: adotta la nuova build solo se ≥ 5 % meglio su PP o TG **e** i test agentici passano tutti **e** la sessione lunga (Fase 8) non mostra regressioni. Altrimenti resta su b11977 (più collaudata dall'utente).

7.4 (Opzionale, solo se tutto il resto è fatto) Build personalizzata con `-DCMAKE_CUDA_ARCHITECTURES=86`. Richiede di scrivere un Dockerfile basato su `nvidia/cuda:12.8.1-devel-ubuntu24.04` e sui Dockerfile ufficiali in `src/llama.cpp/.devops/cuda.Dockerfile`. Salva il Dockerfile in `docker/`. Il guadagno atteso è piccolo; documentalo comunque come esperimento.

---

## FASE 8 — Validazione completa dei finalisti

Seleziona al massimo 2 configurazioni finaliste (la migliore + l'alternativa più conservativa, es. senza `--load-mode none`).

Per ciascuna:
1. **FULL** benchmark (§3.4) — TAG `F1_full`, `F2_full`.
2. **MAX** a 199000 token — TAG `F1_max`. Annota TTFT e TG.
3. `agentic.py --coding-reps 2` — TAG `F1_agentic`.
4. `needle.py` — TAG `F1_needle`.
5. `session.py --target 180000` — TAG `F1_session`. Controlla: nessuna rielaborazione completa del contesto, TTFT per turno, TG nei turni finali.
6. **Stabilità**: ripeti `session.py` una seconda volta senza riavviare il server (sessione prolungata, ~1–2 ore totali). Alla fine: `grep -iE "error|abort|oom" logs`, VRAM e RAM finali, swap usato. Nessun errore ammesso.
7. **Riavvio riproducibile**: ferma e riavvia il container 2 volte, verifica che il primo test SCREEN dia risultati entro ±5 %.

Confronta F1/F2 con E00b nella tabella di `RESULTS.md`.

---

## FASE 9 — Messa in produzione (sostituzione della configurazione originale)

Esegui SOLO dopo la Fase 8 superata.

9.1 Crea `docker/compose.yaml` (o script `scripts/prod_start.sh` / `scripts/prod_stop.sh`) con:
- nome container `qwen36-35b-opt`,
- immagine fissata per digest,
- `-p 127.0.0.1:8080:8080` (solo loopback, stessa porta dell'originale),
- `-v ~/AI/llama-models:/models:ro`,
- `--restart unless-stopped` solo se l'utente lo desidera (di default: `no`, come l'originale),
- tutti gli argomenti della configurazione finale (senza `-lv 4`).

9.2 Procedura di sostituzione (il container originale NON va cancellato):
```bash
bin/dk ps -a --format '{{.Names}} {{.Status}}'   # qwen36-35b deve essere fermo
scripts/prod_start.sh                             # avvia qwen36-35b-opt su 8080
curl -s http://127.0.0.1:8080/v1/models
python3 bench/agentic.py --url http://127.0.0.1:8080 --tag PROD_check
```
Se qualcosa fallisce: `scripts/prod_stop.sh` e riavvia l'originale con `bin/dk start qwen36-35b`. Verifica `curl http://127.0.0.1:8080/v1/models`.

9.3 Scrivi in `OPTIMIZATION.md` le istruzioni di avvio/arresto/rollback e i valori da impostare nei client (OpenCode `limit.context: 200000`, `output` adeguato; Cline idem).

---

## FASE 10 — Documentazione finale (obbligatoria)

Aggiorna/crea tutti i file. Ogni numero deve citare il file sorgente (es. "fonte: `results/bench_F1_full.jsonl`").

### 10.1 `BASELINE.md`
- Configurazione originale esatta (copia da `baseline/original/qwen36-35b.inspect.json`: Cmd, immagine, digest, mount, porte).
- Ripartizione VRAM dal log (modello GPU, KV, MTP KV, RS, MoE cache, compute buffer).
- Tabella FULL E00b (output di `bench/summarize.py E00b_baseline`).
- Risultati agentici e sessione lunga baseline.
- Osservazioni dai log della sessione reale originale (`logs/original_full_run.log`): TG ~26 t/s a ~150K, prompt incrementali ~350 t/s, TTFT ~7 s per ~2.4K token.
- Spiegazione dei due warning originali:
  - "tensor overrides to CPU are used with mmap enabled": con mmap gli esperti non possono usare il buffer pinned `CUDA_Host` → testato in Fase 4.
  - "failed to fit params ... tensor_buft_overrides already set by user": `--fit` si arrende quando si usa `--cpu-moe`; è innocuo (il contesto resta quello richiesto) ma significa che il posizionamento è manuale.

### 10.2 `EXPERIMENTS.md` (registro cronologico, una voce per esperimento, modello §9 sotto)

### 10.3 `RESULTS.md`
- Tabella principale: righe = configurazioni (E00b, migliori di ogni fase, F1, F2), colonne = PP e TG a 8K/32K/128K/192K, TTFT a 32K e 128K, VRAM max, RAM disponibile min, coding pass, needle, esito sessione.
- Tabella MoE cache (MiB → slot min per gruppo → hit rate → TG).
- Tabella ubatch (ubatch → compute buffer → PP 8K/32K).
- Tabella MTP (n_max → accettazione → TG).
- `results/summary.csv` aggiornato (`python3 bench/summarize.py`).

### 10.4 `OPTIMIZATION.md`
- Configurazione finale completa, riga per riga, con il motivo di ogni flag e l'esperimento che la giustifica.
- Guadagni rispetto alla baseline (% per PP, TG, TTFT, lavoro agentico per minuto).
- Limiti dimostrati (es. PCIe Gen3: calcola dai dati la banda effettiva = byte esperti trasferiti / tempo prompt).
- Cose provate e scartate, con motivo.
- "Proposte che richiedono autorizzazione" (es. governor `performance`, persistence mode NVIDIA, Resizable BAR nel BIOS, upgrade a scheda madre con PCIe Gen4) — NON eseguite.
- Istruzioni avvio/arresto/rollback.

### 10.5 `README.md`
Missione, vincoli, mappa dei file, come riprodurre un benchmark in 3 comandi.

### 10.6 `PROGRESS.md`
Aggiornalo alla fine di **ogni fase** (spunta ciò che è fatto, cosa resta, stato dei container).

---

## 9. MODELLO PER OGNI VOCE DI `EXPERIMENTS.md`

```markdown
### E02_ub2048 — 2026-11-06 10:15
- **Ipotesi**: H2 — ubatch 2048 riduce di 4× i trasferimenti PCIe per token di prompt.
- **Collo di bottiglia affrontato**: PCIe H2D degli esperti durante il prompt processing.
- **Modifica (unica)**: `-b 2048 -ub 2048` rispetto a <TAG riferimento>.
- **Comando**: `scripts/exp.sh E02_ub2048 8192,32768 3 1 -- ...`
- **Verifiche**: n_ctx=200192 ✔ | prompt_n=depth ✔ | errori nel log: nessuno ✔
- **Memoria**: CUDA0 compute buffer = ... MiB (rif. 565.5) | vram_max = ... MiB | RAM avail min = ... MiB
- **Risultati (mediane, n=3)**:
  | depth | PP rif | PP nuovo | Δ% | TG rif | TG nuovo | Δ% | acc |
  |---|---|---|---|---|---|---|---|
- **Decisione**: TENUTO / SCARTATO / NEUTRO — motivo secondo §3.5.
- **File**: results/bench_E02_ub2048.jsonl, logs/exp/E02_ub2048.log
```

---

## 10. CHECKLIST FINALE (copiala in fondo a PROGRESS.md e spunta)

- [ ] E00b baseline FULL completa (8K→192K)
- [ ] Baseline agentica + sessione lunga
- [ ] Fase 1 ubatch (1024/2048/4096)
- [ ] Fase 2 KV q8_0 + qualità (coding + needle)
- [ ] Fase 3 ubatch con KV q8_0
- [ ] Fase 4 load-mode none / mlock
- [ ] Fase 5.1 MoE cache sweep con hit rate
- [ ] Fase 5.2 n-cpu-moe vs cache
- [ ] Fase 5.3 MTP n_max 1/3/4 (+ p_min)
- [ ] Fase 6 thread / offload min batch / prio / poll
- [ ] Fase 7 runtime b12124
- [ ] Fase 8 finalisti: FULL, MAX 199K, agentic, needle, sessione ×2, riavvio
- [ ] Fase 9 produzione su 127.0.0.1:8080, originale preservato, rollback provato
- [ ] Fase 10 tutti i documenti: README, HARDWARE, BASELINE, EXPERIMENTS, RESULTS, OPTIMIZATION, PROGRESS
- [ ] Nessun container sperimentale lasciato in esecuzione (`bin/dk ps`)

---

## 11. ERRORI COMUNI DA EVITARE

1. Confrontare run con `GEN` diversi, o una profondità con 1 rep contro una con 3, senza dichiararlo.
2. Dichiarare un miglioramento della TG da una sola ripetizione (l'accettazione MTP varia da 0.8 a 0.99).
3. Lasciare un container `llopt-*` acceso e avviarne un altro (OOM o numeri falsati).
4. Dimenticare che il desktop usa ~0.9 GB di VRAM: se l'utente apre un gioco durante i test, i numeri cambiano.
5. Usare `docker` invece di `bin/dk` (permesso negato).
6. Fare `docker pull ...:server-cuda` (sposta il tag usato dalla configurazione originale).
7. Riscrivere a mano numeri nei documenti: generali sempre con `bench/summarize.py` e copia la tabella.
8. Lanciare benchmark lunghi in primo piano o come semplici processi in background della shell: usa `setsid nohup`.
9. Scegliere una configurazione che regge a 32K ma va in OOM a 192K: la prova a 196608 è obbligatoria prima di adottarla.
10. Considerare l'accettazione MTP come obiettivo: l'obiettivo sono i token/s e il lavoro utile per minuto.
