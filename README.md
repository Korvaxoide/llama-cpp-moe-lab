# llama-optimizer-lab

Laboratorio di ottimizzazione di **llama.cpp** su una singola workstation
(RTX 3080 10 GB + Ryzen 7 3700X, PCIe Gen3 x16), con un solo modello:
`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (MoE ibrido Qwen3.6, contesto 262 144).

Risultato: la configurazione di produzione (`qwen36-35b-opt`) rispetto alla
configurazione originale misurata:

| Metrica | Baseline | Produzione | Δ |
|---|---|---|---|
| Prompt processing 8K / 192K | 445 / 386 t/s | **1473 / 1031 t/s** | **+231 % / +167 %** |
| Generazione 8K / 32K / 192K | 31.4 / 28.5 / 28.9 t/s | **46.9 / 49.7 / 35.0 t/s** | **+49 % / +74 % / +21 %** |
| TTFT 32K / 128K | 73.9 / 328.6 s | **23.0 / 113.3 s** | **−69 % / −66 %** |
| Lavoro agentico | 6.7 task/min | **8.77 task/min** | **+31 %** |
| Qualità (coding / needle) | 10/10 / 5/5 | **10/10 / 5/5** | invariata |

Dettagli e fonti: [`RESULTS.md`](RESULTS.md), [`OPTIMIZATION.md`](OPTIMIZATION.md).

---

## Missione

Trovare, con misure riproducibili e una variabile per volta, la configurazione
migliore di `llama-server` per questo modello su questo hardware, **senza ridurre
il contesto sotto 200 000 token** e senza degradare la qualità. Ogni decisione è
presa secondo la regola §3.5 di [`PLAN_FOR_AGENT.md`](PLAN_FOR_AGENT.md): un
guadagno ≥5 % su PP o TG, oppure neutralità dichiarata; i peggioramenti sono
scartati anche se "sembrano" ragionevoli.

## Vincoli del laboratorio

- Un solo modello: `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`. **Mai** copiato,
  convertito o modificato (mount read-only).
- Contesto sempre ≥ 200 000: `--ctx-size 200000` → `n_ctx = 200192` verificato nei log.
- Un solo server LLM alla volta (10 GB di VRAM), porte pubblicate **solo su
  `127.0.0.1`**.
- Sempre `bin/dk` al posto di `docker` (aggiunge il gruppo docker al processo).
- Niente `sudo`, niente modifiche a governor/swap/zram/driver/BIOS. Le idee che
  li richiedono sono solo proposte in `OPTIMIZATION.md` §6.
- Il container originale `qwen36-35b` e l'immagine b11977 non vengono mai
  cancellati; nessun `docker prune`, nessun pull dell'immagine `:server-cuda`
  (si usa il digest pinnato).

## Mappa dei file

| Percorso | Contenuto |
|---|---|
| `PLAN_FOR_AGENT.md` | piano autoritativo, ipotesi, criteri di decisione, checklist |
| `PROGRESS.md` | stato del lavoro, fase per fase, e checklist finale |
| `HARDWARE.md` | hardware rilevato (CPU, PCIe, RAM, GPU, modello) |
| `BASELINE.md` | configurazione originale, ripartizione VRAM, baseline misurata |
| `EXPERIMENTS.md` | registro cronologico: una voce per esperimento, con decisione |
| `RESULTS.md` | tabelle consolidate (principale, ubatch, cache, n-cpu-moe, MTP, qualità) |
| `OPTIMIZATION.md` | configurazione finale riga per riga, guadagni, limiti, rollback, client |
| `bench/bench.py` | benchmark PP/TG/TTFT/VRAM a profondità multiple |
| `bench/agentic.py` | tool calling, multi-turn, coding (10 prove), task/min |
| `bench/session.py` | sessione lunga a chunk da 6 000 token (riuso del prompt) |
| `bench/needle.py` | 5 aghi a ~150K token (qualità del contesto lungo) |
| `bench/summarize.py` | mediane dai `results/*.jsonl` → tabella + `results/summary.csv` |
| `bench/corpus/` | corpus di prompt usato dai benchmark |
| `scripts/exp.sh` | esperimento completo: server → bench → stop → log |
| `scripts/run_server.sh` | server sperimentale isolato su `127.0.0.1:8081` |
| `scripts/queue.sh` | esecuzione sequenziale di più esperimenti |
| `scripts/prod_start.sh` | avvio produzione `qwen36-35b-opt` su `127.0.0.1:8080` |
| `scripts/prod_stop.sh` | arresto produzione (container preservato) |
| `results/` | tutti i dati grezzi (`bench_*.jsonl`, `agentic_*`, `session_*`, `needle_*`) + `summary.csv` |
| `logs/exp/` | log dei server e indice degli esperimenti (`index.txt`) |
| `baseline/original/` | inspect JSON del container originale e dell'immagine |
| `analysis/` | metadati GGUF e script di ispezione |
| `bin/dk` | wrapper `docker` (newgrp docker) |

## Riprodurre un benchmark in 3 comandi

I comandi vanno eseguiti dalla radice del laboratorio. Il server sperimentale
ascolta su `127.0.0.1:8081`, così non tocca la produzione su `8080`.

```bash
# 1) avvia un server isolato con la configurazione finale (F1)
scripts/run_server.sh llopt-exp -- \
  --ctx-size 200000 --n-predict -1 --n-gpu-layers auto \
  --n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe \
  -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --load-mode none

# 2) esegui il benchmark (3 rep fino a 32K, 2 rep oltre)
python3 bench/bench.py --tag MIO_RUN --depths 8192,32768,131072,196608 --reps 3 --long-reps 2

# 3) calcola le mediane (tabella a schermo + results/summary.csv)
python3 bench/summarize.py MIO_RUN
```

Alla fine ferma e rimuovi il server sperimentale:

```bash
bin/dk stop -t 60 llopt-exp && bin/dk rm llopt-exp
```

In alternativa `scripts/exp.sh MIO_RUN 8192,32768,131072,196608 3 2 -- <flags>`
fa tutto in un comando (avvio, benchmark, stop, salvataggio log in `logs/exp/`).
I benchmark lunghi vanno lanciati in background e controllati con `tail`:

```bash
setsid nohup bash -c 'scripts/exp.sh MIO_RUN 8192,32768,131072,196608 3 2 -- <flags>' \
  > logs/MIO_RUN.out 2>&1 < /dev/null &
tail -3 logs/MIO_RUN.out
```

## Produzione

```bash
scripts/prod_start.sh     # qwen36-35b-opt su http://127.0.0.1:8080
scripts/prod_stop.sh      # arresta, container preservato
```

Rollback all'originale: `scripts/prod_stop.sh && bin/dk start qwen36-35b`
(poi `bin/dk stop qwen36-35b && scripts/prod_start.sh` per tornare
all'ottimizzato). Il rollback è stato provato in Fase 9. Non avviare mai i due
container insieme: la VRAM non basta.

Client (OpenCode, Cline, …): endpoint `http://127.0.0.1:8080/v1`,
`context = 200000`, output ≥ 8192. Dettagli in `OPTIMIZATION.md` §8.

## Nota sui dati

Nessun numero in questo repository è inventato. Ogni tabella deriva da
`results/*.jsonl` o da righe di `logs/exp/*`; le mediane sono generate con
`python3 bench/summarize.py`, che riscrive anche `results/summary.csv`. Dove un
valore è un calcolo derivato (es. banda PCIe) la formula è mostrata accanto.
