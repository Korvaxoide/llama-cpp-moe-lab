# llama-cpp-moe-lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Korvaxoide/llama-cpp-moe-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Korvaxoide/llama-cpp-moe-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.x%20%C2%B7%20stdlib-blue)

Laboratorio di ottimizzazione di **llama.cpp** su una singola workstation
(RTX 3080 10 GB + Ryzen 7 3700X, PCIe Gen3 x16), con un solo modello:
`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (MoE ibrido Qwen3.6, contesto 262 144).

> **TL;DR (English).** Reproducible tuning of `llama-server` for a 35B MoE model on a
> single 10 GB GPU with a 200K-token context. One variable per experiment, every number
> traced to `results/*.jsonl` or `logs/`. Result versus the measured baseline:
> prompt processing **+292 % at 8K** and **+213 % at 192K**, generation **+60…+75 %**,
> time-to-first-token **−75 %**, with coding quality unchanged (10/10).
> The dominant bottleneck was **PCIe Gen3 x16**, not the GPU, and the main levers were
> batch size, KV quantization and MoE placement. ReBAR was measured and gave **no benefit**.
> Docs are in Italian; the data and scripts are language-neutral.

## Risultati

Configurazione di produzione (`qwen36-35b-opt`, profilo **F1r**) rispetto alla
configurazione originale misurata sullo stesso hardware:

| Metrica | Baseline | Produzione (F1r) | Δ |
|---|---|---|---|
| Prompt processing 8K / 192K | 445 / 386 t/s | **1745 / 1209 t/s** | **+292 % / +213 %** |
| Generazione 8K / 32K / 192K | 31.4 / 28.5 / 28.9 t/s | **50.1 / 50.1 / 32.7 t/s** | **+60 % / +75 % / +13 %** |
| TTFT 32K / 128K | 73.9 / 328.6 s | **18.4 / 93.9 s** | **−75 % / −71 %** |
| Lavoro agentico | 6.7 task/min | **8.82 task/min** | **+32 %** |
| Qualità (coding 10 prove) | 10/10 | **10/10** | invariata |

Fonti: [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md).

## Cosa dimostrano i test

- **Il collo di bottiglia era il bus, non la GPU.** La baseline era satura su PCIe
  Gen3 x16 (~12.7 GB/s): ogni ubatch grande deve trasferire gli esperti MoE dalla RAM.
  Il calcolo della banda è in [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) §4.1.
- **L'ubatch è la leva principale sul prompt processing.** 512 → 1024 → 2048 → 3072
  vale circa +69 %, +71 % e +18…25 % per passo, finché la VRAM lo consente.
- **La cache MoE danneggia la generazione.** Con cache a 1024 MiB la TG era 32.3 t/s;
  con cache 0 è 42.2 t/s, perché ogni miss costa un trasferimento PCIe mentre la CPU
  calcola gli esperti dalla DDR4 più in fretta.
- **KV q8_0 non è una velocizzazione, è un abilitatore.** Libera 1.8 GiB di VRAM che
  servono per l'ubatch più grande e per tenere 3 layer di esperti sulla GPU
  (`--n-cpu-moe 37`), senza perdita di qualità misurata (needle 5/5, coding 10/10).
- **Il disaccoppiamento dell'ubatch MTP libera 883 MiB** (`--spec-draft-n-ubatch 512`):
  il buffer compute del contesto draft scende da 942 a 235 MiB al costo dell'1 % di PP.
- **ReBAR non serve** (test A/B dedicato): non alza la banda Gen3 e riduce la VRAM
  disponibile di ~300 MiB. Documentato in [`docs/RESULTS.md`](docs/RESULTS.md) §1.3.

## Missione

Trovare, con misure riproducibili e una variabile per volta, la configurazione
migliore di `llama-server` per questo modello su questo hardware, **senza ridurre
il contesto sotto 200 000 token** e senza degradare la qualità. Ogni decisione è
presa secondo la regola §3.5 di [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): un
guadagno ≥ 5 % su PP o TG, oppure neutralità dichiarata; i peggioramenti sono
scartati anche se "sembrano" ragionevoli.

## Vincoli del laboratorio

- Un solo modello: `/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`. **Mai** copiato,
  convertito o modificato (mount read-only).
- Contesto sempre ≥ 200 000: `--ctx-size 200000` → `n_ctx = 200192` verificato nei log.
- Un solo server LLM alla volta (10 GB di VRAM), porte pubblicate **solo su
  `127.0.0.1`**.
- Sempre `bin/dk` al posto di `docker` (aggiunge il gruppo docker al processo).
- Niente `sudo`, niente modifiche a governor/swap/zram/driver/BIOS. Le idee che
  li richiedono sono solo proposte in [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) §6.
- Il container originale `qwen36-35b` e l'immagine b11977 non vengono mai
  cancellati; nessun `docker prune`, nessun pull dell'immagine `:server-cuda`
  (si usa il digest pinnato).

## Struttura della repo

```
.
├── README.md
├── LICENSE
├── docs/                  documentazione (metodo, dati, configurazione)
│   ├── METHODOLOGY.md     piano operativo usato dall'agente esecutore
│   ├── HARDWARE.md        hardware rilevato (CPU, PCIe, RAM, GPU, modello)
│   ├── BASELINE.md        configurazione originale e baseline misurata
│   ├── EXPERIMENTS.md     registro cronologico, una voce per esperimento
│   ├── RESULTS.md         tabelle consolidate e confronti
│   ├── OPTIMIZATION.md    configurazione finale, limiti, rollback, client
│   └── PROGRESS.md        stato del lavoro e checklist
├── bench/                 benchmark (solo standard library)
│   ├── bench.py           PP / TG / TTFT / VRAM a profondità multiple
│   ├── agentic.py         tool calling, multi-turn, coding, task/min
│   ├── session.py         sessione lunga con riuso del prompt
│   ├── needle.py          5 aghi a ~150K token
│   ├── summarize.py       mediane dai JSONL → tabella + summary.csv
│   └── corpus/            corpus usato dai test
├── scripts/
│   ├── exp.sh             esperimento completo: server → bench → stop → log
│   ├── run_server.sh      server sperimentale isolato su 127.0.0.1:8081
│   ├── queue.sh           esecuzione sequenziale di più esperimenti
│   ├── prod_start.sh      avvio produzione qwen36-35b-opt su 127.0.0.1:8080
│   ├── prod_stop.sh       arresto produzione (container preservato)
│   └── check_repo.py      validazione dati e link (usata dalla CI)
├── results/               dati grezzi JSONL + summary.csv (schema in results/README.md)
├── logs/                  log grezzi dei server (evidenza; vedi logs/README.md)
├── baseline/original/     inspect JSON del container originale e dell'immagine
├── analysis/              metadati GGUF e script di ispezione
└── bin/dk                 wrapper docker (newgrp docker)
```

## Quick start

Prerequisiti: Linux, Docker, NVIDIA Container Toolkit, una GPU con abbastanza VRAM
(qui 10 GB) e il file GGUF scaricato **da te** (non è incluso nella repo).

```bash
git clone https://github.com/Korvaxoide/llama-cpp-moe-lab.git
cd llama-cpp-moe-lab
mkdir -p ~/AI/llama-models && cp /percorso/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf ~/AI/llama-models/
```

Poi, dalla radice:

```bash
# server sperimentale isolato su 127.0.0.1:8081 con la configurazione finale
scripts/run_server.sh llopt-exp -- \
  --ctx-size 200000 --n-predict -1 --n-gpu-layers auto \
  --n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe \
  --spec-draft-n-ubatch 512 \
  -ctk q8_0 -ctv q8_0 -b 3072 -ub 3072 --load-mode none

# 3 ripetizioni fino a 32K, 2 oltre
python3 bench/bench.py --tag MIO_RUN --depths 8192,32768,131072,196608 --reps 3 --long-reps 2

# mediane a schermo + results/summary.csv
python3 bench/summarize.py MIO_RUN

bin/dk stop -t 60 llopt-exp && bin/dk rm llopt-exp
```

In alternativa `scripts/exp.sh MIO_RUN 8192,32768,131072,196608 3 2 -- <flags>`
fa tutto in un comando. I benchmark lunghi vanno lanciati in background:

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
`context = 200000`, output ≥ 8192. Dettagli in [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) §8.
Se la macchina si usa anche per giocare, vedi §9 (convivenza con desktop e gaming).

## Dati e riproducibilità

Nessun numero in questo repository è inventato. Ogni tabella deriva da
`results/*.jsonl` o da righe di `logs/exp/*`; le mediane sono generate con
`python3 bench/summarize.py`, che riscrive anche `results/summary.csv`. Dove un
valore è un calcolo derivato (es. banda PCIe) la formula è mostrata accanto.
Lo schema dei file di dati è in [`results/README.md`](results/README.md); cosa
contengono i log è in [`logs/README.md`](logs/README.md).

La CI verifica a ogni push che tutti i JSONL siano validi, che gli script Python
compilino e che i link relativi dei documenti esistano
(`python3 scripts/check_repo.py`).

## Limiti

- I risultati valgono per **questo** hardware e **questo** runtime (RTX 3080 10 GB,
  Ryzen 7 3700X, PCIe Gen3 x16, llama.cpp b11977, quel GGUF): sono un metodo e un
  confronto, non una garanzia su altre macchine.
- ubatch oltre 3072 e `--n-cpu-moe 36` non sono stati adottati per margine VRAM
  insufficiente, non perché siano lenti.
- Il needle test non è stato ripetuto sulla F1r: rispetto alla F1 cambiano solo
  ubatch e buffer MTP, non la precisione della KV né i pesi.
- Il "task/min" dipende dai task del benchmark, non è un indice assoluto.

## Licenza e disclaimer

Codice e documentazione: [MIT](LICENSE).

I pesi del modello **non** sono distribuiti in questa repo: il GGUF resta fuori dal
versionamento e viene montato in sola lettura. I path personali sono stati sostituiti
con `$HOME` in vista della pubblicazione; i log sono riportati come prodotti dagli
strumenti, con le sole sostituzioni dei path.
