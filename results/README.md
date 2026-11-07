# results/ — dati grezzi

Un file JSONL per esperimento (una riga JSON per misurazione), più il CSV
consolidato. Tutti i numeri citati nei documenti vengono da qui: nessun valore è
inserito a mano.

## Convenzione dei nomi

| Pattern | Prodotto da | Contenuto |
|---|---|---|
| `bench_<TAG>.jsonl` | `bench/bench.py` | prompt processing, generazione, TTFT, VRAM/RAM a profondità multiple |
| `agentic_<TAG>.jsonl` | `bench/agentic.py` | tool calling, multi-turn, coding, task/min |
| `session_<TAG>.jsonl` | `bench/session.py` | turni di una sessione lunga con riuso del prompt |
| `needle_<TAG>.jsonl` | `bench/needle.py` | recupero di 5 aghi in ~150K token |
| `summary.csv` | `bench/summarize.py` | mediane/min/max di **tutti** i `bench_*.jsonl` |

`<TAG>` è il nome dell'esperimento usato anche in `docs/EXPERIMENTS.md`
(es. `F1r_ub3072`).

## Schema — `bench_<TAG>.jsonl`

| Campo | Unità | Significato |
|---|---|---|
| `tag` | — | nome dell'esperimento |
| `depth` | token | token di prompt inviati (8 192 … 196 608) |
| `rep` | — | indice della ripetizione a quella profondità |
| `gen` | token | token massimi richiesti (`max_tokens`) |
| `wall_s` | s | durata totale della richiesta |
| `prompt_n` | token | token effettivamente elaborati (0 se in cache) |
| `cache_n` | token | token riusati dalla cache del server |
| `prompt_ms` | ms | durata del prompt processing |
| `pp_tps` | token/s | `prompt_n / prompt_ms` |
| `predicted_n` | token | token generati |
| `tg_tps` | token/s | velocità di generazione |
| `draft_n` / `draft_acc` / `acc_rate` | token, token, 0–1 | speculative decoding MTP: proposti, accettati, rapporto |
| `n_ctx` | token | contesto del server (deve essere `200192`) |
| `vram_max` | MiB | picco di VRAM usata, campionata con `nvidia-smi` ogni secondo |
| `util_avg` | % | utilizzazione media della GPU durante la richiesta |
| `pcie_gen_max` | — | generazione PCIe osservata (3 = Gen3) |
| `power_avg` | W | potenza media della GPU |
| `ram_avail_min` | MiB | minimo di `MemAvailable` durante la richiesta |
| `swap_used_max` | MiB | picco di swap usato |
| `ts` | ISO 8601 | ora di inizio della misurazione |

## Schema — `agentic_<TAG>.jsonl`

Una riga per test, più una riga `SUMMARY`.

| Campo | Significato |
|---|---|
| `test` | `tool_call`, `tool_call_stream`, `multi_turn`, `code_<nome>`, `SUMMARY` |
| `ok` | esito del test (per il coding: il codice prodotto passa i test unitari) |
| `finish` | `stop` / `tool_calls` / `length` riportato dal server |
| `wall`, `ttft` | durata totale e tempo al primo token (s) |
| `gen_tokens`, `tg_tps`, `draft_acc` | token generati, velocità, accettazione MTP |
| `has_reasoning`, `tool_calls`, `usage` | payload della risposta per i test di tool calling |
| `coding_pass`, `coding_tokens`, `tasks_per_min`, `coding_wall_s` | solo su `SUMMARY` |

## Schema — `session_<TAG>.jsonl`

| Campo | Significato |
|---|---|
| `turn` | numero del turno (1…29) |
| `ctx_tokens` | token di contesto dopo il turno |
| `cache_n`, `prompt_n` | token riusati dalla cache e token realmente elaborati |
| `pp_tps`, `ttft`, `tg_tps` | prestazioni del turno |
| `gen`, `finish`, `wall` | token generati, motivo di stop, durata |
| `answer` | risposta del modello (troncata) |

`prompt_n` stabile tra i turni è la prova che il contesto non viene rielaborato
da zero a ogni richiesta.

## Schema — `needle_<TAG>.jsonl`

| Campo | Significato |
|---|---|
| `expected` | i 5 valori segreti inseriti nel testo |
| `found` | quanti e quali sono stati restituiti esattamente |
| `score` | aghi trovati su 5 |
| `pp_tps`, `tg_tps`, `prompt_n` | prestazioni della prova |

## Schema — `summary.csv`

`tag, depth, n, pp_tps, tg_tps, tg_min, tg_max, acc, ttft_s, vram_max, ram_avail_min, n_ctx`

Contiene le mediane per `(tag, depth)`; `n` è il numero di ripetizioni. È
rigenerato **per intero** a ogni invocazione di `summarize.py`, anche quando si
chiede la tabella di un solo tag.
