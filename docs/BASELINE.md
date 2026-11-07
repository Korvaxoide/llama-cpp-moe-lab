# BASELINE.md — configurazione originale e baseline misurata

Fonti: `baseline/original/qwen36-35b.inspect.json`, `baseline/original/image.inspect.json`,
`logs/exp/E00b_baseline.log`, `results/bench_E00b_baseline.jsonl`,
`results/agentic_E00_agentic.jsonl`, `results/session_E00_session.jsonl`,
`logs/original_full_run.log`, `logs/E00_baseline_server.log`.

## 1. Configurazione originale (container `qwen36-35b`)

| Voce | Valore |
|---|---|
| Nome container | `qwen36-35b` (preservato, mai cancellato) |
| Immagine | `ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276` |
| Versione / commit | b11977 / `7ac5727d6037ec99c31a28c6520f45de154199d9` |
| Mount | `$HOME/AI/llama-models` → `/models` (ro) |
| Porta | `127.0.0.1:8080` → `8080/tcp` (solo loopback) |
| Restart policy | `no` |
| GPU | `--gpus all` (DeviceRequests capabilities `gpu`) |
| Entrypoint | `/app/llama-server` |

Argomenti esatti (`Config.Cmd`):

```
-m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080
--ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024
--flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2
--spec-draft-cpu-moe
```

Valori impliciti letti dal log (`logs/exp/E00b_baseline.log`):

| Parametro | Valore |
|---|---|
| `n_ctx` | **200192** (richiesto 200000) |
| `n_batch` / `n_ubatch` | 2048 / 512 |
| `n_threads` / `n_threads_batch` | 8 / 8 (su 16 thread) |
| `load_mode` | `mmap` |
| KV target / draft MTP | f16 / f16 |
| `n_slots` | 1, `n_ctx_slot = 200192` |
| prompt cache | abilitata, limite 8192 MiB |
| context checkpoints | 32, spaziatura minima 8192 |
| CUDA ARCHS | 500,610,700,750,800,860,890,900,1200; `FA_QUANTS` include `q8_0-q8_0` |

## 2. Ripartizione VRAM dal log (E00b, avvio)

| Buffer | MiB |
|---|---|
| `CUDA0 model buffer size` (non-esperti su GPU) | 1961.70 |
| `CUDA0 KV buffer size` (target, f16, 200K) | 3910.00 |
| `CUDA0 KV buffer size` (contesto draft MTP) | 391.00 |
| `CUDA0 RS buffer size` (stato ricorrente Gated DeltaNet) | 188.44 |
| `CUDA0 MoE cache size` (per 13942 MiB di esperti host) | 1019.49 |
| `CUDA0 compute buffer size` (contesto principale) | 565.50 |
| `CUDA0 compute buffer size` (contesti MTP, 2 voci) | 235.52 + 235.52 |
| `CUDA_Host compute buffer size` | 211.52 / 203.52 / 203.52 |
| Somma buffer GPU dichiarati | ≈ 8507 MiB |
| `vram_max` osservata durante la baseline (include ~0.9 GB desktop) | 9306 MiB (8K) … 9470 MiB (192K) |

Gli esperti (13 942 MiB) restano in RAM host (`CPU_Mapped`, `--cpu-moe`); la MoE cache
ne tiene una copia parziale in VRAM.

## 3. Baseline FULL `E00b_baseline`

Comando (riproducibile):

```bash
setsid nohup bash -c 'scripts/exp.sh E00b_baseline 8192,16384,32768,65536,131072,196608 3 2 -- \
  --ctx-size 200000 --n-predict -1 --n-gpu-layers auto --cpu-moe --moe-cache-mib 1024 \
  --flash-attn on --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-cpu-moe' > logs/E00b.out 2>&1 < /dev/null &
```

Output di `python3 bench/summarize.py E00b_baseline` (fonte `results/bench_E00b_baseline.jsonl`):

| tag | depth | n | PP t/s | TG t/s (min-max) | MTP acc | TTFT s | VRAM max MiB | RAM avail min | n_ctx |
|---|---|---|---|---|---|---|---|---|---|
| E00b_baseline | 8192 | 3 | 445.25 | 31.41 (31.32-32.98) | 0.86 | 18.4 | 9306 | 25690 | 200192 |
| E00b_baseline | 16384 | 3 | 451.23 | 29.34 (27.58-30.46) | 0.91 | 36.31 | 9298 | 24186 | 200192 |
| E00b_baseline | 32768 | 3 | 443.2 | 28.54 (28.25-29.24) | 0.97 | 73.94 | 9294 | 21797 | 200192 |
| E00b_baseline | 65536 | 2 | 424.63 | 28.48 (28.39-28.56) | 0.8 | 154.33 | 9285 | 19394 | 200192 |
| E00b_baseline | 131072 | 2 | 398.85 | 28.15 (27.87-28.43) | 0.96 | 328.63 | 9444 | 18631 | 200192 |
| E00b_baseline | 196608 | 2 | 385.84 | 28.9 (28.35-29.44) | 0.85 | 509.6 | 9470 | 19779 | 200192 |

Verifiche: `n_ctx = 200192` ✔, `prompt_n = depth` e nessun `warning` in tutte le 15 righe ✔,
nessun `CUDA error` / `out of memory` / `GGML_ABORT` ✔.

La baseline parziale `E00_baseline` (8K–64K, 1–2 rep, con `-lv 4`) è coerente entro il
rumore (%: 8K PP 438.73 vs 445.25, TG 29.88 vs 31.41) — fonte `results/bench_E00_baseline.jsonl`.

## 4. Baseline agentica `E00_agentic`

Server con gli stessi argomenti + `-lv 4`; `python3 bench/agentic.py --tag E00_agentic --coding-reps 2`.
Fonte: `results/agentic_E00_agentic.jsonl`.

| Test | Esito |
|---|---|
| tool_call (non-stream) | **ok** (finish `tool_calls`, argomenti JSON validi) |
| tool_call_stream (SSE) | **ok** |
| multi_turn (tool result reimmesso) | **ok** (risposta cita CRC32 / 8 hex) |
| coding (5 problemi × 2 rep) | **10/10** |
| tasks_per_min (codice corretto) | **6.7** |
| TG mediana sui task di coding | ≈ 32.1 t/s |
| accettazione MTP sui task di coding | 0.82–0.97 |

Statistiche MoE cache all'arresto (`logs/exp/E00_agentic_server.log`):
`ubatch <= 8: hits = 73368, misses = 655786, hit rate = 10.06%`.

## 5. Sessione lunga `E00_session`

`python3 bench/session.py --tag E00_session --target 180000 --chunk 6000`, 29 turni,
558.5 s totali. Fonte: `results/session_E00_session.jsonl`.

| Metrica | Valore |
|---|---|
| Turni | 29 |
| Contesto iniziale → finale | 6 612 → 178 647 token |
| `prompt_n` per turno (min/mediana/max) | 6082 / **6085** / 6548 |
| `cache_n` riusato (ultimo turno) | 172 469 |
| TTFT (min/mediana/max) | 14.62 / **16.86** / 18.79 s |
| PP mediana | 364.6 t/s |
| TG mediana | 27.59 t/s (primi 3 turni 27.1–29.4, ultimi 3 25.1–27.5) |
| Accettazione MTP mediana | 0.833 |
| Token generati totali | 2 012 |

**Controllo critico superato**: `prompt_n` resta ≈ 6085 (il chunk da 6000 + poche centinaia
di token di prompt) per tutti i 29 turni; nessun turno ha rielaborato l'intero contesto.
La cache del prompt funziona anche sul modello ibrido.

Statistiche MoE cache all'arresto (`logs/exp/E00_session_server.log`):
`ubatch <= 8: hits = 73123, misses = 479655, hit rate = 13.23%, uploaded = 627994.61 MiB`.

## 6. Sessione reale originale (`logs/original_full_run.log`)

Sessione agentica reale (OpenCode) fino a ~155K token, interrotta dal riavvio del demone
Docker (exit 255), non da un crash del modello. 158 richieste con timing.

| Metrica | Valore misurato |
|---|---|
| TG mediana complessiva | 27.44 t/s |
| TG mediana ultime 20 richieste (contesto ~150K) | **26.85 t/s** |
| Prompt incrementali > 2000 token (30 richieste) | **347.9 t/s** mediani, 2544 token mediani |
| TTFT implicito per ~2.4K token | ≈ 2.4K / 348 ≈ **7.0 s** |

## 7. I due warning originali

1. **`tensor overrides to CPU are used with mmap enabled - consider using --load-mode none`**
   Con `--cpu-moe` gli esperti sono forzati su CPU. Con mmap il loader li mappa in memoria
   pageable (non in un buffer pinned `CUDA_Host`), quindi ogni trasferimento H2D è una DMA
   da pagina non pinnata. L'ipotesi H3 (testata in Fase 4) è che `--load-mode none` li metta
   in `CUDA_Host` pinned e velocizzi PP e i miss della MoE cache. Costo: ~14 GB di RAM pinned.

2. **`common_fit_params: failed to fit params to free device memory: model_params::tensor_buft_overrides already set by user, abort`**
   `--fit` (auto-placement) si arrende perché l'utente ha già fissato i buffer degli esperti
   con `--cpu-moe`: il posizionamento è manuale. È **innocuo**: il contesto resta 200192 e
   `n_ctx` non viene ridotto. Implica solo che `--n-gpu-layers auto` non può ricalibrare.
