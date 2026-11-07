# logs/ — evidenza grezza

Log prodotti dai server e dagli script, non post-elaborati. Servono a verificare
le affermazioni dei documenti (versione del runtime, `n_ctx`, buffer allocati,
errori, OOM) e a ricostruire l'ordine degli esperimenti.

## Cosa c'è dentro

| Percorso | Contenuto |
|---|---|
| `exp/<TAG>.log` | stdout/stderr del server sperimentale di quell'esperimento |
| `exp/<TAG>.mem_idle.txt` | RAM disponibile a server fermo, subito dopo il test (MiB) |
| `exp/<TAG>.vram_idle.txt` | VRAM usata a server fermo, subito dopo il test (MiB) |
| `exp/index.txt` | indice: per ogni esperimento, la riga di comando e i buffer allocati |
| `<TAG>.out` | stdout dello script che ha pilotato l'esperimento (`exp.sh`, `queue.sh`) |
| `*_server.log` | log del server per i test agentici/sessione |
| `help_b12124.txt` | output di `--help` dell'immagine b12124, usato per verificare le flag |

## Come leggerli

Le righe che contano per una configurazione sono poche:

```bash
grep -E "n_ctx|n_batch|n_ubatch"            logs/exp/<TAG>.log
grep -E "buffer size =|projected to use"    logs/exp/<TAG>.log
grep -E "error|abort|out of memory"         logs/exp/<TAG>.log
```

La presenza di `n_ctx = 200192` è il controllo che il requisito di contesto è
rispettato. `CUDA0 compute buffer size` e `KV buffer size` spiegano perché una
configurazione entra o non entra in 10 GB.

## Note

- I file sono **grezzi**: contengono timestamp, warning del driver e messaggi del
  server così come sono usciti. Non vanno interpretati come output pulito.
- I log più voluminosi sono quelli con verbosità alta (`-lv 4`) usati per i test
  agentici; sono mantenuti perché contengono la prova dei tool call.
- I path personali sono stati sostituiti con `$HOME` prima della pubblicazione;
  nessun altro contenuto è stato modificato.
- `logs/` **non** è rigenerabile: è la testimonianza delle misure. I dati
  numerici aggregati stanno invece in `results/`.
