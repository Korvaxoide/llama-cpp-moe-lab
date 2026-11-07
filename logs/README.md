# logs/ — raw evidence

Logs produced by the servers and by the scripts, not post-processed. They serve to verify
the claims of the documents (runtime version, `n_ctx`, allocated buffers,
errors, OOM) and to reconstruct the order of the experiments.

## What is inside

| Path | Content |
|---|---|
| `exp/<TAG>.log` | stdout/stderr of the experimental server of that experiment |
| `exp/<TAG>.mem_idle.txt` | available RAM with the server stopped, right after the test (MiB) |
| `exp/<TAG>.vram_idle.txt` | VRAM used with the server stopped, right after the test (MiB) |
| `exp/index.txt` | index: for each experiment, the command line and the allocated buffers |
| `<TAG>.out` | stdout of the script that drove the experiment (`exp.sh`, `queue.sh`) |
| `*_server.log` | server log for the agentic/session tests |
| `help_b12124.txt` | output of `--help` of image b12124, used to verify the flags |

## How to read them

The lines that matter for a configuration are few:

```bash
grep -E "n_ctx|n_batch|n_ubatch"            logs/exp/<TAG>.log
grep -E "buffer size =|projected to use"    logs/exp/<TAG>.log
grep -E "error|abort|out of memory"         logs/exp/<TAG>.log
```

The presence of `n_ctx = 200192` is the check that the context requirement is
respected. `CUDA0 compute buffer size` and `KV buffer size` explain why a
configuration fits or does not fit in 10 GB.

## Notes

- The files are **raw**: they contain timestamps, driver warnings and server
  messages as they came out. They should not be read as clean output.
- The bulkiest logs are the high-verbosity ones (`-lv 4`) used for the agentic
  tests; they are kept because they contain the evidence of the tool calls.
- Personal paths were replaced with `$HOME` before publication;
  no other content was modified.
- `logs/` is **not** regenerable: it is the testimony of the measurements. The aggregated
  numerical data instead live in `results/`.
