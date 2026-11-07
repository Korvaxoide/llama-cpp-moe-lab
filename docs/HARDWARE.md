# HARDWARE.md — detected hardware (not assumed)

Detected on 2026-11-05 with `lscpu`, `nvidia-smi -q`, `/sys/class/dmi`, `/sys/bus/pci`, `free`, `zramctl`.

## CPU
| Item | Value |
|---|---|
| Model | AMD Ryzen 7 3700X 8-Core (Zen 2, Matisse), family 23 model 113 |
| Cores/threads | 8 cores / 16 threads, 1 socket, 1 NUMA node |
| Cache | L2 4 MiB (8×512K), L3 32 MiB (2 CCX × 16 MiB) |
| Frequencies | 560 MHz – 4427 MHz, boost active |
| Frequency driver | `amd-pstate-epp`, governor `powersave`, EPP `balance_performance` |
| Relevant ISA | AVX2, FMA, F16C, BMI2, SHA (no AVX-512) |

## Motherboard / PCIe
| Item | Value |
|---|---|
| Motherboard | MSI B450 GAMING PLUS MAX (MS-7B86), BIOS H.P1 (2025-09-18) |
| GPU slot | PCIe **Gen3 x16** (8.0 GT/s). The GPU supports Gen4 ("Device Max: 4") but the host is limited to Gen3 ("Host Max: 3") by the B450 chipset |
| Resizable BAR | **active since 2026-11-07** (BIOS option `Re-size BAR Support`): BAR1 = **16384 MiB** (previously 256 MiB). No effect on bandwidth, which stays Gen3 x16; measured **NEUTRAL/negative** for llama.cpp (PP −2.3…−6.5 %, TG equal or worse: `RESULTS.md` §1.3) |
| H2D bandwidth measured indirectly | **~11.8 GiB/s ≈ 12.7 GB/s** (practical Gen3 x16 limit), calculated from 16 ubatch × 13.62 GiB / 18.4 s in the baseline prompt processing (`results/bench_E00b_baseline.jsonl`, 8K, `prompt_ms` 18 398.6) |

**Hardware limit that cannot be overcome in software**: every byte of MoE experts that resides in RAM and is needed by the GPU passes over a bus of ~12.7 GB/s (Gen3 x16). The baseline was saturated on this bus; see `OPTIMIZATION.md` §4.1.

## Memory
| Item | Value |
|---|---|
| RAM | 32 GB DDR4 (32010 MiB seen by the kernel) |
| ZRAM | /dev/zram0, zstd, 31.3 GiB, used as swap |
| vm.swappiness | 150 (typical CachyOS value with zram) |
| THP | `always`, defrag `defer+madvise` |

## GPU
| Item | Value |
|---|---|
| Model | NVIDIA GeForce RTX 3080 (Ampere, GA102, compute capability 8.6) |
| VRAM | 10240 MiB; with the desktop active CUDA sees 9905–9926 MiB total and ~9000–9180 MiB free at llama.cpp startup (F1: 9179, F1r: 9029, pre-ReBAR baseline: 9035, with ReBAR: 8724) |
| Desktop usage | ~0.9 GB occupied by KWin/Plasma/Xwayland/Steam |
| Max clock | SM 2115 MHz, memory 9501 MHz |
| Power limit | 370 W (default = max) |
| Driver | 615.78.08, CUDA UMD 13.4 |
| Persistence mode | Disabled |
| BAR1 / ReBAR | 16384 MiB (ReBAR active). Driver reserve `memory.reserved` 335 MiB; visible VRAM drops by ~21 MiB (9926 → 9905) |

Note: with ReBAR active the free VRAM at startup is lower by ~300–450 MiB, partly because of
the driver reserve (`memory.reserved` 335 MiB), partly because of desktop variability.
This is why the old F1 configuration (MTP compute buffer 942 MiB) no
longer fits and production uses `--spec-draft-n-ubatch 512` (buffer 235.52 MiB).

## Software
| Item | Value |
|---|---|
| OS | CachyOS, kernel 7.3.1-1-cachyos (PREEMPT_DYNAMIC) |
| Docker | 29.9.3 |
| NVIDIA Container Toolkit | 1.20.1 |
| Runtime image | `ghcr.io/ggml-org/llama.cpp@sha256:0283744a…` = b11977, commit 7ac5727d6, CUDA 12.8.1, ARCHS 500..1200 |
| Models disk | NVMe (`/dev/nvme0n1p2`, btrfs) |

## Model (from the GGUF metadata, `analysis/gguf_metadata.txt`)
| Item | Value |
|---|---|
| File | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`, 17 104 402 720 bytes (Unsloth, imatrix) |
| Architecture | `qwen35moe`: 40 hybrid layers + 1 MTP layer (nextn) |
| Layers | 30 Gated DeltaNet (recurrent, fixed state) + 10 full attention (1 every 4) |
| Attention | 16 Q heads, 2 KV heads, head dim 256 → KV f16 = 20 KiB/token (10 layers) |
| MoE | 256 experts, 8 active + 1 shared expert, expert FFN 512 |
| Experts | 13 942 MiB (IQ3_XXS gate/up, IQ4_XS down; last layers Q3_K/Q4_K/Q6_K) |
| Non-experts | ~2 360 MiB (Q8_0 attn/ssm, Q6_K embd/output) |
| Training context | 262 144 |
| Vocabulary | 248 320 |
| Active expert weights/token | ~10.4 MiB/layer × 40 ≈ 415 MiB/token |
