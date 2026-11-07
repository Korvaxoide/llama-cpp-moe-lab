# HARDWARE.md — hardware rilevato (non presunto)

Rilevato il 2026-11-05 con `lscpu`, `nvidia-smi -q`, `/sys/class/dmi`, `/sys/bus/pci`, `free`, `zramctl`.

## CPU
| Voce | Valore |
|---|---|
| Modello | AMD Ryzen 7 3700X 8-Core (Zen 2, Matisse), family 23 model 113 |
| Core/thread | 8 core / 16 thread, 1 socket, 1 nodo NUMA |
| Cache | L2 4 MiB (8×512K), L3 32 MiB (2 CCX × 16 MiB) |
| Frequenze | 560 MHz – 4427 MHz, boost attivo |
| Driver frequenza | `amd-pstate-epp`, governor `powersave`, EPP `balance_performance` |
| ISA rilevanti | AVX2, FMA, F16C, BMI2, SHA (niente AVX-512) |

## Scheda madre / PCIe
| Voce | Valore |
|---|---|
| Scheda madre | MSI B450 GAMING PLUS MAX (MS-7B86), BIOS H.P1 (2025-09-18) |
| Slot GPU | PCIe **Gen3 x16** (8.0 GT/s). La GPU supporta Gen4 ("Device Max: 4") ma l'host è limitato a Gen3 ("Host Max: 3") dal chipset B450 |
| Resizable BAR | disattivato (BAR1 = 256 MiB) |
| Banda H2D misurata indirettamente | **~11.8 GiB/s ≈ 12.7 GB/s** (limite pratico Gen3 x16), calcolata da 16 ubatch × 13.62 GiB / 18.4 s nel prompt processing baseline (`results/bench_E00b_baseline.jsonl`, 8K, `prompt_ms` 18 398.6) |

**Limite hardware non superabile via software**: ogni byte di esperti MoE che risiede in RAM e serve alla GPU passa su un bus da ~12.7 GB/s (Gen3 x16). La baseline era satura su questo bus; vedi `OPTIMIZATION.md` §4.1.

## Memoria
| Voce | Valore |
|---|---|
| RAM | 32 GB DDR4 (32010 MiB visti dal kernel) |
| ZRAM | /dev/zram0, zstd, 31.3 GiB, usata come swap |
| vm.swappiness | 150 (valore tipico CachyOS con zram) |
| THP | `always`, defrag `defer+madvise` |

## GPU
| Voce | Valore |
|---|---|
| Modello | NVIDIA GeForce RTX 3080 (Ampere, GA102, compute capability 8.6) |
| VRAM | 10240 MiB (9926 MiB visti da CUDA, ~9070 MiB liberi con il desktop attivo) |
| Uso desktop | ~0.9 GB occupati da KWin/Plasma/Xwayland/Steam |
| Clock max | SM 2115 MHz, memoria 9501 MHz |
| Power limit | 370 W (default = max) |
| Driver | 615.78.08, CUDA UMD 13.4 |
| Persistence mode | Disabled |

## Software
| Voce | Valore |
|---|---|
| OS | CachyOS, kernel 7.3.1-1-cachyos (PREEMPT_DYNAMIC) |
| Docker | 29.9.3 |
| NVIDIA Container Toolkit | 1.20.1 |
| Immagine runtime | `ghcr.io/ggml-org/llama.cpp@sha256:0283744a…` = b11977, commit 7ac5727d6, CUDA 12.8.1, ARCHS 500..1200 |
| Disco modelli | NVMe (`/dev/nvme0n1p2`, btrfs) |

## Modello (dai metadati GGUF, `analysis/gguf_metadata.txt`)
| Voce | Valore |
|---|---|
| File | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`, 17 104 402 720 byte (Unsloth, imatrix) |
| Architettura | `qwen35moe`: 40 layer ibridi + 1 layer MTP (nextn) |
| Layer | 30 Gated DeltaNet (ricorrenti, stato fisso) + 10 full attention (1 ogni 4) |
| Attenzione | 16 head Q, 2 head KV, head dim 256 → KV f16 = 20 KiB/token (10 layer) |
| MoE | 256 esperti, 8 attivi + 1 shared expert, FFN esperto 512 |
| Esperti | 13 942 MiB (IQ3_XXS gate/up, IQ4_XS down; ultimi layer Q3_K/Q4_K/Q6_K) |
| Non-esperti | ~2 360 MiB (Q8_0 attn/ssm, Q6_K embd/output) |
| Contesto training | 262 144 |
| Vocabolario | 248 320 |
| Pesi esperto attivi/token | ~10.4 MiB/layer × 40 ≈ 415 MiB/token |
