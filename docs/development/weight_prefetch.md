# Weight Prefetch Guide

This branch adds an opt-in `--prefetch-weights` / `-pw` path for partial host-offload workloads where the GPU still executes the compute but some model weights live in system RAM.

It was built for the failure mode "prompt processing is too slow when the model is larger than VRAM", not for full-VRAM fits.

## What It Does

- stages host-resident weights ahead of GPU use
- prefers backend async copies / events when the backend exposes them
- keeps default behavior unchanged unless `-pw` is passed
- supports both dense and MoE paths
- also enables the same staging path for decode/generation on this branch

## When To Use It

Use it when all of the following are true:

- the model does not fit fully in VRAM
- you still want GPU execution
- some weights are intentionally host-resident
- prompt processing throughput is the main problem

Validated high-value cases on this branch:

- dense partial offload via `--override-tensor 'blk\.[0-9]+\.ffn_.*=CPU'`
- MoE partial offload via `--n-cpu-moe 999`
- `--no-mmap`

Known bad or weak cases:

- full-VRAM fits
- mmap-enabled host-offload comparisons
- very large `ubatch` values that activate too much MoE state or create extra staging churn

## Flags

Core flags:

- `-pw`, `--prefetch-weights`
- `--prefetch-weights-stats`
- `--prefetch-weights-min-batch N`
- `--prefetch-weights-max-mib N`

Typical companion flags:

- `--no-mmap`
- dense partial offload: `-ot 'blk\.[0-9]+\.ffn_.*=CPU'`
- MoE partial offload: `-ncmoe 999`

## Recommended Starting Points

Dense:

```bash
./build-vulkan/bin/llama-server \
  -m /path/to/Qwen3.6-27B-UD-IQ2_XXS.gguf \
  -dev Vulkan0 \
  --no-mmap \
  -ot 'blk\.[0-9]+\.ffn_.*=CPU' \
  -c 4096 \
  -b 1024 \
  -ub 128 \
  -pw
```

MoE:

```bash
./build-vulkan/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  -dev Vulkan0 \
  --no-mmap \
  -ncmoe 999 \
  -c 4096 \
  -b 1024 \
  -ub 128 \
  -pw
```

Decode-enabled variant on this branch:

```bash
./build-vulkan/bin/llama-bench \
  -m /path/to/model.gguf \
  -dev Vulkan0 \
  -p 2048 -n 64 \
  -b 2048 -ub 128 \
  --no-mmap \
  -pw
```

## Test Environment

Benchmarks below were collected on:

- OS: CachyOS / Arch-based Linux
- CPU: AMD Ryzen 5 9600X
- GPU: AMD Radeon RX 5700 XT 8 GiB
- RAM: 30 GiB usable system RAM, DDR5-6400 class
- primary backend: Vulkan
- secondary backend: HIP/ROCm

Models:

- dense: `unsloth/Qwen3.6-27B-GGUF` `Qwen3.6-27B-UD-IQ2_XXS.gguf`
- MoE: `unsloth/Qwen3.6-35B-A3B-GGUF` `Qwen3.6-35B-A3B-UD-IQ1_M.gguf`

## Benchmark Summary

All rows compare stock baseline vs this branch under matched settings.

### Prompt Processing

| model | arch | backend | prompt | ubatch | placement | mmap | baseline pp tok/s | prefetch pp tok/s | delta |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B IQ2_XXS | dense | Vulkan | 512 | 128 | FFN on CPU | 0 | 107.66 | 108.02 | +0.33% |
| Qwen3.6-27B IQ2_XXS | dense | Vulkan | 2048 | 128 | FFN on CPU | 0 | 44.50 | 46.76 | +5.08% |
| Qwen3.6-27B IQ2_XXS | dense | Vulkan | 8192 | 128 | FFN on CPU | 0 | 98.63 | 111.27 | +12.82% |
| Qwen3.6-27B IQ2_XXS | dense | Vulkan | 2048 | 512 | FFN on CPU | 0 | 51.33 | 50.50 | -1.62% |
| Qwen3.6-27B IQ2_XXS | dense | HIP | 2048 | 128 | FFN on CPU | 0 | 71.06 | 81.62 | +14.86% |
| Qwen3.6-35B-A3B IQ1_M | MoE | Vulkan | 512 | 128 | experts on CPU | 0 | 214.96 | 661.44 | +207.71% |
| Qwen3.6-35B-A3B IQ1_M | MoE | Vulkan | 2048 | 128 | experts on CPU | 0 | 206.94 | 649.55 | +213.89% |
| Qwen3.6-35B-A3B IQ1_M | MoE | Vulkan | 8192 | 128 | experts on CPU | 0 | 199.53 | 612.58 | +207.01% |
| Qwen3.6-35B-A3B IQ1_M | MoE | HIP | 2048 | 128 | experts on CPU | 0 | 173.25 | 505.49 | +191.77% |

### Decode / Generation

These rows used `p=2048`, `n=64`, `ub=128`, `--no-mmap`, with the same host-offload placements as above.

| model | backend | baseline pp tok/s | prefetch pp tok/s | pp delta | baseline tg tok/s | prefetch tg tok/s | tg delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B IQ2_XXS | Vulkan | 102.17 | 116.92 | +14.44% | 5.612 | 5.665 | +0.93% |
| Qwen3.6-35B-A3B IQ1_M | Vulkan | 208.17 | 661.80 | +217.91% | 37.48 | 37.67 | +0.49% |
| Qwen3.6-27B IQ2_XXS | HIP | 71.18 | 82.41 | +15.77% | 6.139 | 6.117 | -0.35% |

### Memory-Logged Spot Checks

These runs captured process RSS plus sampled device/host memory telemetry.

| case | pp tok/s | tg tok/s | peak RSS MiB | peak total VRAM MiB | peak total GTT MiB | peak llama VRAM MiB | peak llama GTT MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense Vulkan baseline | 101.49 | 5.71 | 267.51 | 5518 | 4997 | 4536 | 4885 |
| dense Vulkan prefetch | 116.17 | 5.63 | 267.30 | 5564 | 4993 | 4582 | 4885 |
| MoE Vulkan baseline | 208.87 | 39.27 | 205.99 | 2590 | 8351 | 1609 | 8247 |
| MoE Vulkan prefetch | 658.30 | 39.18 | 205.09 | 2588 | 8351 | 1607 | 8247 |
| dense HIP baseline | 71.79 | 6.14 | 5264.76 | 5842 | 111 | 4808 | 8 |
| dense HIP prefetch | 82.35 | 6.11 | 5281.82 | 5842 | 111 | 4808 | 8 |

## Interpreting The Results

- Dense: real but conditional win. The best gains showed up at larger prompt sizes with partial host offload and modest `ubatch`.
- MoE: strong win when experts were host-resident and the GPU still ran the compute.
- Decode: much smaller effect than prompt processing. This branch enables decode staging, but the main value remains prompt processing.
- `--no-mmap` mattered. mmap-backed runs were weaker or misleading for this path.
- This is not a default-on optimization. Full-offload / full-VRAM cases can regress.

## Practical Tuning Rules

- Start with `-ub 128`.
- Increase `-b` before increasing `-ub`.
- If the machine gets unstable, lower `-ub` first, then `-b`, then `-c`.
- For MoE, do not assume larger `ubatch` is always better; too much expert activation can erase the gain.
- For dense, the benefit usually grows with prompt size until copy pressure or memory pressure dominates.

## Benchmark Commands Used

Prompt-processing examples:

```bash
./build-vulkan/bin/llama-bench \
  -m /models/Qwen3.6-27B-UD-IQ2_XXS.gguf \
  -dev Vulkan0 \
  -p 2048 -n 0 -b 2048 -ub 128 \
  --no-mmap \
  -ot 'blk\.[0-9]+\.ffn_.*=CPU' \
  -pw -o json

./build-vulkan/bin/llama-bench \
  -m /models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  -dev Vulkan0 \
  -p 2048 -n 0 -b 2048 -ub 128 \
  --no-mmap \
  -ncmoe 999 \
  -pw -o json
```

Decode example:

```bash
./build-vulkan/bin/llama-bench \
  -m /models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  -dev Vulkan0 \
  -p 2048 -n 64 -b 2048 -ub 128 \
  --no-mmap \
  -ncmoe 999 \
  -pw -o json
```

## Verdict

- dense prefill: win, but conditional
- MoE prefill: strong win
- decode: small win or neutral
- recommended default: keep `--prefetch-weights` opt-in
