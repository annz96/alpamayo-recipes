## Current Optimization Results

Covers 2B / 8B / 32B model sizes and both training stages (S1: full SFT, S2: frozen-backbone expert training), measured at each configuration's production topology (8-64 nodes): **up to ~50% fewer GPU-hours**.

![Training cost, final optimization stack vs baseline. Stage 1 (full SFT) and Stage 2 (expert) each compare 2B/8B/32B model sizes at the production batch (green) and the doubled alternate batch b8/b2 (purple) in GPU-hours; the hatched portion is the cost the optimizations removed, and each bar is labelled with its percentage reduction and 100k-step end-to-end wall time.](assets/combined_cost_frontier.png)

| Model · stage | Per-GPU batch | Topology | Before (s/step) | After (s/step) | GPU-hours change | 100k-step wall time |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2B S1 | b4 | FSDP8 × 16 GPUs | 2.324 | **1.315** | **−43%** | 2.69d → 1.52d |
| 2B S2 | b4 | dp_shard1 × 128 GPUs | 2.402 | **0.737** | **−69%** | 2.78d → 0.85d |
| 8B S1 | b4 | FSDP8 × 16 GPUs | 4.645 | **3.082** | **−34%** | 5.38d → 3.57d |
| 8B S2 | b4 | FSDP8 × 16 GPUs | 2.523 | **1.029** | **−59%** | 2.92d → 1.19d |
| 32B S1 | b1 | FSDP16 × 32 GPUs | 3.908 | **3.265** | **−58%** | 4.52d → 3.78d |
| 32B S2 | b4 | FSDP8 × 16 GPUs | 3.320 | **2.431** | **−63%** | 3.84d → 2.81d |

> wall (s)/step is true wall-clock time (not the trainer log's iteration time).
>
> 100k-E2E GPU-hours are normalized to global batch 512 over 100k steps.
>
> The purple b8/b2 alternate batches in the chart: they halve the GPU count and double the per-GPU batch, keeping the same global batch of 512, so the 100k-step end-to-end wall time is actually longer (2B S1 b8: 2.6 days, 8B S1 b8: 6.2 days, 32B S1 b2: 6.2 days) — but they finish on fewer GPUs and at a lower per-sample cost (2B −13.2%, 8B −10.5%, 32B S1 −14.6%). It's a "fewer GPUs, longer wall time, lower total cost" alternative.

## Optimization Categories

```
[01 Data Loading] ──CPU──► [02 Transport & Step Overhead] ──H2D(CPU→GPU)──► [03 Forward & Backward] ──GPU──► [04 Parallelism & Memory]
```

- **01 Data Loading**: storage → decode → CPU preprocessing → worker CPU scheduling & memory
- **02 Transport & Step Overhead**: packing / H2D copy / GC freeze / fused collective reporting
- **03 Forward & Backward**: forward-backward / attention kernels / the compiler
- **04 Parallelism & Memory**: parallel layout / FSDP / cross-GPU communication / memory management

Two more categories are infrastructure:

- **05 Measurement** (runs through 01-04): per-rank telemetry, CUDA-event decomposition, MFU/HFU/OFU, regression bands on a fixed reference run — answers "how do you know which stage is slow"
- **06 Methodology** (runs through 01-04): same-window paired runs, bitwise-invariance gates, dose-response ladders, the falsification loop — answers "how do you prove a change actually works and didn't break correctness"

## Note: Terminology

**Stall**

How long the slowest rank (GPU) in a training step has to wait for its data.

- Severe stall: within one step, the spread in when different ranks' data becomes ready exceeds 2 seconds — meaning one rank (whose data isn't ready) makes every other rank (GPU) sit idle for more than 2 seconds, which is pure waste.
- Stall burden: how large a share of all steps hit a severe stall, e.g. "9.1% (89/980 steps)" means 89 out of 980 training steps hit a severe stall.

**S1 / S2**

The two training stages in the Alpamayo training pipeline.

- S1 (Stage 1): full SFT training, training the whole model (including the backbone)
- S2 (Stage 2): expert-head training, the backbone stays frozen and only a small expert module is trained

S1 and S2 differ a lot in compute per step: S2's step is cheap, so the CPU-side fixed overhead (worker decode, memory allocation, etc.) is a larger share of the step and exposes data-side bottlenecks more easily; the optimal worker count, batch size, etc. also differ between the two, so gains in this document are usually reported separately for S1 and S2.

---

This directory will keep adding training-optimization write-ups over time; some content is still being organized — stay tuned.
