# Training Efficiency — Overview

## Current Optimization Results

Covers 2B / 8B / 32B model sizes and both training stages (S1: full SFT, S2: frozen-backbone expert
training), measured at each configuration's production topology (8-64 nodes):

- **Up to 69% fewer GPU-hours**: 2B S2's 100k-step training run drops from 2.78 days to 0.85 days
- **The configs that were data-bound gain the most** (2B, 8B S2): these were CPU/data-loading
  bound before, and now save **43%-69%** of GPU-hours
- **Configs already GPU compute-bound (8B S1, 32B) still gain 34%-58%**: mostly from
  compute/kernel-side work (compilation, attention kernels), not data loading — the two
  optimization tracks are independent and stack

![Training cost, final optimization stack vs baseline. Stage 1 (full SFT) and Stage 2 (expert) each compare 2B/8B/32B model sizes at the production batch (green) and the doubled alternate batch b8/b2 (purple) in GPU-hours; the hatched portion is the cost the optimizations removed, and each bar is labelled with its percentage reduction and 100k-step end-to-end wall time.](assets/combined_cost_frontier.png)

| Model · stage | Per-GPU batch | Topology | Before (s/step) | After (s/step) | GPU-hours change | 100k-step wall time |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2B S1 | b4 | FSDP8 × 16 GPUs | 2.324 | **1.315** | **−43%** | 2.69d → 1.52d |
| 2B S2 | b4 | dp_shard1 × 128 GPUs | 2.402 | **0.737** | **−69%** | 2.78d → 0.85d |
| 8B S1 | b4 | FSDP8 × 16 GPUs | 4.645 | **3.082** | **−34%** | 5.38d → 3.57d |
| 8B S2 | b4 | FSDP8 × 16 GPUs | 2.523 | **1.029** | **−59%** | 2.92d → 1.19d |
| 32B S1 | b1 | FSDP16 × 32 GPUs | 3.908 | **3.265** | **−58%** | 4.52d → 3.78d |
| 32B S2 | b4 | FSDP8 × 16 GPUs | 3.320 | **2.431** | **−63%** | 3.84d → 2.81d |

> s/step is true wall-clock time (not the trainer log's iteration time); GPU-hours are normalized to
> global batch 512 over 100k steps. Quality is preserved throughout — every change is bitwise-exact
> except one (image resize precision dropped from float32 to uint8), which passed two separate
> 100k-step model-quality gates before shipping.
>
> The table doesn't list the purple b8/b2 alternate batches from the chart: they halve the GPU
> count and double the per-GPU batch, keeping the same global batch of 512, so the 100k-step
> end-to-end wall time is actually longer (2B S1 b8: 2.6 days, 8B S1 b8: 6.2 days, 32B S1 b2:
> 6.2 days) — but they finish on fewer GPUs and at a lower per-sample cost (2B −13.2%, 8B −10.5%,
> 32B S1 −14.6%). It's a "fewer GPUs, longer wall time, lower total cost" alternative, not a faster
> one than the production batch.

## Optimization Categories

The category boundaries follow the latest structure of alpamayo's internal MR !2911 documentation
(`projects/alpax/docs/training_optimizations/`): techniques are grouped by *which stage of a
training step* spends the time or memory, not by the order they were discovered in. That makes it
easy to go straight from a symptom (which stage is slow, an OOM, low GPU utilization) to the
category that covers it.

A training step is a pipeline; 01-04 are the four stages it passes through in order:

```
[01 Data Loading] ──CPU──► [02 Transport & Step Overhead] ──H2D(CPU→GPU)──► [03 Forward & Backward] ──GPU──► [04 Parallelism & Memory]
```

What each stage covers:

- **01 Data Loading**: storage → decode → CPU preprocessing → worker CPU scheduling & memory
- **02 Transport & Step Overhead**: packing / H2D copy / GC freeze / fused collective reporting
- **03 Forward & Backward**: forward-backward / attention kernels / the compiler
- **04 Parallelism & Memory**: parallel layout / FSDP / cross-GPU communication / memory management

Two more categories are not stages of the pipeline — they are infrastructure that cuts across all of 01-04:

- **05 Measurement** (runs through 01-04): per-rank telemetry, CUDA-event decomposition, MFU/HFU/OFU, regression bands on a fixed reference run — answers "how do you know which stage is slow"
- **06 Methodology** (runs through 01-04): same-window paired runs, bitwise-invariance gates, dose-response ladders, the falsification loop — answers "how do you prove a change actually works and didn't break correctness"

Two things that differ from the intuitive split:

- **Worker-process CPU scheduling and core partitioning** (easy to mistake for its own category) is
  folded into "01 Data Loading" — it is still a problem the data-loading stage has to solve (the
  worker processes are part of the data-loading pipeline), not a separate stage.
- **Measurement and methodology are not stages of the pipeline** — they are two layers that cut
  across the whole pipeline: measurement answers "how do you find which stage is slow", methodology
  answers "how do you prove a change actually works and didn't break correctness". Without these
  two layers none of the specific optimizations in 01-04 could have been discovered or safely shipped.

## Note: Alpamayo Model Iteration & Training Iteration Roadmap

TBD — to be filled in later (e.g. SFT → VLM RL → Action Expert RL → ...)

## Note: Terminology

**Stall**

How long the slowest rank (GPU) in a training step has to wait for its data.

- Severe stall: within one step, the spread in when different ranks' data becomes ready exceeds
  2 seconds — meaning one rank (whose data isn't ready) makes every other rank (GPU) sit idle for
  more than 2 seconds, which is pure waste.
- Stall burden: how large a share of all steps hit a severe stall, e.g. "9.1% (89/980 steps)" means
  89 out of 980 training steps hit a severe stall.

**S1 / S2**

The two training stages in the Alpamayo training pipeline.

- S1 (Stage 1): full SFT training, training the whole model (including the backbone)
- S2 (Stage 2): expert-head training, the backbone stays frozen and only a small expert module is trained

S1 and S2 differ a lot in compute per step: S2's step is cheap, so the CPU-side fixed overhead
(worker decode, memory allocation, etc.) is a larger share of the step and exposes data-side
bottlenecks more easily; the optimal worker count, batch size, etc. also differ between the two, so
gains in this document are usually reported separately for S1 and S2.
