# Training Efficiency — Overview

## 当前优化成果(Current Optimization Results)

TBD — 待后续补充(包括优化迭代的效果路线图)

## 优化方向(Optimization Directions)

```
[01 Storage → Data Decoding]──CPU──►[02 Dataloader CPU Preprocessing/Memory]──►[03 Host↔GPU Data Transfer]──►[05 Model Compute/Kernel]──GPU──►[06 Distributed Training/Communication]──across GPUs
        CPU                              CPU                        CPU→GPU (H2D)                  GPU                        multi-GPU/multi-node
                                            ▲
                              [04 CPU Scheduling/Cross-Process Interference]: Stages 1 and 2 run in
                              multiple worker processes, which share CPU hardware with the
                              trainer process. This category governs "how cores are divided
                              among processes" — it's a resource-scheduling concern that cuts
                              across layers 1 and 2, not a standalone stage in the pipeline

──────────────────────────────────────────────────────────────────────────────────
[07 Diagnostic Methodology/Reproducibility Infrastructure]: Runs through all of 01-06 —
without per-rank telemetry (locating problems), the bitwise invariance gate
(ensuring changes are safe), and the falsification loop (investigating mechanisms),
none of the specific optimizations in 01-06 could have been discovered or validated
```

## Note: Alpamayo 模型迭代与训练迭代 Roadmap

TBD — 待后续补充(例如 SFT → VLM RL → Action Expert RL → ...)
