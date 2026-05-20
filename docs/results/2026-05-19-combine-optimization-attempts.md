# Combine optimization attempts - 2026-05-19

## Scope

Target: EP=16, BF16 dispatch/combine, hidden=7168, topk=8, experts=256, 4 GB300 trays:

```
pod4-gb300-2-tray03-f3
pod4-gb300-2-tray04-f3
pod4-gb300-2-tray05-f3
pod4-gb300-2-tray06-f3
```

Main metric: combine `api Scale-Up GB/s`.

## Baseline

Reference V2 combine before the combine experiments:

| tokens | combine api GB/s |
|---:|---:|
| 16 | 8.9 |
| 256 | 208.6 |
| 512 | 305.8 |
| 4096 | 561.4 |
| 8192 | 584.4 |

## Attempt 1: stage topk weights into TMA buffer

Commit: `bb5bb90 combine: stage topk weights into tma buffer`

Change:

- Move normal combine `topk_weights` into the existing per-warp token TMA buffer.
- Send hidden + weights as one token slot.
- Remove the separate per-token 32B peer store.
- No API change, no extra shared memory.

Result:

| tokens | baseline | attempt | change |
|---:|---:|---:|---:|
| 16 | 8.9 | 8.8 | -1.1% |
| 256 | 208.6 | 207.8 | -0.4% |
| 512 | 305.8 | 310.5 | +1.5% |
| 4096 | 561.4 | 561.5 | +0.0% |
| 8192 | 584.4 | 587.6 | +0.5% |

Conclusion:

No meaningful gain. The 32B weight store is not the bottleneck. Hardware likely coalesces this small contiguous lane store well enough, and large-token performance is dominated by hidden transfer and reduce.

Status: reverted.

## Attempt 2: combine TMA double buffer with 8 warps

Commit: `e23d571 combine: gate double-buffer TMA experiment`

Change:

- Add `DEEPEP_COMBINE_DOUBLE_BUFFER=1`.
- Cap direct combine warps from 16 to 8 so shared memory can fit two token buffers per warp.
- Alternate two TMA buffers and use `tma_store_wait<1>()` so the next hidden load can overlap the previous peer store.

Reason for 8 warps:

```
single token buffer ~= 14.4 KB / warp
16 warps * 1 buffer ~= 230 KB, already at GB300 shared-memory limit
16 warps * 2 buffers ~= 460 KB, impossible
8 warps * 2 buffers ~= 230 KB, fits
```

Result:

| tokens | baseline | attempt | change |
|---:|---:|---:|---:|
| 16 | 8.9 | 8.9 | +0.0% |
| 256 | 208.6 | 210.7 | +1.0% |
| 512 | 305.8 | 323.7 | +5.9% |
| 4096 | 561.4 | 556.1 | -0.9% |
| 8192 | 584.4 | 586.8 | +0.4% |

Conclusion:

Only the 512-token case improved. Large tokens do not benefit because reducing warps from 16 to 8 cuts the number of concurrent TMA channels, which cancels the pipeline overlap.

Status: reverted.

## Attempt 3: direct-store combine epilogue output

Commit: `3637c3a combine: gate direct-store epilogue experiment`

Change:

- Add `DEEPEP_COMBINE_EPILOGUE_DIRECT_STORE=1`.
- In `combine_reduce_epilogue`, write reduced vectors directly to `combined_x` with coalesced lane stores.
- Avoid staging epilogue output in shared memory followed by a TMA store.
- Raise epilogue warps to 32 and launch with zero dynamic shared memory when enabled.

Result:

| tokens | baseline | attempt | change |
|---:|---:|---:|---:|
| 16 | 8.9 | 8.7 | -2.2% |
| 256 | 208.6 | 212.5 | +1.9% |
| 512 | 305.8 | 315.4 | +3.1% |
| 4096 | 561.4 | 560.1 | -0.2% |
| 8192 | 584.4 | 587.9 | +0.6% |

Conclusion:

Direct store does not materially improve combine. The epilogue is not dominated by the final TMA store; reduction reads and BF16/FP32 accumulation remain the main cost.

Status: reverted.

## Attempt 4: push/reduce overlap with ready flags

Commits:

- `b12d932 combine: gate push-reduce overlap experiment`
- `8a26d64 combine: reserve SMs for overlap experiment`

Change:

- Add `DEEPEP_COMBINE_OVERLAP=1`.
- Add a per-slot ready flag region after the direct combine recv/send buffers.
- `combine_impl` clears ready flags, triggers the PDL epilogue early, and publishes `ready(slot, token)=1` with release semantics after each NVLink slot push.
- `combine_reduce_epilogue` keeps `cudaGridDependencySynchronize()`, then waits per required slot with acquire loads before reduce.
- Later reserve producer SMs (`64 -> 48`) to leave SMs for the PDL epilogue, because an all-SM cooperative producer leaves no room for true overlap.

Result:

| tokens | baseline | attempt | change |
|---:|---:|---:|---:|
| 16 | 8.9 | 8.4 | -5.6% |
| 256 | 208.6 | 202.5 | -2.9% |
| 512 | 305.8 | 310.9 | +1.7% |
| 4096 | 561.4 | 560.8 | -0.1% |
| 8192 | 584.4 | 584.4 | +0.0% |

With reserved producer SMs, single-point tests showed occasional mid-token gains but no stable large-token benefit:

| config | tokens | attempt | change |
|---|---:|---:|---:|
| reserved SMs, single run | 512 | 319.9 | +4.6% |
| reserved SMs | 4096 | 562.6 | +0.2% |

SM sweep for EP16/tokens=512 with overlap enabled:

| num_sms | attempt | change |
|---:|---:|---:|
| 40 | 311.1 | +1.7% |
| 48 | 302.1 | -1.2% |
| 56 | 304.8 | -0.3% |
| 64 | 303.3 | -0.8% |

Conclusion:

Ready-flag overlap does not produce stable improvement. Without reserving SMs, the PDL epilogue cannot materially overlap because the cooperative producer occupies the device. Reserving SMs allows some overlap, but it slows the producer enough to cancel the reduce overlap on large tokens. Mid-token gains are within run-to-run jitter and do not justify keeping the code.

Status: reverted.

## Attempt 5: direct slot plan in reduce epilogue

Commits:

- `9d5682d combine: gate direct slot plan experiment`
- `de1ac76 combine: fix direct slot epilogue args`

Change:

- Add `DEEPEP_COMBINE_DIRECT_SLOTS=1`.
- Bypass the compact top-k slot list construction in `combine_reduce_epilogue`.
- Replace per-token `gather`/`compute_topk_slots` compaction with direct lane-shuffled slot selection for the direct scale-up, top-k slot layout.

Result:

| tokens | baseline | attempt | change |
|---:|---:|---:|---:|
| 512 | 305.8 | 314.4 | +2.8% |

Conclusion:

Not stable. The combine kernel bandwidth stayed around the same `~564 GB/s` level, indicating that slot-list construction is not the primary bottleneck. The observed API gain is within the same jitter range seen in other single-point tests.

Status: reverted.

## Why combine is harder than dispatch

Dispatch fast-path can remove a pure copy/unpack epilogue by changing destination layout.

Combine is different:

- Dispatch: one source token fans out to destination slots.
- Combine: several expert outputs fan in and must be reduced into one output token.
- The combine epilogue performs real reduction work, not just data movement.
- Avoiding the epilogue launch alone is not enough to produce a large bandwidth gain.

## Current conclusion

None of the tested combine micro-optimizations should be kept as default. The large-token combine path is mainly constrained by:

- hidden payload TMA transfer,
- number of concurrent TMA channels,
- reduce epilogue memory reads and accumulation,
- shared-memory capacity limiting double buffering.

Further exact combine work needs a more structural change than small TMA/store rearrangements, ready-flag overlap, or slot-list rewrites. Do not repeat these experiments unless the kernel structure or hardware constraints change.
