# Live batch memory allocation, 2026-10-09

The reported batch ran one sample while Windows retained at least 9 GiB of
available RAM. Its completed BaxiNo2 log used the old memory policy and peaked
at 1.93 GiB private memory against a 4 GiB estimate. The previous allocation also
calculated a fixed worker count at batch startup by charging every slot the
largest selected input estimate. A 10.24 GiB SN14 estimate could therefore
limit the whole batch to one worker, even when smaller samples would fit or
more RAM became available later.

## Allocation changes

The batch GUI now enables `live_memory_admission` in `allocate_resources`.
CPU and sample counts determine an upper slot count; startup RAM is no longer
a permanent concurrency limit. For six inputs on 16 logical CPUs with two
threads per sample, the ceiling is six supervising threads. These threads
launch sample processes only when `MemoryAdmission` accepts each individual
budget. Waiting does not load sample geometry into a child process.

The existing atomic gate checks available physical RAM and Windows commit
headroom on every attempt, subtracts active samples' unused budgets, and keeps
2 GiB for the desktop. Waiting workers retry every 0.25 seconds, so a batch
can grow from one active sample to two or more without being restarted when
other applications release memory. Paused samples retain their reservations;
unknown capacity admits only one sample. Manual concurrency remains an upper
limit and automatic sample threads remain capped at two.

This is a concurrency ceiling, not a promise that six sample processes will
fit at once. Only the bounded number of supervising slots can compete for
admission. The queue display retains its order; a smaller waiting sample can
start while a larger one is still waiting for sufficient capacity.

The default allocation API remains RAM-limited for callers without a live
gate. `BatchScheduler.automatic(..., live_memory_admission=True)` attaches
the required gate automatically; direct allocation callers using this option
must also gate their launches.

## Another 50% reduction of initial budgets and 5% live headroom

Policy `geometry-private-peak-v4` halves every initial sample estimate from
`geometry-private-peak-v3`. This is 30% of the original v2 calibration:

`ceil(0.30 * max(3 GiB, 2 GiB + 8192 * full_vertices + 192 * faces + 48 * original_vertices))`.

Rounding is upward to a whole byte using integer arithmetic. The reduction
includes the minimum (1.8 to 0.9 GiB), all geometry terms, and the unknown-input
fallback (2.4 to 1.2 GiB). The pre-spawn input estimate and decoded geometry
estimate use the same policy. Admission never changes analysis sampling,
geometry, candidates, segmentation, hierarchy or exports.

The monitor raises active budgets to the maximum of their existing budget,
decoded estimate and 105% of the observed private peak, rounded up to bytes.
This reduces observed-peak growth headroom from 25% to the requested 5%.
A later smaller working set does not erase that high-water budget.
Resource logs identify the policy
and distinguish `geometry_estimated_peak_bytes` from `estimated_peak_bytes`.
Actual observed usage is never discounted. The shared desktop
reserve remains 2 GiB and both physical-RAM and Windows commit checks remain
active.

| Sample | Maximum recorded private peak (GiB) | Previous v3 budget (GiB) | Reduced initial v4 budget (GiB) | Peak + 5% (GiB) |
|---|---:|---:|---:|---:|
| BaxiNo2 | 1.93 | 1.80 | 0.90 | 2.03 |
| Kaixinlv | 2.56 | 2.86 | 1.43 | 2.69 |
| SN14 | 4.88 | 3.74 | 1.87 | 5.12 |
| w5168 | 2.08 | 2.02 | 1.01 | 2.18 |
| W82 water | 2.17 | 1.99 | 1.00 | 2.28 |
| W82 MS4 | 2.33 | 2.05 | 1.03 | 2.45 |

The last column shows the budget derived from the historical maximum peak as
an illustration. Live reservations use the peak observed in the active run;
historical logs are not loaded to set a runtime reservation.

`recorded_peaks.json` freezes 25 completed, exit-code-zero logs across these
six inputs, using two threads per sample: 12 runs in the October 4 folder,
six in the October 5 folder, and seven in the October 7 folder, including
the reported manual-endpoint batch's BaxiNo2 result. Recent runs include noise
reduction. Each record includes the relative source log path and SHA-256,
decoded counts, peak private and resident bytes, and its original budget.
The reduced initial v4 budgets are below the maximum recorded private peak
for all six samples. For example, SN14 starts with 1.87 GiB against a recorded
4.88 GiB peak; the live monitor will raise its reservation as private usage
grows. More samples may
start together, but simultaneous memory spikes can exhaust memory before
monitoring reacts, especially with the narrower 5% headroom. Existing workers
are not evicted when budgets rise.
These logs are observed evidence, not a new scientific replay or an automatic
runtime training database.

## Validation and limits

Focused validation: **56 passed in 21.06 s** across memory admission,
scheduler, process lifecycle, batch GUI and desktop GUI tests:

```powershell
New-Item -ItemType Directory -Path 'memory_allocation_validation_20261009' -Force | Out-Null
.\.venv\Scripts\python.exe -m pytest -q tests/test_memory_admission.py tests/test_batch.py tests/test_batch_process.py tests/test_batch_guidance_gui.py tests/test_desktop_gui.py --basetemp memory_allocation_validation_20261009\reduced50 -p no:cacheprovider --tb=short
```

Regression coverage includes:

- All 25 recorded geometries receive half their frozen v3 budget (rounded
  up to a byte); live peak-derived reservations retain 5% headroom.
- One active worker at 3 GiB available grows to two at 4 GiB and three at
  5 GiB in a deterministic scheduler simulation, with a fourth still queued.
- With 9 GiB available and a simulated Kaixinlv worker using 1.5 GiB resident
  memory and its recorded peak plus 5%, five w5168-sized neighbors can start.
  A sixth neighbor cannot exceed the remaining budget. This only simulates
  admission; it does not predict their combined eventual peak usage.
- Unexpected private growth raises the live reservation by 5% and the
  increased budget reaches admission and survives lower later usage.
- Small and non-divisible observed byte counts round upward; zero or invalid
  negative peaks do not produce a negative reservation.
- Existing commit-headroom, unknown-telemetry, cancellation, pause, failed
  process cleanup and individually oversized-input checks remain covered.

These are estimates and admission checks, not enforced per-process limits.
Different inputs, optional analysis modes, higher thread counts and abrupt
allocation spikes may need more memory. No full scientific batch or measured
throughput benchmark was rerun for this scheduling-only change.

Existing GUI instances retain their loaded scheduler. After the current batch
finishes, reopen the GUI and use **Auto** concurrency to apply the new policy.
