# Warm exposed-segment helper benchmark

Isolated exact exposed-segment helper using all real Baxi vertices and emitted final exposed root polylines; not pipeline elapsed.

Queries: 101,102 vertices; exposed roots: 82; subdivided segments: 7,832. Both backends use the same immutable CPU index. Query-result caching is disabled. All four returned evidence arrays match exactly on the first call and every warm repetition.

CUDA context initialization: 0.226s. CPU first helper call: 0.627s. CUDA first helper call: 0.538s. The compiled-kernel disk cache may already be warm from validation/full runs; the first call is a fresh process/context measurement, not cold JIT compilation.

CPU warm seconds: [0.6216366000007838, 0.6108054000069387, 0.5908354000421241]. CUDA warm seconds: [0.5568297000136226, 0.5577998000080697, 0.5586723000160418]. Median ratio CPU/CUDA: 1.095×. Warm times include actual host/device transfers, CPU index queries, CUDA projection, exact CPU contender refinement and deterministic sorting.

Phase and actual kernel/pair/allocation counters are retained in the accompanying JSON. These isolated measurements do not replace the six complete pipeline comparisons.
