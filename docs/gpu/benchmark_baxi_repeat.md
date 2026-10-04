# Baxi CPU timing repeat

Comparison performed by GPT-6.1 Sol, Extra High.

One later CPU replay after all twelve primary runs and the warm helper, using the identical frozen source/runtime/config; this checks drift and does not estimate variance.

First CPU elapsed: 191.840s; later CPU elapsed: 143.403s; CUDA elapsed: 154.187s. Later/first CPU ratio: 0.748. Observed first CPU/CUDA ratio: 1.244; later CPU/CUDA ratio: 0.930.

Exact scientific parity for the repeated CPU outputs: True (30 files). Exact presentation pixel parity: True (4 PNGs).

Changes in CPU-only tracing/fitting stages between the primary CPU and CUDA runs mean the whole observed elapsed difference cannot be attributed to CUDA. Use isolated warm helper timings for backend attribution.
