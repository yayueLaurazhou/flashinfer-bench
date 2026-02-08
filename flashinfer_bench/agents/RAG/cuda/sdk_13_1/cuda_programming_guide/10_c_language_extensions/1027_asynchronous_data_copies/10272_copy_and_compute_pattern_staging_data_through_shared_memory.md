# 10.27.2. Copy and Compute Pattern - Staging Data Through Shared Memory

### 10.27.2. Copy and Compute Pattern - Staging Data Through Shared Memory[](#copy-and-compute-pattern-staging-data-through-shared-memory "Permalink to this headline")

CUDA applications often employ a *copy and compute* pattern that:

* fetches data from global memory,
* stores data to shared memory, and
* performs computations on shared memory data, and potentially writes results back to global memory.

The following sections illustrate how this pattern can be expressed without and with the `memcpy_async` feature:

* [Without memcpy\_async](#without-memcpy-async) introduces an example that does not overlap computation with data movement and uses an intermediate register to copy data.
* [With memcpy\_async](#with-memcpy-async) improves the previous example by introducing the [memcpy\_async](#collectives-cg-memcpy-async) and the `cuda::memcpy_async` APIs to directly copy data from global to shared memory without using intermediate registers.
* [Asynchronous Data Copies using cuda::barrier](#memcpy-async-barrier) shows memcpy with cooperative groups and barrier.
* [Single-Stage Asynchronous Data Copies using cuda::pipeline](#with-memcpy-async-pipeline-pattern-single) shows memcpy with single stage pipeline.
* [Multi-Stage Asynchronous Data Copies using cuda::pipeline](#with-memcpy-async-pipeline-pattern-multi) shows memcpy with multi stage pipeline.