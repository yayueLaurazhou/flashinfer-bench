# 10.26.1. Simple Synchronization Pattern

### 10.26.1. Simple Synchronization Pattern[](#simple-synchronization-pattern "Permalink to this headline")

Without the arrive/wait barrier, synchronization is achieved using `__syncthreads()` (to synchronize all threads in a block) or `group.sync()` when using [Cooperative Groups](#cooperative-groups).

```
#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */
        block.sync(); /* wait for all threads to arrive here */
        /* code after wait */
    }
}
```

Threads are blocked at the synchronization point (`block.sync()`) until all threads have reached the synchronization point. In addition, memory updates that happened before the synchronization point are guaranteed to be visible to all threads in the block after the synchronization point, i.e., equivalent to `atomic_thread_fence(memory_order_seq_cst, thread_scope_block)` as well as the `sync`.

This pattern has three stages:

* Code **before** sync performs memory updates that will be read **after** the sync.
* Synchronization point
* Code **after** sync point with visibility of memory updates that happened **before** sync point.