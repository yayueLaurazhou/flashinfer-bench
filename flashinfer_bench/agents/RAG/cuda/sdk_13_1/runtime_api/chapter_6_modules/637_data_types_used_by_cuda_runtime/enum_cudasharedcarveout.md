# enum cudaSharedCarveout

Shared memory carveout configurations. These may be passed to cudaFuncSetAttribute

##### Values

**cudaSharedmemCarveoutDefault = -1**
No preference for shared memory or L1 (default)
**cudaSharedmemCarveoutMaxShared = 100**
Prefer maximum available shared memory, minimum L1 cache
**cudaSharedmemCarveoutMaxL1 = 0**
Prefer maximum available L1 cache, minimum shared memory