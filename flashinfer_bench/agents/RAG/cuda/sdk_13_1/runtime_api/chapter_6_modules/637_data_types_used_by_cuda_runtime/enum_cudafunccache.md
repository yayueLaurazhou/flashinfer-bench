# enum cudaFuncCache

CUDA function cache configurations

##### Values

**cudaFuncCachePreferNone = 0**
Default function cache configuration, no preference
**cudaFuncCachePreferShared = 1**


CUDA Runtime API vRelease Version  |  556


Modules


Prefer larger shared memory and smaller L1 cache
**cudaFuncCachePreferL1 = 2**
Prefer larger L1 cache and smaller shared memory
**cudaFuncCachePreferEqual = 3**
Prefer equal size L1 cache and shared memory