# enum CUfunc_cache

Function cache configurations

###### Values

**CU_FUNC_CACHE_PREFER_NONE = 0x00**
no preference for shared memory or L1 (default)
**CU_FUNC_CACHE_PREFER_SHARED = 0x01**
prefer larger shared memory and smaller L1 cache
**CU_FUNC_CACHE_PREFER_L1 = 0x02**
prefer larger L1 cache and smaller shared memory
**CU_FUNC_CACHE_PREFER_EQUAL = 0x03**
prefer equal sized L1 cache and shared memory