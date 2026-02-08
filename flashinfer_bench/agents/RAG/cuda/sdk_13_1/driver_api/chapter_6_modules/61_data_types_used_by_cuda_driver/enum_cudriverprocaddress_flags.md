# enum CUdriverProcAddress_flags

Flags to specify search options. For more details see cuGetProcAddress

###### Values

**CU_GET_PROC_ADDRESS_DEFAULT = 0**
Default search mode for driver symbols.
**CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1<<0**
Search for legacy versions of driver symbols.
**CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 1<<1**
Search for per-thread versions of driver symbols.