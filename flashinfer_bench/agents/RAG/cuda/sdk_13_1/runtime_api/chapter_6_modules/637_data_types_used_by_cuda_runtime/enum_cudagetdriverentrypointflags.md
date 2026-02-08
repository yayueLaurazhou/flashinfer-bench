# enum cudaGetDriverEntryPointFlags

Flags to specify search options to be used with cudaGetDriverEntryPoint For more details see
cuGetProcAddress

##### Values

**cudaEnableDefault = 0x0**
Default search mode for driver symbols.
**cudaEnableLegacyStream = 0x1**
Search for legacy versions of driver symbols.
**cudaEnablePerThreadDefaultStream = 0x2**
Search for per-thread versions of driver symbols.