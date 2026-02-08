# enum cudaLimit

CUDA Limits

##### Values

**cudaLimitStackSize = 0x00**
GPU thread stack size
**cudaLimitPrintfFifoSize = 0x01**
GPU printf FIFO size
**cudaLimitMallocHeapSize = 0x02**
GPU malloc heap size
**cudaLimitDevRuntimeSyncDepth = 0x03**
GPU device runtime synchronize depth
**cudaLimitDevRuntimePendingLaunchCount = 0x04**
GPU device runtime pending launch count
**cudaLimitMaxL2FetchGranularity = 0x05**
A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a
hint
**cudaLimitPersistingL2CacheSize = 0x06**


CUDA Runtime API vRelease Version  |  569


Modules


A size in bytes for L2 persisting lines cache size