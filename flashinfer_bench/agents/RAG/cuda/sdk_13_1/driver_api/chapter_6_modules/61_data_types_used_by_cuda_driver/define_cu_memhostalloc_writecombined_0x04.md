# #define CU_MEMHOSTALLOC_WRITECOMBINED 0x04

If set, host memory is allocated as write-combined - fast to write, faster to DMA, slow to read except
via SSE4 streaming load instruction (MOVNTDQA). Flag for cuMemHostAlloc()


CUDA Driver API TRM-06703-001 _vRelease Version  |  92


Modules