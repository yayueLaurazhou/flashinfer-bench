# #define CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_ME 0x02

When the flags parameter of CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS contains
this flag, it indicates that waiting on an external semaphore object should skip performing appropriate
memory synchronization operations over all the external memory objects that are imported as
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF, which otherwise are performed by
default to ensure data coherency with other importers of the same NvSciBuf memory objects.