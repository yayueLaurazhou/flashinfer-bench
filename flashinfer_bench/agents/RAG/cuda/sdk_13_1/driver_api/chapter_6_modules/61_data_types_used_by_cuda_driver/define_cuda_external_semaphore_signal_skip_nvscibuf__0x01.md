# #define CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_ 0x01

When the flags parameter of CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS contains
this flag, it indicates that signaling an external semaphore object should skip performing appropriate
memory synchronization operations over all the external memory objects that are imported as
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF, which otherwise are performed by
default to ensure data coherency with other importers of the same NvSciBuf memory objects.