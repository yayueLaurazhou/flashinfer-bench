# unsigned int CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1::flags

Only when CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal a
CUexternalSemaphore of type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC,
the valid flag is CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC
which indicates that while signaling the CUexternalSemaphore, no memory synchronization
operations should be performed for any external memory object imported as
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. For all other types of
CUexternalSemaphore, flags must be zero.