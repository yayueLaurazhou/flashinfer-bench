# unsigned int cudaExternalSemaphoreWaitParams::flags

Only when cudaExternalSemaphoreSignalParams is used to signal a cudaExternalSemaphore_t
of type cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is
cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates that while waiting for the
cudaExternalSemaphore_t, no memory synchronization operations should be performed for any
external memory object imported as cudaExternalMemoryHandleTypeNvSciBuf. For all other types of
cudaExternalSemaphore_t, flags must be zero.