# #define cudaExternalSemaphoreSignalSkipNvSciBufMemSync 0x01

When the /p flags parameter of cudaExternalSemaphoreSignalParams contains this flag, it
indicates that signaling an external semaphore object should skip performing appropriate
memory synchronization operations over all the external memory objects that are imported as
cudaExternalMemoryHandleTypeNvSciBuf, which otherwise are performed by default to ensure data
coherency with other importers of the same NvSciBuf memory objects.