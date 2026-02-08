# #define cudaExternalSemaphoreWaitSkipNvSciBufMemSync 0x02

When the /p flags parameter of cudaExternalSemaphoreWaitParams contains this flag, it
indicates that waiting an external semaphore object should skip performing appropriate
memory synchronization operations over all the external memory objects that are imported as
cudaExternalMemoryHandleTypeNvSciBuf, which otherwise are performed by default to ensure data
coherency with other importers of the same NvSciBuf memory objects.