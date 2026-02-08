# 15.11.4. IPC Export Pool Limitations

### 15.11.4. IPC Export Pool Limitations[](#ipc-export-pool-limitations "Permalink to this headline")

IPC pools currently do not support releasing physical blocks back to the OS. As a result the `cudaMemPoolTrimTo` API acts as a no-op and the `cudaMemPoolAttrReleaseThreshold` effectively gets ignored. This behavior is controlled by the driver, not the runtime and may change in a future driver update.