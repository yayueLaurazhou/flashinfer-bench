# enum CUflushGPUDirectRDMAWritesScope

The scopes for cuFlushGPUDirectRDMAWrites

###### Values

**CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100**
Blocks until remote writes are visible to the CUDA device context owning the data.
**CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200**
Blocks until remote writes are visible to all CUDA device contexts.