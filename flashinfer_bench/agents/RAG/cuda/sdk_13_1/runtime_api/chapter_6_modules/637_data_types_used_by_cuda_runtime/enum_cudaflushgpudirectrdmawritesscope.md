# enum cudaFlushGPUDirectRDMAWritesScope

CUDA GPUDirect RDMA flush writes scopes

##### Values

**cudaFlushGPUDirectRDMAWritesToOwner = 100**
Blocks until remote writes are visible to the CUDA device context owning the data.


CUDA Runtime API vRelease Version  |  555


Modules


**cudaFlushGPUDirectRDMAWritesToAllDevices = 200**
Blocks until remote writes are visible to all CUDA device contexts.