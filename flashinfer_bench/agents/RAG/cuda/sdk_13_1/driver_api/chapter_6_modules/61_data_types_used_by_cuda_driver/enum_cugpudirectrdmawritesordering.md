# enum CUGPUDirectRDMAWritesOrdering

Platform native ordering for GPUDirect RDMA writes

###### Values

**CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0**
The device does not natively support ordering of remote writes. cuFlushGPUDirectRDMAWrites()
can be leveraged if supported.
**CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100**
Natively, the device can consistently consume remote writes, although other CUDA devices may
not.
**CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200**
Any CUDA device in the system can consistently consume remote writes to this device.