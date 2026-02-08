# enum CUflushGPUDirectRDMAWritesOptions

Bitmasks for CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS

###### Values

**CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = 1<<0**

cuFlushGPUDirectRDMAWrites() and its CUDA Runtime API counterpart are supported on the
device.
**CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 1<<1**
The CU_STREAM_WAIT_VALUE_FLUSH flag and the
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device.