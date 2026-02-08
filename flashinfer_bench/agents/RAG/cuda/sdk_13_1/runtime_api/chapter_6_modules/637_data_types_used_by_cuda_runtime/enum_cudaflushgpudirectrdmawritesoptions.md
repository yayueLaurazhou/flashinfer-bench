# enum cudaFlushGPUDirectRDMAWritesOptions

CUDA GPUDirect RDMA flush writes APIs supported on the device

##### Values

**cudaFlushGPUDirectRDMAWritesOptionHost = 1<<0**

cudaDeviceFlushGPUDirectRDMAWrites() and its CUDA Driver API counterpart are supported on
the device.
**cudaFlushGPUDirectRDMAWritesOptionMemOps = 1<<1**
The CU_STREAM_WAIT_VALUE_FLUSH flag and the
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the CUDA
device.