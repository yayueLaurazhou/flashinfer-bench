# enum cudaGPUDirectRDMAWritesOrdering

CUDA GPUDirect RDMA flush writes ordering features of the device

##### Values

**cudaGPUDirectRDMAWritesOrderingNone = 0**
The device does not natively support ordering of GPUDirect RDMA writes.
cudaFlushGPUDirectRDMAWrites() can be leveraged if supported.
**cudaGPUDirectRDMAWritesOrderingOwner = 100**
Natively, the device can consistently consume GPUDirect RDMA writes, although other CUDA
devices may not.
**cudaGPUDirectRDMAWritesOrderingAllDevices = 200**
Any CUDA device in the system can consistently consume GPUDirect RDMA writes to this device.