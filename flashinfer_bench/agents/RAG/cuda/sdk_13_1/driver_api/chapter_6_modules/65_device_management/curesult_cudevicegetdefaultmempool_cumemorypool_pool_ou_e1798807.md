# CUresult cuDeviceGetDefaultMemPool (CUmemoryPool *pool_out, CUdevice dev)

Returns the default mempool of a device.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_NOT_SUPPORTED

###### Description

The default mempool of a device contains device memory from that device.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemAllocAsync, cuMemPoolTrimTo, cuMemPoolGetAttribute, cuMemPoolSetAttribute,
cuMemPoolSetAccess, cuDeviceGetMemPool, cuMemPoolCreate