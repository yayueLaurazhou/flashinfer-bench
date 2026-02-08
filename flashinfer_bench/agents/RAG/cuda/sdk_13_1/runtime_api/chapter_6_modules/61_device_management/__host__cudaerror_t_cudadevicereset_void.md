# __host__cudaError_t cudaDeviceReset (void)

Destroy all allocations and reset all state on the current device in the current process.

##### Returns

cudaSuccess

##### Description

Explicitly destroys and cleans up all resources associated with the current device in the current
process. It is the caller's responsibility to ensure that the resources are not accessed or passed
in subsequent API calls and doing so will result in undefined behavior. These resources
include CUDA types cudaStream_t, cudaEvent_t, cudaArray_t, cudaMipmappedArray_t,
cudaPitchedPtr, cudaTextureObject_t, cudaSurfaceObject_t, textureReference, surfaceReference,
cudaExternalMemory_t, cudaExternalSemaphore_t and cudaGraphicsResource_t. These resources
also include memory allocations by cudaMalloc, cudaMallocHost, cudaMallocManaged and
cudaMallocPitch. Any subsequent API call to this device will reinitialize the device.

Note that this function will reset the device immediately. It is the caller's responsibility to ensure that
the device is not being accessed by any other host threads from the process when this function is called.







CUDA Runtime API vRelease Version  |  25


Modules


See also:

cudaDeviceSynchronize