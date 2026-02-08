# __host____device__cudaError_t cudaMalloc (void **devPtr, size_t size)

Allocate memory on the device.

##### Parameters

**devPtr**

  - Pointer to allocated device memory
**size**

  - Requested allocation size in bytes

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Allocates size bytes of linear memory on the device and returns in *devPtr a pointer to the
allocated memory. The allocated memory is suitably aligned for any kind of variable. The memory is
not cleared. cudaMalloc() returns cudaErrorMemoryAllocation in case of failure.

The device version of cudaFree cannot be used with a *devPtr allocated using the host API, and vice
versa.



See also:

cudaMallocPitch, cudaFree, cudaMallocArray, cudaFreeArray, cudaMalloc3D, cudaMalloc3DArray,
cudaMallocHost ( C API), cudaFreeHost, cudaHostAlloc, cuMemAlloc


CUDA Runtime API vRelease Version  |  134


Modules