# __host____device__cudaError_t cudaFree (void *devPtr)

Frees memory on the device.

##### Parameters

**devPtr**

  - Device pointer to memory to free

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  121


Modules

##### Description

Frees the memory space pointed to by devPtr, which must have been returned by a previous
call to one of the following memory allocation APIs - cudaMalloc(), cudaMallocPitch(),
cudaMallocManaged(), cudaMallocAsync(), cudaMallocFromPoolAsync().

Note - This API will not perform any implicit synchronization when the pointer was allocated with
cudaMallocAsync or cudaMallocFromPoolAsync. Callers must ensure that all accesses to these pointer
have completed before invoking cudaFree. For best performance and memory reuse, users should
use cudaFreeAsync to free memory allocated via the stream ordered memory allocator. For all other
pointers, this API may perform implicit synchronization.

If cudaFree(devPtr) has already been called before, an error is returned. If devPtr is 0, no operation
is performed. cudaFree() returns cudaErrorValue in case of failure.

The device version of cudaFree cannot be used with a *devPtr allocated using the host API, and vice
versa.



See also:

cudaMalloc, cudaMallocPitch, cudaMallocManaged, cudaMallocArray, cudaFreeArray,
cudaMallocAsync, cudaMallocFromPoolAsync cudaMallocHost ( C API), cudaFreeHost,
cudaMalloc3D, cudaMalloc3DArray, cudaFreeAsync cudaHostAlloc, cuMemFree