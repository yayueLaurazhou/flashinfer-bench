# __host__cudaError_t cudaMallocPitch (void **devPtr, size_t *pitch, size_t width, size_t height)

Allocates pitched memory on the device.

##### Parameters

**devPtr**

  - Pointer to allocated pitched device memory
**pitch**

  - Pitch for allocation
**width**

  - Requested pitched allocation width (in bytes)
**height**

  - Requested pitched allocation height

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation


CUDA Runtime API vRelease Version  |  146


Modules

##### Description

Allocates at least width (in bytes) * height bytes of linear memory on the device and returns
in *devPtr a pointer to the allocated memory. The function may pad the allocation to ensure
that corresponding pointers in any given row will continue to meet the alignment requirements
for coalescing as the address is updated from row to row. The pitch returned in *pitch by
cudaMallocPitch() is the width in bytes of the allocation. The intended usage of pitch is as a separate
parameter of the allocation, used to compute addresses within the 2D array. Given the row and column
of an array element of type T, the address is computed as:
‎  T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;

For allocations of 2D arrays, it is recommended that programmers consider performing pitch
allocations using cudaMallocPitch(). Due to pitch alignment restrictions in the hardware, this is
especially true if the application will be performing 2D memory copies between different regions of
device memory (whether linear memory or CUDA arrays).



See also:

cudaMalloc, cudaFree, cudaMallocArray, cudaFreeArray, cudaMallocHost ( C API), cudaFreeHost,
cudaMalloc3D, cudaMalloc3DArray, cudaHostAlloc, cuMemAllocPitch