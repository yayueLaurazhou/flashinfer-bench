# __host__cudaError_t cudaMalloc3D (cudaPitchedPtr *pitchedDevPtr, cudaExtent extent)

Allocates logical 1D, 2D, or 3D memory objects on the device.

##### Parameters

**pitchedDevPtr**

  - Pointer to allocated pitched device memory
**extent**

  - Requested allocation size (width field in bytes)

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Allocates at least width * height * depth bytes of linear memory on the device and returns
a cudaPitchedPtr in which ptr is a pointer to the allocated memory. The function may pad the
allocation to ensure hardware alignment requirements are met. The pitch returned in the pitch field of
pitchedDevPtr is the width in bytes of the allocation.

The returned cudaPitchedPtr contains additional fields xsize and ysize, the logical width and
height of the allocation, which are equivalent to the width and height extent parameters
provided by the programmer during allocation.

For allocations of 2D and 3D objects, it is highly recommended that programmers perform allocations
using cudaMalloc3D() or cudaMallocPitch(). Due to alignment restrictions in the hardware, this
is especially true if the application will be performing memory copies involving 2D or 3D objects
(whether linear memory or CUDA arrays).



See also:

cudaMallocPitch, cudaFree, cudaMemcpy3D, cudaMemset3D, cudaMalloc3DArray, cudaMallocArray,
cudaFreeArray, cudaMallocHost ( C API), cudaFreeHost, cudaHostAlloc, make_cudaPitchedPtr,
make_cudaExtent, cuMemAllocPitch


CUDA Runtime API vRelease Version  |  135


Modules