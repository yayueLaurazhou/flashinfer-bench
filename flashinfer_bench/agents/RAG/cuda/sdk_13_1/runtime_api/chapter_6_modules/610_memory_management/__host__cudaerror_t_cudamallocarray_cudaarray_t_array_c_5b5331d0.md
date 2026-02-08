# __host__cudaError_t cudaMallocArray (cudaArray_t *array, const cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags)

Allocate an array on the device.

##### Parameters

**array**

  - Pointer to allocated array in device memory


CUDA Runtime API vRelease Version  |  138


Modules


**desc**

  - Requested channel format
**width**

  - Requested array allocation width
**height**

  - Requested array allocation height
**flags**

  - Requested properties of allocated array

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Allocates a CUDA array according to the cudaChannelFormatDesc structure desc and returns a
handle to the new CUDA array in *array.

The cudaChannelFormatDesc is defined as:

cudaChannelFormatKindUnsigned, or cudaChannelFormatKindFloat.

The flags parameter enables different options to be specified that affect the allocation, as follows.

cudaArrayDefault: This flag's value is defined to be 0 and provides default array allocation

##### **‣**

cudaArraySurfaceLoadStore: Allocates an array that can be read from or written to using a surface

##### **‣**

reference
cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on

##### **‣**

the array.
cudaArraySparse: Allocates a CUDA array without physical backing memory. The subregions

##### **‣**

within this sparse array can later be mapped onto a physical memory allocation by calling
cuMemMapArrayAsync. The physical backing memory must be allocated via cuMemCreate.
cudaArrayDeferredMapping: Allocates a CUDA array without physical backing memory.

##### **‣**

The entire array can later be mapped onto a physical memory allocation by calling
cuMemMapArrayAsync. The physical backing memory must be allocated via cuMemCreate.

width and height must meet certain size requirements. See cudaMalloc3DArray() for more details.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  139


Modules





See also:

cudaMalloc, cudaMallocPitch, cudaFree, cudaFreeArray, cudaMallocHost ( C API), cudaFreeHost,
cudaMalloc3D, cudaMalloc3DArray, cudaHostAlloc, cuArrayCreate