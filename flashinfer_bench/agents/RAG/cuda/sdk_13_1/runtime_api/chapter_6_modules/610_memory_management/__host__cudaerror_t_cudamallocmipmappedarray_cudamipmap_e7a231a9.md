# __host__cudaError_t cudaMallocMipmappedArray (cudaMipmappedArray_t *mipmappedArray, const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int numLevels, unsigned int flags)

Allocate a mipmapped array on the device.

##### Parameters

**mipmappedArray**

  - Pointer to allocated mipmapped array in device memory
**desc**

  - Requested channel format
**extent**

  - Requested allocation size (width field in elements)
**numLevels**

  - Number of mipmap levels to allocate
**flags**

  - Flags for extensions


CUDA Runtime API vRelease Version  |  143


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Allocates a CUDA mipmapped array according to the cudaChannelFormatDesc structure desc
and returns a handle to the new CUDA mipmapped array in *mipmappedArray. numLevels
specifies the number of mipmap levels to be allocated. This value is clamped to the range [1, 1 +
floor(log2(max(width, height, depth)))].

The cudaChannelFormatDesc is defined as:

cudaChannelFormatKindUnsigned, or cudaChannelFormatKindFloat.

cudaMallocMipmappedArray() can allocate the following:

A 1D mipmapped array is allocated if the height and depth extents are both zero.

##### **‣**

A 2D mipmapped array is allocated if only the depth extent is zero.

##### **‣**

A 3D mipmapped array is allocated if all three extents are non-zero.

##### **‣**

A 1D layered CUDA mipmapped array is allocated if only the height extent is zero and the

##### **‣**

cudaArrayLayered flag is set. Each layer is a 1D mipmapped array. The number of layers is
determined by the depth extent.
A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and the

##### **‣**

cudaArrayLayered flag is set. Each layer is a 2D mipmapped array. The number of layers is
determined by the depth extent.
A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the

##### **‣**

cudaArrayCubemap flag is set. Width must be equal to height, and depth must be six. The order of
the six layers in memory is the same as that listed in cudaGraphicsCubeFace.
A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, and both,

##### **‣**

cudaArrayCubemap and cudaArrayLayered flags are set. Width must be equal to height, and depth
must be a multiple of six. A cubemap layered CUDA mipmapped array is a special type of 2D
layered CUDA mipmapped array that consists of a collection of cubemap mipmapped arrays. The
first six layers represent the first cubemap mipmapped array, the next six layers form the second
cubemap mipmapped array, and so on.

The flags parameter enables different options to be specified that affect the allocation, as follows.

cudaArrayDefault: This flag's value is defined to be 0 and provides default mipmapped array

##### **‣**

allocation
cudaArrayLayered: Allocates a layered CUDA mipmapped array, with the depth extent indicating

##### **‣**

the number of layers


CUDA Runtime API vRelease Version  |  144


Modules


cudaArrayCubemap: Allocates a cubemap CUDA mipmapped array. Width must be equal to

##### **‣**

height, and depth must be six. If the cudaArrayLayered flag is also set, depth must be a multiple of
six.
cudaArraySurfaceLoadStore: This flag indicates that individual mipmap levels of the CUDA

##### **‣**

mipmapped array will be read from or written to using a surface reference.
cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on

##### **‣**

the CUDA array. Texture gather can only be performed on 2D CUDA mipmapped arrays, and the
gather operations are performed only on the most detailed mipmap level.
cudaArraySparse: Allocates a CUDA mipmapped array without physical backing memory.

##### **‣**

The subregions within this sparse array can later be mapped onto a physical memory allocation
by calling cuMemMapArrayAsync. This flag can only be used for creating 2D, 3D or 2D
layered sparse CUDA mipmapped arrays. The physical backing memory must be allocated via
cuMemCreate.
cudaArrayDeferredMapping: Allocates a CUDA mipmapped array without physical backing

##### **‣**

memory. The entire array can later be mapped onto a physical memory allocation by calling
cuMemMapArrayAsync. The physical backing memory must be allocated via cuMemCreate.

The width, height and depth extents must meet certain size requirements as listed in the following table.
All values are specified in elements.














|CUDA<br>array type|Valid extents that must always be met<br>{(width range in elements), (height<br>range), (depth range)}|Valid extents with<br>cudaArraySurfaceLoadStore set<br>{(width range in elements), (height<br>range), (depth range)}|
|---|---|---|
|1D|{ (1,maxTexture1DMipmap), 0, 0 }|{ (1,maxSurface1D), 0, 0 }|
|2D|{ (1,maxTexture2DMipmap[0]),<br>(1,maxTexture2DMipmap[1]), 0 }|{ (1,maxSurface2D[0]), (1,maxSurface2D[1]),<br>0 }|
|3D|{ (1,maxTexture3D[0]), (1,maxTexture3D[1]),<br>(1,maxTexture3D[2]) } OR<br>{ (1,maxTexture3DAlt[0]),<br>(1,maxTexture3DAlt[1]),<br>(1,maxTexture3DAlt[2]) }|{ (1,maxSurface3D[0]), (1,maxSurface3D[1]),<br>(1,maxSurface3D[2]) }|
|1D Layered|{ (1,maxTexture1DLayered[0]), 0,<br>(1,maxTexture1DLayered[1]) }|{ (1,maxSurface1DLayered[0]), 0,<br>(1,maxSurface1DLayered[1]) }|
|2D Layered|{ (1,maxTexture2DLayered[0]),<br>(1,maxTexture2DLayered[1]),<br>(1,maxTexture2DLayered[2]) }|{ (1,maxSurface2DLayered[0]),<br>(1,maxSurface2DLayered[1]),<br>(1,maxSurface2DLayered[2]) }|
|Cubemap|{ (1,maxTextureCubemap),<br>(1,maxTextureCubemap), 6 }|{ (1,maxSurfaceCubemap),<br>(1,maxSurfaceCubemap), 6 }|



CUDA Runtime API vRelease Version  |  145


Modules






|CUDA<br>array type|Valid extents that must always be met<br>{(width range in elements), (height<br>range), (depth range)}|Valid extents with<br>cudaArraySurfaceLoadStore set<br>{(width range in elements), (height<br>range), (depth range)}|
|---|---|---|
|Cubemap<br>Layered|{ (1,maxTextureCubemapLayered[0]),<br>(1,maxTextureCubemapLayered[0]),<br>(1,maxTextureCubemapLayered[1]) }|{ (1,maxSurfaceCubemapLayered[0]),<br>(1,maxSurfaceCubemapLayered[0]),<br>(1,maxSurfaceCubemapLayered[1]) }|



See also:

cudaMalloc3D, cudaMalloc, cudaMallocPitch, cudaFree, cudaFreeArray, cudaMallocHost ( C API),
cudaFreeHost, cudaHostAlloc, make_cudaExtent, cuMipmappedArrayCreate