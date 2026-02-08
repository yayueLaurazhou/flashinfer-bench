# CUresult cuArray3DCreate (CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)

Creates a 3D CUDA array.

###### Parameters

**pHandle**

  - Returned array
**pAllocateArray**

  - 3D array descriptor

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN


CUDA Driver API TRM-06703-001 _vRelease Version  |  175


Modules

###### Description

Creates a CUDA array according to the CUDA_ARRAY3D_DESCRIPTOR structure
pAllocateArray and returns a handle to the new CUDA array in *pHandle. The
CUDA_ARRAY3D_DESCRIPTOR is defined as:


,, and are the width, height, and depth of the CUDA array (in elements);

###### ‣ Width Height Depth

the following types of CUDA arrays can be allocated:

A 1D array is allocated if and extents are both zero.

###### ‣ Height Depth

A 2D array is allocated if only extent is zero.

###### ‣ Depth

A 3D array is allocated if all three extents are non-zero.

###### **‣**

A 1D layered CUDA array is allocated if only is zero and the

###### ‣ Height

CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number of layers is
determined by the depth extent.
A 2D layered CUDA array is allocated if all three extents are non-zero and the

###### **‣**

CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number of layers is
determined by the depth extent.
A cubemap CUDA array is allocated if all three extents are non-zero and the

###### **‣**

CUDA_ARRAY3D_CUBEMAP flag is set. Width must be equal to Height, and Depth
must be six. A cubemap is a special type of 2D layered CUDA array, where the six layers
represent the six faces of a cube. The order of the six layers in memory is the same as that
listed in CUarray_cubemap_face.
A cubemap layered CUDA array is allocated if all three extents are non-zero, and both,

###### **‣**

CUDA_ARRAY3D_CUBEMAP and CUDA_ARRAY3D_LAYERED flags are set. Width
must be equal to Height, and Depth must be a multiple of six. A cubemap layered CUDA
array is a special type of 2D layered CUDA array that consists of a collection of cubemaps.
The first six layers represent the first cubemap, the next six layers form the second cubemap,
and so on.

Format specifies the format of the elements; CUarray_format is defined as:

###### **‣**

CUDA Driver API TRM-06703-001 _vRelease Version  |  176


Modules


specifies the number of packed components per CUDA array element; it may be

###### ‣ NumChannels

1, 2, or 4;

Flags may be set to

###### **‣**

CUDA_ARRAY3D_LAYERED to enable creation of layered CUDA arrays. If this flag is set,

###### **‣**

Depth specifies the number of layers, not the depth of a 3D array.
CUDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to the CUDA

###### **‣**

array. If this flag is not set, cuSurfRefSetArray will fail when attempting to bind the CUDA
array to a surface reference.
CUDA_ARRAY3D_CUBEMAP to enable creation of cubemaps. If this flag is set,

###### ‣ Width

must be equal to Height, and Depth must be six. If the CUDA_ARRAY3D_LAYERED
flag is also set, then Depth must be a multiple of six.
CUDA_ARRAY3D_TEXTURE_GATHER to indicate that the CUDA array will be used for

###### **‣**

texture gather. Texture gather can only be performed on 2D CUDA arrays.


CUDA Driver API TRM-06703-001 _vRelease Version  |  177


Modules


Width, Height and Depth must meet certain size requirements as listed in the following
table. All values are specified in elements. Note that for brevity's sake, the full name of the
device attribute is not specified. For ex., TEXTURE1D_WIDTH refers to the device attribute
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH.

Note that 2D CUDA arrays have different size requirements if the
CUDA_ARRAY3D_TEXTURE_GATHER flag is set. Width and Height must not be
greater than CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH and
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT respectively, in that
case.



CUDA array type Valid extents that must always be
met {(width range in elements),
(height range), (depth range)}



Valid extents with
CUDA_ARRAY3D_SURFACE_LDST
set {(width range in elements),
(height range), (depth range)}



1D { (1,TEXTURE1D_WIDTH), 0, 0 } { (1,SURFACE1D_WIDTH), 0, 0 }



2D { (1,TEXTURE2D_WIDTH),
(1,TEXTURE2D_HEIGHT), 0 }



{ (1,SURFACE2D_WIDTH),
(1,SURFACE2D_HEIGHT), 0 }



3D { (1,TEXTURE3D_WIDTH), { (1,SURFACE3D_WIDTH),
(1,TEXTURE3D_HEIGHT), (1,SURFACE3D_HEIGHT),
(1,TEXTURE3D_DEPTH) } OR (1,SURFACE3D_DEPTH) }
{ (1,TEXTURE3D_WIDTH_ALTERNATE),
(1,TEXTURE3D_HEIGHT_ALTERNATE),
(1,TEXTURE3D_DEPTH_ALTERNATE) }



1D Layered { (1,TEXTURE1D_LAYERED_WIDTH),{ (1,SURFACE1D_LAYERED_WIDTH),
0, 0,
(1,TEXTURE1D_LAYERED_LAYERS) }(1,SURFACE1D_LAYERED_LAYERS) }



2D Layered { (1,TEXTURE2D_LAYERED_WIDTH),{ (1,SURFACE2D_LAYERED_WIDTH),
(1,TEXTURE2D_LAYERED_HEIGHT),(1,SURFACE2D_LAYERED_HEIGHT),
(1,TEXTURE2D_LAYERED_LAYERS) }(1,SURFACE2D_LAYERED_LAYERS) }



Cubemap { (1,TEXTURECUBEMAP_WIDTH),{ (1,SURFACECUBEMAP_WIDTH),
(1,TEXTURECUBEMAP_WIDTH), (1,SURFACECUBEMAP_WIDTH),
6 } 6 }



Cubemap Layered { (1,TEXTURECUBEMAP_LAYERED_WIDTH),{ (1,SURFACECUBEMAP_LAYERED_WIDTH),
(1,TEXTURECUBEMAP_LAYERED_WIDTH),(1,SURFACECUBEMAP_LAYERED_WIDTH),
(1,TEXTURECUBEMAP_LAYERED_LAYERS) }(1,SURFACECUBEMAP_LAYERED_LAYERS) }



Here are examples of CUDA array descriptions:

Description for a CUDA array of 2048 floats:
‎  CUDA_ARRAY3D_DESCRIPTOR desc;


CUDA Driver API TRM-06703-001 _vRelease Version  |  178


Modules


Description for a 64 x 64 CUDA array of floats:


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor, cuMemAlloc,
cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync, cuMemcpy2DUnaligned,
cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaMalloc3DArray