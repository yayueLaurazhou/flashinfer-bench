# CUresult cuTensorMapEncodeIm2colWide (CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapIm2ColWideMode mode, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)

Create a tensor map descriptor object representing im2col memory region, but where the elements are
exclusively loaded along the W dimension.

###### Parameters

**tensorMap**

  - Tensor map object to create
**tensorDataType**

  - Tensor data type
**tensorRank**

  - Dimensionality of tensor; must be at least 3
**globalAddress**

  - Starting address of memory region described by tensor
**globalDim**

  - Array containing tensor size (number of elements) along each of the tensorRank dimensions
**globalStrides**

  - Array containing stride size (in bytes) along each of the tensorRank - 1 dimensions
**pixelBoxLowerCornerWidth**

  - Width offset of left box corner
**pixelBoxUpperCornerWidth**

  - Width offset of right box corner
**channelsPerPixel**

  - Number of channels per pixel


CUDA Driver API TRM-06703-001 _vRelease Version  |  541


Modules


**pixelsPerColumn**

  - Number of pixels per column
**elementStrides**

  - Array containing traversal stride in each of the tensorRank dimensions
**interleave**

  - Type of interleaved layout the tensor addresses
**mode**

  - W or W128 mode
**swizzle**

  - Bank swizzling pattern inside shared memory
**l2Promotion**

  - L2 promotion size
**oobFill**

  - Indicate whether zero or special NaN constant will be used to fill out-of-bound elements

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a descriptor for Tensor Memory Access (TMA) object specified by the parameters describing
a im2col memory layout and where the row is always loaded along the W dimensuin and returns it in
tensorMap. This assumes the tensor layout in memory is either NDHWC, NHWC, or NWC.

This API is only supported on devices of compute capability 10.0 or higher. Additionally, a tensor map
object is an opaque value, and, as such, should only be accessed through CUDA APIs and PTX.

The parameters passed are bound to the following requirements:

address must be aligned to 64 bytes.

###### ‣ tensorMap

has to be an enum from CUtensorMapDataType which is defined as:

###### ‣ tensorDataType

‎  typedef enum CUtensorMapDataType_enum {
CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,    // 1 byte
CU_TENSOR_MAP_DATA_TYPE_UINT16,     // 2 bytes
CU_TENSOR_MAP_DATA_TYPE_UINT32,     // 4 bytes
CU_TENSOR_MAP_DATA_TYPE_INT32,      // 4 bytes
CU_TENSOR_MAP_DATA_TYPE_UINT64,     // 8 bytes
CU_TENSOR_MAP_DATA_TYPE_INT64,      // 8 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT16,     // 2 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT32,     // 4 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT64,     // 8 bytes
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,    // 2 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,   // 4 bytes
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,    // 4 bytes
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ   // 4 bytes
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,  // 4 bits
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,  // 4 bits
CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B  // 6 bits
} CUtensorMapDataType;


CUDA Driver API TRM-06703-001 _vRelease Version  |  542


Modules


CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B copies '16 x U4' packed
values to memory aligned as 8 bytes. There are no gaps between packed values.
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B copies '16 x U4' packed values to
memory aligned as 16 bytes. There are 8 byte gaps between every 8 byte chunk of packed values.
CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B copies '16 x U6' packed values to memory
aligned as 16 bytes. There are 4 byte gaps between every 12 byte chunk of packed values.

, which specifies the number of tensor dimensions, must be 3, 4, or 5.

###### ‣ tensorRank

, which specifies the starting address of the memory region described, must be

###### ‣ globalAddress

16 byte aligned. The following requirements need to also be met:

When is CU_TENSOR_MAP_INTERLEAVE_32B, must

###### ‣ interleave globalAddress

be 32 byte aligned.
When is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or

###### ‣ tensorDataType

CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, globalAddress must be 32 byte
aligned.

array, which specifies tensor size of each of the dimensions, must be

###### ‣ globalDim tensorRank

non-zero and less than or equal to 2^32. Additionally, the following requirements need to be met
for the packed data types:

When is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or

###### ‣ tensorDataType

CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, globalDim[0] must be a multiple of
128.
When is CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,

###### ‣ tensorDataType

globalDim[0] must be a multiple of 2.
Dimension for the packed data types must reflect the number of individual U# values.

###### **‣**

array, which specifies tensor stride of each of the lower             - 1

###### ‣ globalStrides tensorRank

dimensions in bytes, must be a multiple of 16 and less than 2^40. Additionally, the following
requirements need to be met:

When is CU_TENSOR_MAP_INTERLEAVE_32B, the strides must be a

###### ‣ interleave

multiple of 32.
When is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or

###### ‣ tensorDataType

CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, the strides must be a multiple of 32.
Each following dimension specified includes previous dimension stride:
‎  globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) +
padding[0];
for (i = 1; i < tensorRank - 1; i++)
globalStrides[i] = globalStrides[i – 1] * (globalDim[i] +
padding[i]);
assert(globalStrides[i] >= globalDim[i]);

specifies the coordinate offset W of the bounding box from left

###### ‣ pixelBoxLowerCornerWidth

corner. The offset must be within range [-32768, 32767].


CUDA Driver API TRM-06703-001 _vRelease Version  |  543


Modules


specifies the coordinate offset W of the bounding box from

###### ‣ pixelBoxUpperCornerWidth

right corner. The offset must be within range [-32768, 32767].

The bounding box specified by pixelBoxLowerCornerWidth and
pixelBoxUpperCornerWidth must have non-zero area. Note that the size of the box along D and
H dimensions is always equal to one.

, which specifies the number of elements which must be

###### ‣ channelsPerPixel

accessed along C dimension, must be less than or equal to 256. Additionally, when
tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, channelsPerPixel must be 128.

, which specifies the number of elements that must be accessed along

###### ‣ pixelsPerColumn

the W dimension, must be less than or equal to 1024. This field is ignored when mode is
CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128.

array, which specifies the iteration step along each of the

###### ‣ elementStrides tensorRank

dimensions, must be non-zero and less than or equal to 8. Note that when interleave is
CU_TENSOR_MAP_INTERLEAVE_NONE, the first element of this array is ignored since TMA
doesn’t support the stride for dimension zero. When all elements of the elementStrides array
are one, boxDim specifies the number of elements to load. However, if elementStrides[i] is
not equal to one for some i, then TMA loads ceil( boxDim[i] / elementStrides[i]) number of
elements along i-th dimension. To load N elements along i-th dimension, boxDim[i] must be set to
N * elementStrides[i].

specifies the interleaved layout of type CUtensorMapInterleave, which is defined

###### ‣ interleave

as:
‎  typedef enum CUtensorMapInterleave_enum {
CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
CU_TENSOR_MAP_INTERLEAVE_16B,
CU_TENSOR_MAP_INTERLEAVE_32B
} CUtensorMapInterleave;
TMA supports interleaved layouts like NC/8HWC8 where C8 utilizes 16 bytes in memory
assuming 2 byte per channel or NC/16HWC16 where C16 uses 32 bytes. When interleave is
CU_TENSOR_MAP_INTERLEAVE_NONE, the bounding box inner dimension (computed as
channelsPerPixel multiplied by element size in bytes derived from tensorDataType)
must be less than or equal to the swizzle size.

CU_TENSOR_MAP_SWIZZLE_64B requires the bounding box inner dimension to be <= 64.

###### **‣**

CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box

###### **‣**

inner dimension to be <= 128. Additionally, tensorDataType of
CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B requires interleave to be
CU_TENSOR_MAP_INTERLEAVE_NONE.

, which describes loading of elements loaded along the W dimension, has to be one of the

###### ‣ mode

following CUtensorMapIm2ColWideMode types:
‎     CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128


CUDA Driver API TRM-06703-001 _vRelease Version  |  544


Modules


CU_TENSOR_MAP_IM2COL_WIDE_MODE_W allows the number of elements loaded along
the W dimension to be specified via the pixelsPerColumn field.

, which specifies the shared memory bank swizzling pattern, must be one of the

###### ‣ swizzle

following CUtensorMapSwizzle modes (other swizzle modes are not supported):
‎  typedef enum CUtensorMapSwizzle_enum {
CU_TENSOR_MAP_SWIZZLE_64B,          // Swizzle 16B chunks
within 64B span
CU_TENSOR_MAP_SWIZZLE_128B,         // Swizzle 16B chunks
within 128B span
CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,     // Swizzle 32B chunks
within 128B span
} CUtensorMapSwizzle;
Data are organized in a specific order in global memory; however, this may not match the order
in which the application accesses data in shared memory. This difference in data organization
may cause bank conflicts when shared memory is accessed. In order to avoid this problem,
data can be loaded to shared memory with shuffling across shared memory banks. When the
tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B, only the
following swizzle modes are supported:

CU_TENSOR_MAP_SWIZZLE_128B (Load & Store)

###### **‣**

CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B (Load & Store) When the

###### **‣**

tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, only the
following swizzle modes are supported:
CU_TENSOR_MAP_SWIZZLE_128B (Load only)

###### **‣**

CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B (Load only)

###### **‣**

specifies L2 fetch size which indicates the byte granularity at which L2 requests

###### ‣ l2Promotion

are filled from DRAM. It must be of type CUtensorMapL2promotion, which is defined as:
‎  typedef enum CUtensorMapL2promotion_enum {
CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
CU_TENSOR_MAP_L2_PROMOTION_L2_256B
} CUtensorMapL2promotion;

, which indicates whether zero or a special NaN constant should be used to fill out-of###### ‣ oobFill
bound elements, must be of type CUtensorMapFloatOOBfill which is defined as:
‎  typedef enum CUtensorMapFloatOOBfill_enum {
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
} CUtensorMapFloatOOBfill;
Note that CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
can only be used when tensorDataType represents a floating-point data type, and
when tensorDataType is not CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, and
CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B.


See also:

cuTensorMapEncodeTiled, cuTensorMapEncodeIm2col, cuTensorMapReplaceAddress


CUDA Driver API TRM-06703-001 _vRelease Version  |  545


Modules