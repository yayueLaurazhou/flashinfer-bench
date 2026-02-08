# CUresult cuMemMapArrayAsync (CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream)

Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.

###### Parameters

**mapInfoList**

  - List of CUarrayMapInfo
**count**

  - Count of CUarrayMapInfo in mapInfoList
**hStream**

  - Stream identifier for the stream to use for map or unmap operations

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE

###### Description

Performs map or unmap operations on subregions of sparse CUDA arrays and sparse CUDA
mipmapped arrays. Each operation is specified by a CUarrayMapInfo entry in the mapInfoList


CUDA Driver API TRM-06703-001 _vRelease Version  |  280


Modules


unsigned int deviceBitMask;
unsigned int flags;
unsigned int reserved[2];
} CUarrayMapInfo;

where CUarrayMapInfo::resourceType specifies the type of resource to be operated on. If
CUarrayMapInfo::resourceType is set to CUresourcetype::CU_RESOURCE_TYPE_ARRAY
then CUarrayMapInfo::resource::array must be set to a valid sparse CUDA array
handle. The CUDA array must be either a 2D, 2D layered or 3D CUDA array and
must have been allocated using cuArrayCreate or cuArray3DCreate with the flag
CUDA_ARRAY3D_SPARSE or CUDA_ARRAY3D_DEFERRED_MAPPING.
For CUDA arrays obtained using cuMipmappedArrayGetLevel,
CUDA_ERROR_INVALID_VALUE will be returned. If CUarrayMapInfo::resourceType
is set to CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY then
CUarrayMapInfo::resource::mipmap must be set to a valid sparse CUDA mipmapped array handle. The
CUDA mipmapped array must be either a 2D, 2D layered or 3D CUDA mipmapped array and must
have been allocated using cuMipmappedArrayCreate with the flag CUDA_ARRAY3D_SPARSE or
CUDA_ARRAY3D_DEFERRED_MAPPING.

CUarrayMapInfo::subresourceType specifies the type of subresource within the resource.
CUarraySparseSubresourceType_enum is defined as:
‎  typedef enum CUarraySparseSubresourceType_enum {
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0,
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
} CUarraySparseSubresourceType;

where
CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL
indicates a sparse-miplevel which spans at least one tile in every dimension. The remaining miplevels
which are too small to span at least one tile in any dimension constitute the mip tail region as indicated
by CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL
subresource type.

If CUarrayMapInfo::subresourceType is set to
CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL
then CUarrayMapInfo::subresource::sparseLevel struct must contain valid array
subregion offsets and extents. The CUarrayMapInfo::subresource::sparseLevel::offsetX,
CUarrayMapInfo::subresource::sparseLevel::offsetY and
CUarrayMapInfo::subresource::sparseLevel::offsetZ must specify valid X, Y and Z
offsets respectively. The CUarrayMapInfo::subresource::sparseLevel::extentWidth,
CUarrayMapInfo::subresource::sparseLevel::extentHeight and
CUarrayMapInfo::subresource::sparseLevel::extentDepth must specify valid width, height and
depth extents respectively. These offsets and extents must be aligned to the corresponding tile
dimension. For CUDA mipmapped arrays CUarrayMapInfo::subresource::sparseLevel::level must
specify a valid mip level index. Otherwise, must be zero. For layered CUDA arrays and layered
CUDA mipmapped arrays CUarrayMapInfo::subresource::sparseLevel::layer must specify a valid
layer index. Otherwise, must be zero. CUarrayMapInfo::subresource::sparseLevel::offsetZ must
be zero and CUarrayMapInfo::subresource::sparseLevel::extentDepth must be set to 1 for 2D and


CUDA Driver API TRM-06703-001 _vRelease Version  |  281


Modules


2D layered CUDA arrays and CUDA mipmapped arrays. Tile extents can be obtained by calling
cuArrayGetSparseProperties and cuMipmappedArrayGetSparseProperties

If CUarrayMapInfo::subresourceType is set to
CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL
then CUarrayMapInfo::subresource::miptail struct must contain valid mip tail offset in
CUarrayMapInfo::subresource::miptail::offset and size in CUarrayMapInfo::subresource::miptail::size.
Both, mip tail offset and mip tail size must be aligned to the tile size. For layered CUDA mipmapped
arrays which don't have the flag CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
set in CUDA_ARRAY_SPARSE_PROPERTIES::flags as returned by
cuMipmappedArrayGetSparseProperties, CUarrayMapInfo::subresource::miptail::layer must specify a
valid layer index. Otherwise, must be zero.

If CUarrayMapInfo::resource::array or CUarrayMapInfo::resource::mipmap was created with
CUDA_ARRAY3D_DEFERRED_MAPPING flag set the CUarrayMapInfo::subresourceType and the
contents of CUarrayMapInfo::subresource will be ignored.

CUarrayMapInfo::memOperationType specifies the type of operation. CUmemOperationType is
defined as:
‎  typedef enum CUmemOperationType_enum {
CU_MEM_OPERATION_TYPE_MAP = 1,
CU_MEM_OPERATION_TYPE_UNMAP = 2
} CUmemOperationType;
If CUarrayMapInfo::memOperationType is set to
CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP then the subresource
will be mapped onto the tile pool memory specified by CUarrayMapInfo::memHandle
at offset CUarrayMapInfo::offset. The tile pool allocation has to be created
by specifying the CU_MEM_CREATE_USAGE_TILE_POOL flag when
calling cuMemCreate. Also, CUarrayMapInfo::memHandleType must be set to
CUmemHandleType::CU_MEM_HANDLE_TYPE_GENERIC.

If CUarrayMapInfo::memOperationType is set to
CUmemOperationType::CU_MEM_OPERATION_TYPE_UNMAP then an unmapping operation is
performed. CUarrayMapInfo::memHandle must be NULL.

CUarrayMapInfo::deviceBitMask specifies the list of devices that must map or unmap physical
memory. Currently, this mask must have exactly one bit set, and the corresponding device must
match the device associated with the stream. If CUarrayMapInfo::memOperationType is set to
CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP, the device must also match the
device associated with the tile pool memory allocation as specified by CUarrayMapInfo::memHandle.

CUarrayMapInfo::flags and CUarrayMapInfo::reserved[] are unused and must be set to zero.


See also:

cuMipmappedArrayCreate, cuArrayCreate, cuArray3DCreate, cuMemCreate,
cuArrayGetSparseProperties, cuMipmappedArrayGetSparseProperties


CUDA Driver API TRM-06703-001 _vRelease Version  |  282


Modules