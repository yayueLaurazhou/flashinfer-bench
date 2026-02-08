# CUresult cuPointerGetAttributes (unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr)

Returns information about a pointer.

###### Parameters

**numAttributes**

  - Number of attributes to query


CUDA Driver API TRM-06703-001 _vRelease Version  |  328


Modules


**attributes**

  - An array of attributes to query (numAttributes and the number of attributes in this array should
match)
**data**

  - A two-dimensional array containing pointers to memory locations where the result of each
attribute query will be written to.
**ptr**

  - Pointer to query

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

###### Description

The supported attributes are (refer to cuPointerGetAttribute for attribute descriptions and restrictions):

CU_POINTER_ATTRIBUTE_CONTEXT

###### **‣**

CU_POINTER_ATTRIBUTE_MEMORY_TYPE

###### **‣**

CU_POINTER_ATTRIBUTE_DEVICE_POINTER

###### **‣**

CU_POINTER_ATTRIBUTE_HOST_POINTER

###### **‣**

CU_POINTER_ATTRIBUTE_SYNC_MEMOPS

###### **‣**

CU_POINTER_ATTRIBUTE_BUFFER_ID

###### **‣**

CU_POINTER_ATTRIBUTE_IS_MANAGED

###### **‣**

CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL

###### **‣**

CU_POINTER_ATTRIBUTE_RANGE_START_ADDR

###### **‣**

CU_POINTER_ATTRIBUTE_RANGE_SIZE

###### **‣**

CU_POINTER_ATTRIBUTE_MAPPED

###### **‣**

CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE

###### **‣**

CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES

###### **‣**

CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE

###### **‣**

CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE

###### **‣**

Unlike cuPointerGetAttribute, this function will not return an error when the ptr encountered is not a
valid CUDA pointer. Instead, the attributes are assigned default NULL values and CUDA_SUCCESS
is returned.

If ptr was not allocated by, mapped by, or registered with a CUcontext which uses UVA (Unified
Virtual Addressing), CUDA_ERROR_INVALID_CONTEXT is returned.


Note:


CUDA Driver API TRM-06703-001 _vRelease Version  |  329


Modules


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuPointerGetAttribute, cuPointerSetAttribute, cudaPointerGetAttributes