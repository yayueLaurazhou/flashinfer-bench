# CUresult cuMemRangeGetAttributes (void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count)

Query attributes of a given memory range.

###### Parameters

**data**

  - A two-dimensional array containing pointers to memory locations where the result of each
attribute query will be written to.
**dataSizes**

  - Array containing the sizes of each result
**attributes**

  - An array of attributes to query (numAttributes and the number of attributes in this array should
match)
**numAttributes**

  - Number of attributes to query
**devPtr**

  - Start of the range to query
**count**

  - Size of the range to query

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

###### Description

Query attributes of the memory range starting at devPtr with a size of count bytes. The
memory range must refer to managed memory allocated via cuMemAllocManaged or declared via
__managed__ variables. The attributes array will be interpreted to have numAttributes
entries. The dataSizes array will also be interpreted to have numAttributes entries. The results
of the query will be stored in data.

The list of supported attributes are given below. Please refer to cuMemRangeGetAttribute for attribute
descriptions and restrictions.


CUDA Driver API TRM-06703-001 _vRelease Version  |  324


Modules


CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY

###### **‣**

CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION

###### **‣**

CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY

###### **‣**

CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION

###### **‣**

CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE

###### **‣**

CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID

###### **‣**

CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE

###### **‣**

CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemRangeGetAttribute, cuMemAdvise, cuMemPrefetchAsync, cudaMemRangeGetAttributes