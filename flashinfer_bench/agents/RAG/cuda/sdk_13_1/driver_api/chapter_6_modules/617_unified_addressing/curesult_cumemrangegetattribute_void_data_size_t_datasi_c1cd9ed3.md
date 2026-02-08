# CUresult cuMemRangeGetAttribute (void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count)

Query an attribute of a given memory range.

###### Parameters

**data**

  - A pointers to a memory location where the result of each attribute query will be written to.
**dataSize**

  - Array containing the size of data
**attribute**

  - The attribute to query
**devPtr**

  - Start of the range to query


CUDA Driver API TRM-06703-001 _vRelease Version  |  321


Modules


**count**

  - Size of the range to query

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

###### Description

Query an attribute about the memory range starting at devPtr with a size of count bytes. The
memory range must refer to managed memory allocated via cuMemAllocManaged or declared via
__managed__ variables.

The attribute parameter can take the following values:

CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY: If this attribute is specified, will be

###### ‣ data

interpreted as a 32-bit integer, and dataSize must be 4. The result returned will be 1 if all pages
in the given memory range have read-duplication enabled, or 0 otherwise.
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION: If this attribute is specified,

###### **‣**

data will be interpreted as a 32-bit integer, and dataSize must be 4. The result returned will
be a GPU device id if all pages in the memory range have that GPU as their preferred location,
or it will be CU_DEVICE_CPU if all pages in the memory range have the CPU as their preferred
location, or it will be CU_DEVICE_INVALID if either all the pages don't have the same preferred
location or some of the pages don't have a preferred location at all. Note that the actual location
of the pages in the memory range at the time of the query may be different from the preferred
location.
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY: If this attribute is specified, will be

###### ‣ data

interpreted as an array of 32-bit integers, and dataSize must be a non-zero multiple of 4. The
result returned will be a list of device ids that had CU_MEM_ADVISE_SET_ACCESSED_BY
set for that entire memory range. If any device does not have that advice set for the entire memory
range, that device will not be included. If data is larger than the number of devices that have that
advice set for that memory range, CU_DEVICE_INVALID will be returned in all the extra space
provided. For ex., if dataSize is 12 (i.e. data has 3 elements) and only device 0 has the advice
set, then the result returned will be { 0, CU_DEVICE_INVALID, CU_DEVICE_INVALID }. If
data is smaller than the number of devices that have that advice set, then only as many devices
will be returned as can fit in the array. There is no guarantee on which specific devices will be
returned, however.
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION: If this attribute is specified,

###### **‣**

data will be interpreted as a 32-bit integer, and dataSize must be 4. The result returned
will be the last location to which all pages in the memory range were prefetched explicitly via
cuMemPrefetchAsync. This will either be a GPU id or CU_DEVICE_CPU depending on whether
the last location for prefetch was a GPU or the CPU respectively. If any page in the memory
range was never explicitly prefetched or if all pages were not prefetched to the same location,
CU_DEVICE_INVALID will be returned. Note that this simply returns the last location that the


CUDA Driver API TRM-06703-001 _vRelease Version  |  322


Modules


application requested to prefetch the memory range to. It gives no indication as to whether the
prefetch operation to that location has completed or even begun.
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE: If this attribute

###### **‣**

is specified, data will be interpreted as a CUmemLocationType, and dataSize
must be sizeof(CUmemLocationType). The CUmemLocationType returned will be
CU_MEM_LOCATION_TYPE_DEVICE if all pages in the memory range have the same GPU as
their preferred location, or CUmemLocationType will be CU_MEM_LOCATION_TYPE_HOST
if all pages in the memory range have the CPU as their preferred location, or it will
be CU_MEM_LOCATION_TYPE_HOST_NUMA if all the pages in the memory
range have the same host NUMA node ID as their preferred location or it will be
CU_MEM_LOCATION_TYPE_INVALID if either all the pages don't have the same preferred
location or some of the pages don't have a preferred location at all. Note that the actual location
type of the pages in the memory range at the time of the query may be different from the preferred
location type.

CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID: If this attribute is

###### **‣**

specified, data will be interpreted as a 32-bit integer, and dataSize must be 4. If the
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE query for the same
address range returns CU_MEM_LOCATION_TYPE_DEVICE, it will be a valid device
ordinal or if it returns CU_MEM_LOCATION_TYPE_HOST_NUMA, it will be a valid host
NUMA node ID or if it returns any other location type, the id should be ignored.
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE: If this attribute

###### **‣**

is specified, data will be interpreted as a CUmemLocationType, and dataSize must
be sizeof(CUmemLocationType). The result returned will be the last location to which
all pages in the memory range were prefetched explicitly via cuMemPrefetchAsync. The
CUmemLocationType returned will be CU_MEM_LOCATION_TYPE_DEVICE if the
last prefetch location was a GPU or CU_MEM_LOCATION_TYPE_HOST if it was the
CPU or CU_MEM_LOCATION_TYPE_HOST_NUMA if the last prefetch location was a
specific host NUMA node. If any page in the memory range was never explicitly prefetched
or if all pages were not prefetched to the same location, CUmemLocationType will be
CU_MEM_LOCATION_TYPE_INVALID. Note that this simply returns the last location type that
the application requested to prefetch the memory range to. It gives no indication as to whether the
prefetch operation to that location has completed or even begun.

CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID: If this attribute

###### **‣**

is specified, data will be interpreted as a 32-bit integer, and dataSize must be 4. If the
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE query for the
same address range returns CU_MEM_LOCATION_TYPE_DEVICE, it will be a valid device
ordinal or if it returns CU_MEM_LOCATION_TYPE_HOST_NUMA, it will be a valid host
NUMA node ID or if it returns any other location type, the id should be ignored.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  323


Modules







See also:

cuMemRangeGetAttributes, cuMemPrefetchAsync, cuMemAdvise, cudaMemRangeGetAttribute