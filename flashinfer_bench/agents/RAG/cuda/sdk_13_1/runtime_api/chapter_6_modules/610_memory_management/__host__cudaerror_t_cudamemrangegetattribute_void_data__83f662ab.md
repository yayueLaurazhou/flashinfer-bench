# __host__cudaError_t cudaMemRangeGetAttribute (void *data, size_t dataSize, cudaMemRangeAttribute attribute, const void *devPtr, size_t count)

Query an attribute of a given memory range.

##### Parameters

**data**

  - A pointers to a memory location where the result of each attribute query will be written to.
**dataSize**

  - Array containing the size of data
**attribute**

  - The attribute to query
**devPtr**

  - Start of the range to query
**count**

  - Size of the range to query

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Query an attribute about the memory range starting at devPtr with a size of count bytes. The
memory range must refer to managed memory allocated via cudaMallocManaged or declared via
__managed__ variables.

The attribute parameter can take the following values:

cudaMemRangeAttributeReadMostly: If this attribute is specified, will be interpreted as a

##### ‣ data

32-bit integer, and dataSize must be 4. The result returned will be 1 if all pages in the given
memory range have read-duplication enabled, or 0 otherwise.


CUDA Runtime API vRelease Version  |  189


Modules


cudaMemRangeAttributePreferredLocation: If this attribute is specified, will be interpreted

##### ‣ data

as a 32-bit integer, and dataSize must be 4. The result returned will be a GPU device
id if all pages in the memory range have that GPU as their preferred location, or it will be
cudaCpuDeviceId if all pages in the memory range have the CPU as their preferred location, or it
will be cudaInvalidDeviceId if either all the pages don't have the same preferred location or some
of the pages don't have a preferred location at all. Note that the actual location of the pages in the
memory range at the time of the query may be different from the preferred location.
cudaMemRangeAttributeAccessedBy: If this attribute is specified, will be interpreted as an

##### ‣ data

array of 32-bit integers, and dataSize must be a non-zero multiple of 4. The result returned will
be a list of device ids that had cudaMemAdviceSetAccessedBy set for that entire memory range.
If any device does not have that advice set for the entire memory range, that device will not be
included. If data is larger than the number of devices that have that advice set for that memory
range, cudaInvalidDeviceId will be returned in all the extra space provided. For ex., if dataSize
is 12 (i.e. data has 3 elements) and only device 0 has the advice set, then the result returned
will be { 0, cudaInvalidDeviceId, cudaInvalidDeviceId }. If data is smaller than the number of
devices that have that advice set, then only as many devices will be returned as can fit in the array.
There is no guarantee on which specific devices will be returned, however.
cudaMemRangeAttributeLastPrefetchLocation: If this attribute is specified, will

##### ‣ data

be interpreted as a 32-bit integer, and dataSize must be 4. The result returned will be
the last location to which all pages in the memory range were prefetched explicitly via
cudaMemPrefetchAsync. This will either be a GPU id or cudaCpuDeviceId depending on whether
the last location for prefetch was a GPU or the CPU respectively. If any page in the memory
range was never explicitly prefetched or if all pages were not prefetched to the same location,
cudaInvalidDeviceId will be returned. Note that this simply returns the last location that the
applicaton requested to prefetch the memory range to. It gives no indication as to whether the
prefetch operation to that location has completed or even begun.
cudaMemRangeAttributePreferredLocationType: If this attribute is specified, will be

##### ‣ data

interpreted as a cudaMemLocationType, and dataSize must be sizeof(cudaMemLocationType).
The cudaMemLocationType returned will be cudaMemLocationTypeDevice if all pages in
the memory range have the same GPU as their preferred location, or cudaMemLocationType
will be cudaMemLocationTypeHost if all pages in the memory range have the CPU as their
preferred location, or or it will be cudaMemLocationTypeHostNuma if all the pages in the
memory range have the same host NUMA node ID as their preferred location or it will be
cudaMemLocationTypeInvalid if either all the pages don't have the same preferred location or
some of the pages don't have a preferred location at all. Note that the actual location type of the
pages in the memory range at the time of the query may be different from the preferred location
type.

cudaMemRangeAttributePreferredLocationId: If this attribute is specified,

##### **‣**

data will be interpreted as a 32-bit integer, and dataSize must be 4. If the
cudaMemRangeAttributePreferredLocationType query for the same address range
returns cudaMemLocationTypeDevice, it will be a valid device ordinal or if it returns


CUDA Runtime API vRelease Version  |  190


Modules


cudaMemLocationTypeHostNuma, it will be a valid host NUMA node ID or if it returns any
other location type, the id should be ignored.
cudaMemRangeAttributeLastPrefetchLocationType: If this attribute is specified, will be

##### ‣ data

interpreted as a cudaMemLocationType, and dataSize must be sizeof(cudaMemLocationType).
The result returned will be the last location type to which all pages in the memory range
were prefetched explicitly via cuMemPrefetchAsync. The cudaMemLocationType
returned will be cudaMemLocationTypeDevice if the last prefetch location was the GPU
or cudaMemLocationTypeHost if it was the CPU or cudaMemLocationTypeHostNuma if
the last prefetch location was a specific host NUMA node. If any page in the memory range
was never explicitly prefetched or if all pages were not prefetched to the same location,
CUmemLocationType will be cudaMemLocationTypeInvalid. Note that this simply returns the last
location type that the application requested to prefetch the memory range to. It gives no indication
as to whether the prefetch operation to that location has completed or even begun.

cudaMemRangeAttributeLastPrefetchLocationId: If this attribute is specified,

##### **‣**

data will be interpreted as a 32-bit integer, and dataSize must be 4. If the
cudaMemRangeAttributeLastPrefetchLocationType query for the same address range
returns cudaMemLocationTypeDevice, it will be a valid device ordinal or if it returns
cudaMemLocationTypeHostNuma, it will be a valid host NUMA node ID or if it returns any
other location type, the id should be ignored.













See also:

cudaMemRangeGetAttributes, cudaMemPrefetchAsync, cudaMemAdvise, cuMemRangeGetAttribute


CUDA Runtime API vRelease Version  |  191


Modules