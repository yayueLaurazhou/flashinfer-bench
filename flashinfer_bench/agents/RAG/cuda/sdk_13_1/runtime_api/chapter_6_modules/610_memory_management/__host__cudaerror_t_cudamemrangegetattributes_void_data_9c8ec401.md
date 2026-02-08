# __host__cudaError_t cudaMemRangeGetAttributes (void **data, size_t *dataSizes, cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count)

Query attributes of a given memory range.

##### Parameters

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

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Query attributes of the memory range starting at devPtr with a size of count bytes. The memory
range must refer to managed memory allocated via cudaMallocManaged or declared via __managed__
variables. The attributes array will be interpreted to have numAttributes entries. The
dataSizes array will also be interpreted to have numAttributes entries. The results of the query
will be stored in data.

The list of supported attributes are given below. Please refer to cudaMemRangeGetAttribute for
attribute descriptions and restrictions.

cudaMemRangeAttributeReadMostly

##### **‣**

cudaMemRangeAttributePreferredLocation

##### **‣**

cudaMemRangeAttributeAccessedBy

##### **‣**

cudaMemRangeAttributeLastPrefetchLocation

##### **‣**

:: cudaMemRangeAttributePreferredLocationType

##### **‣**

CUDA Runtime API vRelease Version  |  192


Modules



:: cudaMemRangeAttributePreferredLocationId

##### **‣**

:: cudaMemRangeAttributeLastPrefetchLocationType

##### **‣**

:: cudaMemRangeAttributeLastPrefetchLocationId

##### **‣**

See also:

cudaMemRangeGetAttribute, cudaMemAdvise, cudaMemPrefetchAsync, cuMemRangeGetAttributes