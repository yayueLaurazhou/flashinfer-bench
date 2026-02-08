# __host__cudaError_t cudaDeviceGetGraphMemAttribute (int device, cudaGraphMemAttributeType attr, void *value)

Query asynchronous allocation attributes related to graphs.

##### Parameters

**device**

  - Specifies the scope of the query
**attr**

  - attribute to get
**value**

  - retrieved value

##### Returns

cudaSuccess, cudaErrorInvalidDevice

##### Description

Valid attributes are:

cudaGraphMemAttrUsedMemCurrent: Amount of memory, in bytes, currently associated with

##### **‣**

graphs
cudaGraphMemAttrUsedMemHigh: High watermark of memory, in bytes, associated with graphs

##### **‣**

since the last time it was reset. High watermark can only be reset to zero.


CUDA Runtime API vRelease Version  |  320


Modules


cudaGraphMemAttrReservedMemCurrent: Amount of memory, in bytes, currently allocated for

##### **‣**

use by the CUDA graphs asynchronous allocator.
cudaGraphMemAttrReservedMemHigh: High watermark of memory, in bytes, currently allocated

##### **‣**

for use by the CUDA graphs asynchronous allocator.











See also:

cudaDeviceSetGraphMemAttribute, cudaGraphAddMemAllocNode, cudaGraphAddMemFreeNode,
cudaDeviceGraphMemTrim, cudaMallocAsync, cudaFreeAsync