# __host__cudaError_t cudaDeviceSetGraphMemAttribute (int device, cudaGraphMemAttributeType attr, void *value)

Set asynchronous allocation attributes related to graphs.

##### Parameters

**device**

  - Specifies the scope of the query
**attr**

  - attribute to get
**value**

  - pointer to value to set

##### Returns

cudaSuccess, cudaErrorInvalidDevice

##### Description

Valid attributes are:

cudaGraphMemAttrUsedMemHigh: High watermark of memory, in bytes, associated with graphs

##### **‣**

since the last time it was reset. High watermark can only be reset to zero.
cudaGraphMemAttrReservedMemHigh: High watermark of memory, in bytes, currently allocated

##### **‣**

for use by the CUDA graphs asynchronous allocator.











CUDA Runtime API vRelease Version  |  322


Modules


See also:

cudaDeviceGetGraphMemAttribute, cudaGraphAddMemAllocNode, cudaGraphAddMemFreeNode,
cudaDeviceGraphMemTrim, cudaMallocAsync, cudaFreeAsync