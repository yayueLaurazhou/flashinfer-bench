# CUresult cuDeviceSetGraphMemAttribute (CUdevice device, CUgraphMem_attribute attr, void *value)

Set asynchronous allocation attributes related to graphs.

###### Parameters

**device**

  - Specifies the scope of the query
**attr**

  - attribute to get
**value**

  - pointer to value to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_DEVICE

###### Description

Valid attributes are:

CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: High watermark of memory, in bytes,

###### **‣**

associated with graphs since the last time it was reset. High watermark can only be reset to zero.


CUDA Driver API TRM-06703-001 _vRelease Version  |  415


Modules


CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: High watermark of memory, in bytes,

###### **‣**

currently allocated for use by the CUDA graphs asynchronous allocator.


See also:

cuDeviceGetGraphMemAttribute, cuGraphAddMemAllocNode, cuGraphAddMemFreeNode