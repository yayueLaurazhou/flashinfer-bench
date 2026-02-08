# CUresult cuDeviceGetGraphMemAttribute (CUdevice device, CUgraphMem_attribute attr, void *value)

Query asynchronous allocation attributes related to graphs.

###### Parameters

**device**

  - Specifies the scope of the query
**attr**

  - attribute to get
**value**

  - retrieved value

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_DEVICE

###### Description

Valid attributes are:

CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT: Amount of memory, in bytes, currently

###### **‣**

associated with graphs
CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: High watermark of memory, in bytes,

###### **‣**

associated with graphs since the last time it was reset. High watermark can only be reset to zero.
CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT: Amount of memory, in bytes,

###### **‣**

currently allocated for use by the CUDA graphs asynchronous allocator.
CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: High watermark of memory, in bytes,

###### **‣**

currently allocated for use by the CUDA graphs asynchronous allocator.


See also:

cuDeviceSetGraphMemAttribute, cuGraphAddMemAllocNode, cuGraphAddMemFreeNode


CUDA Driver API TRM-06703-001 _vRelease Version  |  414


Modules