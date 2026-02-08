# CUresult cuGraphKernelNodeCopyAttributes (CUgraphNode dst, CUgraphNode src)

Copies attributes from source node to destination node.

###### Parameters

**dst**
Destination node
**src**
Source node For list of attributes see CUkernelNodeAttrID

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Copies attributes from source node src to destination node dst. Both node must have the same
context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

CUaccessPolicyWindow