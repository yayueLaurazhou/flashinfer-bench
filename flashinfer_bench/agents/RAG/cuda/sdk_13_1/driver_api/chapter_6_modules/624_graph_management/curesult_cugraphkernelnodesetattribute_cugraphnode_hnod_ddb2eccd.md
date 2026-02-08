# CUresult cuGraphKernelNodeSetAttribute (CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue *value)

Sets node attribute.

###### Parameters

**hNode**
**attr**
**value**

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE

###### Description

Sets attribute attr on node hNode from corresponding attribute of value.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

CUaccessPolicyWindow