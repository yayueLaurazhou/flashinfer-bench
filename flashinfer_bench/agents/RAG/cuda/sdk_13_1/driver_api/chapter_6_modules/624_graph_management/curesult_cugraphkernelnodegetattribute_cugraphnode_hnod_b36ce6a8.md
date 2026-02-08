# CUresult cuGraphKernelNodeGetAttribute (CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue *value_out)

Queries node attribute.

###### Parameters

**hNode**
**attr**
**value_out**

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE


CUDA Driver API TRM-06703-001 _vRelease Version  |  475


Modules

###### Description

Queries attribute attr from node hNode and stores it in corresponding member of value_out.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

CUaccessPolicyWindow