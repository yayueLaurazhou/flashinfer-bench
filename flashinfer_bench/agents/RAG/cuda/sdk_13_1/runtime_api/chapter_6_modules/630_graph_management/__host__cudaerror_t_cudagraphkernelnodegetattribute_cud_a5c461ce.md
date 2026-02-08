# __host__cudaError_t cudaGraphKernelNodeGetAttribute (cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue *value_out)

Queries node attribute.

##### Parameters

**hNode**
**attr**
**value_out**

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Queries attribute attr from node hNode and stores it in corresponding member of value_out.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaAccessPolicyWindow