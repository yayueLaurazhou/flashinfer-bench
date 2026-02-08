# __host__cudaError_t cudaGraphKernelNodeSetAttribute (cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue *value)

Sets node attribute.

##### Parameters

**hNode**
**attr**
**value**

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle


CUDA Runtime API vRelease Version  |  397


Modules

##### Description

Sets attribute attr on node hNode from corresponding attribute of value.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaAccessPolicyWindow