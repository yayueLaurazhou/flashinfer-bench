# __host__cudaError_t cudaGraphKernelNodeCopyAttributes (cudaGraphNode_t hDst, cudaGraphNode_t hSrc)

Copies attributes from source node to destination node.

##### Parameters

**hDst**
Destination node
**hSrc**
Source node For list of attributes see cudaKernelNodeAttrID

##### Returns

cudaSuccess, cudaErrorInvalidContext

##### Description

Copies attributes from source node hSrc to destination node hDst. Both node must have the same
context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  395


Modules


See also:

cudaAccessPolicyWindow