# __host__cudaError_t cudaStreamGetAttribute (cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue *value_out)

Queries stream attribute.

##### Parameters

**hStream**
**attr**


CUDA Runtime API vRelease Version  |  60


Modules


**value_out**

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Queries attribute attr from hStream and stores it in corresponding member of value_out.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaAccessPolicyWindow