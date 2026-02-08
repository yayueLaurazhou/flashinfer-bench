# __host__cudaError_t cudaStreamSetAttribute (cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue *value)

Sets stream attribute.

##### Parameters

**hStream**
**attr**
**value**

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Sets attribute attr on hStream from corresponding attribute of value. The updated attribute will
be applied to subsequent work submitted to the stream. It will not affect previously submitted work.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaAccessPolicyWindow