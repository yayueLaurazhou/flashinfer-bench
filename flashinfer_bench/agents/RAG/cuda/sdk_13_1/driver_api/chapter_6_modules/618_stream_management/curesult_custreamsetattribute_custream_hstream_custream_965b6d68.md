# CUresult cuStreamSetAttribute (CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value)

Sets stream attribute.

###### Parameters

**hStream**
**attr**
**value**

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE

###### Description

Sets attribute attr on hStream from corresponding attribute of value. The updated attribute will
be applied to subsequent work submitted to the stream. It will not affect previously submitted work.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

CUaccessPolicyWindow


CUDA Driver API TRM-06703-001 _vRelease Version  |  350


Modules