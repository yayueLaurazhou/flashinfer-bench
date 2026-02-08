# CUresult cuStreamCopyAttributes (CUstream dst, CUstream src)

Copies attributes from source stream to destination stream.

###### Parameters

**dst**
Destination stream
**src**
Source stream For list of attributes see CUstreamAttrID

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Copies attributes from source stream src to destination stream dst. Both streams must have the same
context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

CUaccessPolicyWindow