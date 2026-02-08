# __host__cudaError_t cudaStreamCopyAttributes (cudaStream_t dst, cudaStream_t src)

Copies attributes from source stream to destination stream.

##### Parameters

**dst**
Destination stream
**src**
Source stream For attributes see cudaStreamAttrID

##### Returns

cudaSuccess, cudaErrorNotSupported

##### Description

Copies attributes from source stream src to destination stream dst. Both streams must have the same
context.


CUDA Runtime API vRelease Version  |  55


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaAccessPolicyWindow