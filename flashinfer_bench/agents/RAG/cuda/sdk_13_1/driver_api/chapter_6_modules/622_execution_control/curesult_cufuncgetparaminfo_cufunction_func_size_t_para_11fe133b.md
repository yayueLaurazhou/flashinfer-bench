# CUresult cuFuncGetParamInfo (CUfunction func, size_t paramIndex, size_t *paramOffset, size_t *paramSize)

Returns the offset and size of a kernel parameter in the device-side parameter layout.

###### Parameters

**func**

  - The function to query
**paramIndex**

  - The parameter index to query
**paramOffset**

  - Returns the offset into the device-side parameter layout at which the parameter resides
**paramSize**

  - Optionally returns the size of the parameter in the device-side parameter layout

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Queries the kernel parameter at paramIndex into func's list of parameters, and returns in
paramOffset and paramSize the offset and size, respectively, where the parameter will reside
in the device-side parameter layout. This information can be used to update kernel node parameters
from the device via cudaGraphKernelNodeSetParam() and cudaGraphKernelNodeUpdatesApply().


CUDA Driver API TRM-06703-001 _vRelease Version  |  385


Modules


paramIndex must be less than the number of parameters that func takes. paramSize can be set to
NULL if only the parameter offset is desired.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuKernelGetParamInfo