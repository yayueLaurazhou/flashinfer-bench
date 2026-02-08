# __host__cudaError_t cudaFuncGetParamInfo (const void *func, size_t paramIndex, size_t *paramOffset, size_t *paramSize)

Returns the offset and size of a kernel parameter in the device-side parameter layout.

##### Parameters

**func**

  - The function to query
**paramIndex**

  - The parameter index to query
**paramOffset**

  - The offset into the device-side parameter layout at which the parameter resides
**paramSize**

  - The size of the parameter in the device-side parameter layout

##### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

##### Description

Queries the kernel parameter at paramIndex in func's list of parameters and returns parameter
information via paramOffset and paramSize. paramOffset returns the offset of the


CUDA Runtime API vRelease Version  |  97


Modules


parameter in the device-side parameter layout. paramSize returns the size in bytes of the
parameter. This information can be used to update kernel node parameters from the device via
cudaGraphKernelNodeSetParam() and cudaGraphKernelNodeUpdatesApply(). paramIndex must be
less than the number of parameters that func takes.