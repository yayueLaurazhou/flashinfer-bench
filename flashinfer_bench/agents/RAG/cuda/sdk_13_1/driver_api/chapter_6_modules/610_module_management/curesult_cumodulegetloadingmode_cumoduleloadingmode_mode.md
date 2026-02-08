# CUresult cuModuleGetLoadingMode (CUmoduleLoadingMode *mode)

Query lazy loading mode.

###### Parameters

**mode**

  - Returns the lazy loading mode

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Returns lazy loading mode Module loading mode is controlled by CUDA_MODULE_LOADING env
variable


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuModuleLoad,