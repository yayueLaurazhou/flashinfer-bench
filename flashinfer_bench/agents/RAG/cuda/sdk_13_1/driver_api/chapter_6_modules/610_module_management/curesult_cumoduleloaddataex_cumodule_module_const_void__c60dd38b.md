# CUresult cuModuleLoadDataEx (CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)

Load a module's data with options.

###### Parameters

**module**

  - Returned module
**image**

  - Module data to load
**numOptions**

  - Number of options
**options**

  - Options for JIT
**optionValues**

  - Option values for JIT

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_PTX, CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU,
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
CUDA_ERROR_JIT_COMPILER_NOT_FOUND


CUDA Driver API TRM-06703-001 _vRelease Version  |  154


Modules

###### Description

Takes a pointer image and loads the corresponding module module into the current context. The
image may be a cubin or fatbin as output by nvcc, or a NULL-terminated PTX, either as output by
nvcc or hand-written, or Tile IR data.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuModuleGetFunction, cuModuleGetGlobal, cuModuleGetTexRef, cuModuleLoad,
cuModuleLoadData, cuModuleLoadFatBinary, cuModuleUnload