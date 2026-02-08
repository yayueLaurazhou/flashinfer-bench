# CUresult cuLibraryLoadData (CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions)

Load a library with specified code and options.

###### Parameters

**library**

  - Returned library
**code**

  - Code to load
**jitOptions**

  - Options for JIT
**jitOptionsValues**

  - Option values for JIT
**numJitOptions**

  - Number of options
**libraryOptions**

  - Options for loading
**libraryOptionValues**

  - Option values for loading
**numLibraryOptions**

  - Number of options for loading


CUDA Driver API TRM-06703-001 _vRelease Version  |  171


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_PTX,
CUDA_ERROR_UNSUPPORTED_PTX_VERSION, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_NO_BINARY_FOR_GPU,
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
CUDA_ERROR_JIT_COMPILER_NOT_FOUND, CUDA_ERROR_NOT_SUPPORTED

###### Description

Takes a pointer code and loads the corresponding library library based on the application defined
library loading mode:

If module loading is set to EAGER, via the environment variables described in "Module loading",

###### **‣**

library is loaded eagerly into all contexts at the time of the call and future contexts at the time
of creation until the library is unloaded with cuLibraryUnload().
If the environment variables are set to LAZY, is not immediately loaded onto all

###### ‣ library

existent contexts and will only be loaded when a function is needed for that context, such as a
kernel launch.

These environment variables are described in the CUDA programming guide under the "CUDA
environment variables" section.

The code may be a cubin or fatbin as output by nvcc, or a NULL-terminated PTX, either as output
by nvcc or hand-written, or Tile IR data. A fatbin should also contain relocatable code when doing
separate compilation.

Options are passed as an array via jitOptions and any corresponding parameters are passed in
jitOptionsValues. The number of total JIT options is supplied via numJitOptions. Any
outputs will be returned via jitOptionsValues.

Library load options are passed as an array via libraryOptions and any corresponding parameters
are passed in libraryOptionValues. The number of total library load options is supplied via
numLibraryOptions.





See also:

cuLibraryLoadFromFile, cuLibraryUnload, cuModuleLoad, cuModuleLoadData,
cuModuleLoadDataEx


CUDA Driver API TRM-06703-001 _vRelease Version  |  172


Modules