# __host__cudaError_t cudaLibraryLoadData (cudaLibrary_t *library, const void *code, cudaJitOption *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions)

Load a library with specified code and options.

##### Parameters

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

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorInitializationError,
cudaErrorCudartUnloading, cudaErrorInvalidPtx, cudaErrorUnsupportedPtxVersion,
cudaErrorNoKernelImageForDevice, cudaErrorSharedObjectSymbolNotFound,
cudaErrorSharedObjectInitFailed, cudaErrorJitCompilerNotFound

##### Description

Takes a pointer code and loads the corresponding library library based on the application defined
library loading mode:


CUDA Runtime API vRelease Version  |  438


Modules


If module loading is set to EAGER, via the environment variables described in "Module loading",

##### **‣**

library is loaded eagerly into all contexts at the time of the call and future contexts at the time
of creation until the library is unloaded with cudaLibraryUnload().
If the environment variables are set to LAZY, is not immediately loaded onto all

##### ‣ library

existent contexts and will only be loaded when a function is needed for that context, such as a
kernel launch.

These environment variables are described in the CUDA programming guide under the "CUDA
environment variables" section.

The code may be a cubin or fatbin as output by nvcc, or a NULL-terminated PTX, either as output
by nvcc or hand-written, or Tile IR data. A fatbin should also contain relocatable code when doing
[separate compilation. Please also see the documentation for nvrtc (https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/nvrtc/index.html)
[nvrtc/index.html), nvjitlink (https://docs.nvidia.com/cuda/nvjitlink/index.html), and nvfatbin (https://](https://docs.nvidia.com/cuda/nvrtc/index.html)
[docs.nvidia.com/cuda/nvfatbin/index.html) for more information on generating loadable code at](https://docs.nvidia.com/cuda/nvfatbin/index.html)
runtime.

Options are passed as an array via jitOptions and any corresponding parameters are passed in
jitOptionsValues. The number of total JIT options is supplied via numJitOptions. Any
outputs will be returned via jitOptionsValues.

Library load options are passed as an array via libraryOptions and any corresponding parameters
are passed in libraryOptionValues. The number of total library load options is supplied via
numLibraryOptions.


See also:

cudaLibraryLoadFromFile, cudaLibraryUnload, cuLibraryLoadData