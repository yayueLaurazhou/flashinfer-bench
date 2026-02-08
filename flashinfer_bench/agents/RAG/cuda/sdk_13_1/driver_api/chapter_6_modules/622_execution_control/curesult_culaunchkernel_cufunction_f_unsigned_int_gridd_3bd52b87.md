# CUresult cuLaunchKernel (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)

Launches a CUDA function CUfunction or a CUDA kernel CUkernel.

###### Parameters

**f**

  - Function CUfunction or Kernel CUkernel to launch
**gridDimX**

  - Width of grid in blocks
**gridDimY**

  - Height of grid in blocks


CUDA Driver API TRM-06703-001 _vRelease Version  |  396


Modules


**gridDimZ**

  - Depth of grid in blocks
**blockDimX**

  - X dimension of each thread block
**blockDimY**

  - Y dimension of each thread block
**blockDimZ**

  - Z dimension of each thread block
**sharedMemBytes**

  - Dynamic shared-memory size per thread block in bytes
**hStream**

  - Stream identifier
**kernelParams**

  - Array of pointers to kernel parameters
**extra**

  - Extra options

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
CUDA_ERROR_LAUNCH_TIMEOUT,
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, CUDA_ERROR_NOT_FOUND

###### Description

Invokes the function CUfunction or the kernel CUkernel f on a gridDimX x gridDimY x
gridDimZ grid of blocks. Each block contains blockDimX x blockDimY x blockDimZ threads.

sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread
block.

Kernel parameters to f can be specified in one of two ways:

1) Kernel parameters can be specified via kernelParams. If f has N parameters, then
kernelParams needs to be an array of N pointers. Each of kernelParams[0] through
kernelParams[N-1] must point to a region of memory from which the actual kernel parameter will
be copied. The number of kernel parameters and their offsets and sizes do not need to be specified as
that information is retrieved directly from the kernel's image.

2) Kernel parameters can also be packaged by the application into a single buffer that is passed in via
the extra parameter. This places the burden on the application of knowing each kernel parameter's


CUDA Driver API TRM-06703-001 _vRelease Version  |  397


Modules


size and alignment/padding within the buffer. Here is an example of using the extra parameter in this
manner:


The extra parameter exists to allow cuLaunchKernel to take additional less commonly used
arguments. extra specifies a list of names of extra settings and their corresponding values. Each extra
setting name is immediately followed by the corresponding value. The list must be terminated with
either NULL or CU_LAUNCH_PARAM_END.

CU_LAUNCH_PARAM_END, which indicates the end of the array;

###### ‣ extra

CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next value in will

###### ‣ extra

be a pointer to a buffer containing all the kernel parameters for launching kernel f;
CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next value in

###### **‣**

extra will be a pointer to a size_t containing the size of the buffer specified with
CU_LAUNCH_PARAM_BUFFER_POINTER;

The error CUDA_ERROR_INVALID_VALUE will be returned if kernel parameters are specified with
both kernelParams and extra (i.e. both kernelParams and extra are non-NULL).

Calling cuLaunchKernel() invalidates the persistent function state set through the following
deprecated APIs: cuFuncSetBlockShape(), cuFuncSetSharedSize(), cuParamSetSize(), cuParamSeti(),
cuParamSetf(), cuParamSetv().

Note that to use cuLaunchKernel(), the kernel f must either have been compiled with
toolchain version 3.2 or later so that it will contain kernel parameter information, or have no
kernel parameters. If either of these conditions is not met, then cuLaunchKernel() will return
CUDA_ERROR_INVALID_IMAGE.

Note that the API can also be used to launch context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to
launch the kernel on will either be taken from the specified stream hStream or the current context in
case of NULL stream.





See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  398


Modules


cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuFuncSetCacheConfig, cuFuncGetAttribute,
cudaLaunchKernel, cuLibraryGetKernel, cuKernelSetCacheConfig, cuKernelGetAttribute,
cuKernelSetAttribute