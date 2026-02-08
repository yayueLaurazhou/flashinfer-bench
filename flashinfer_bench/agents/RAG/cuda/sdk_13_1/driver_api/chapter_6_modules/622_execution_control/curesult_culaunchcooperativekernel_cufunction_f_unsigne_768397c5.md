# CUresult cuLaunchCooperativeKernel (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)

Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can
cooperate and synchronize as they execute.

###### Parameters

**f**

  - Function CUfunction or Kernel CUkernel to launch
**gridDimX**

  - Width of grid in blocks
**gridDimY**

  - Height of grid in blocks
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

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
CUDA_ERROR_LAUNCH_TIMEOUT,
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,


CUDA Driver API TRM-06703-001 _vRelease Version  |  390


Modules


CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, CUDA_ERROR_NOT_FOUND

###### Description

Invokes the function CUfunction or the kernel CUkernel f on a gridDimX x gridDimY x
gridDimZ grid of blocks. Each block contains blockDimX x blockDimY x blockDimZ threads.

sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread
block.

The device on which this kernel is invoked must have a non-zero value for the device attribute
CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.

The total number of blocks launched cannot exceed the maximum number of blocks
per multiprocessor as returned by cuOccupancyMaxActiveBlocksPerMultiprocessor (or
cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors as
specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.

The kernel cannot make use of CUDA dynamic parallelism.

Kernel parameters must be specified via kernelParams. If f has N parameters, then
kernelParams needs to be an array of N pointers. Each of kernelParams[0] through
kernelParams[N-1] must point to a region of memory from which the actual kernel parameter will
be copied. The number of kernel parameters and their offsets and sizes do not need to be specified as
that information is retrieved directly from the kernel's image.

Calling cuLaunchCooperativeKernel() sets persistent function state that is the same as function state set
through cuLaunchKernel API

When the kernel f is launched via cuLaunchCooperativeKernel(), the previous block shape, shared size
and parameter info associated with f is overwritten.

Note that to use cuLaunchCooperativeKernel(), the kernel f must either have been compiled with
toolchain version 3.2 or later so that it will contain kernel parameter information, or have no kernel
parameters. If either of these conditions is not met, then cuLaunchCooperativeKernel() will return
CUDA_ERROR_INVALID_IMAGE.

Note that the API can also be used to launch context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to
launch the kernel on will either be taken from the specified stream hStream or the current context in
case of NULL stream.





CUDA Driver API TRM-06703-001 _vRelease Version  |  391


Modules


See also:

cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuFuncSetCacheConfig, cuFuncGetAttribute,
cuLaunchCooperativeKernelMultiDevice, cudaLaunchCooperativeKernel, cuLibraryGetKernel,
cuKernelSetCacheConfig, cuKernelGetAttribute, cuKernelSetAttribute