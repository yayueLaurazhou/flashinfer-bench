# CUresult cuLaunchCooperativeKernelMultiDevice (CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags)

Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as
they execute.

###### Parameters

**launchParamsList**

  - List of launch parameters, one per device
**numDevices**

  - Size of the launchParamsList array
**flags**

  - Flags to control launch behavior

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
CUDA_ERROR_LAUNCH_TIMEOUT,
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

###### Description

Deprecated This function is deprecated as of CUDA 11.3.

Invokes kernels as specified in the launchParamsList array where each element of the array
specifies all the parameters required to perform a single kernel launch. These kernels can cooperate and
synchronize as they execute. The size of the array is specified by numDevices.

No two kernels can be launched on the same device. All the devices targeted by this multidevice launch must be identical. All devices must have a non-zero value for the device attribute
CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH.

All kernels launched must be identical with respect to the compiled code. Note that any __device__,
__constant__ or __managed__ variables present in the module that owns the kernel launched on each


CUDA Driver API TRM-06703-001 _vRelease Version  |  392


Modules


device, are independently instantiated on every device. It is the application's responsibility to ensure
these variables are initialized and used appropriately.

The size of the grids as specified in blocks, the size of the blocks themselves and the amount of shared
memory used by each thread block must also match across all launched kernels.

The streams used to launch these kernels must have been created via either cuStreamCreate
or cuStreamCreateWithPriority. The NULL stream or CU_STREAM_LEGACY or
CU_STREAM_PER_THREAD cannot be used.

The total number of blocks launched per kernel cannot exceed the maximum number of blocks
per multiprocessor as returned by cuOccupancyMaxActiveBlocksPerMultiprocessor (or
cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
as specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
Since the total number of blocks launched per device has to match across all devices, the maximum
number of blocks that can be launched per device will be limited by the device with the least number of
multiprocessors.

The kernels cannot make use of CUDA dynamic parallelism.

The CUDA_LAUNCH_PARAMS structure is defined as:


CUDA_LAUNCH_PARAMS::function specifies the kernel to be launched. All functions

###### **‣**

must be identical with respect to the compiled code. Note that you can also specify contextless kernel CUkernel by querying the handle using cuLibraryGetKernel() and then casting to
CUfunction. In this case, the context to launch the kernel on be taken from the specified stream
CUDA_LAUNCH_PARAMS::hStream.
CUDA_LAUNCH_PARAMS::gridDimX is the width of the grid in blocks. This must match across

###### **‣**

all kernels launched.
CUDA_LAUNCH_PARAMS::gridDimY is the height of the grid in blocks. This must match

###### **‣**

across all kernels launched.
CUDA_LAUNCH_PARAMS::gridDimZ is the depth of the grid in blocks. This must match across

###### **‣**

all kernels launched.
CUDA_LAUNCH_PARAMS::blockDimX is the X dimension of each thread block. This must

###### **‣**

match across all kernels launched.


CUDA Driver API TRM-06703-001 _vRelease Version  |  393


Modules


CUDA_LAUNCH_PARAMS::blockDimX is the Y dimension of each thread block. This must

###### **‣**

match across all kernels launched.
CUDA_LAUNCH_PARAMS::blockDimZ is the Z dimension of each thread block. This must

###### **‣**

match across all kernels launched.
CUDA_LAUNCH_PARAMS::sharedMemBytes is the dynamic shared-memory size per thread

###### **‣**

block in bytes. This must match across all kernels launched.
CUDA_LAUNCH_PARAMS::hStream is the handle to the stream to perform the launch in. This

###### **‣**

cannot be the NULL stream or CU_STREAM_LEGACY or CU_STREAM_PER_THREAD.
The CUDA context associated with this stream must match that associated with
CUDA_LAUNCH_PARAMS::function.
CUDA_LAUNCH_PARAMS::kernelParams is an array of pointers to kernel

###### **‣**

parameters. If CUDA_LAUNCH_PARAMS::function has N parameters,
then CUDA_LAUNCH_PARAMS::kernelParams needs to be an array of N
pointers. Each of CUDA_LAUNCH_PARAMS::kernelParams[0] through
CUDA_LAUNCH_PARAMS::kernelParams[N-1] must point to a region of memory from which
the actual kernel parameter will be copied. The number of kernel parameters and their offsets and
sizes do not need to be specified as that information is retrieved directly from the kernel's image.

By default, the kernel won't begin execution on any GPU until all prior work in all the
specified streams has completed. This behavior can be overridden by specifying the flag
CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC. When this
flag is specified, each kernel will only wait for prior work in the stream corresponding to that GPU to
complete before it begins execution.

Similarly, by default, any subsequent work pushed in any of the specified streams will not begin
execution until the kernels on all GPUs have completed. This behavior can be overridden by specifying
the flag CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC.
When this flag is specified, any subsequent work pushed in any of the specified streams will only
wait for the kernel launched on the GPU corresponding to that stream to complete before it begins
execution.

Calling cuLaunchCooperativeKernelMultiDevice() sets persistent function state that is the same
as function state set through cuLaunchKernel API when called individually for each element in
launchParamsList.

When kernels are launched via cuLaunchCooperativeKernelMultiDevice(), the previous block shape,
shared size and parameter info associated with each CUDA_LAUNCH_PARAMS::function in
launchParamsList is overwritten.

Note that to use cuLaunchCooperativeKernelMultiDevice(), the kernels must either have
been compiled with toolchain version 3.2 or later so that it will contain kernel parameter
information, or have no kernel parameters. If either of these conditions is not met, then
cuLaunchCooperativeKernelMultiDevice() will return CUDA_ERROR_INVALID_IMAGE.


CUDA Driver API TRM-06703-001 _vRelease Version  |  394


Modules





See also:

cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuFuncSetCacheConfig, cuFuncGetAttribute,
cuLaunchCooperativeKernel, cudaLaunchCooperativeKernelMultiDevice