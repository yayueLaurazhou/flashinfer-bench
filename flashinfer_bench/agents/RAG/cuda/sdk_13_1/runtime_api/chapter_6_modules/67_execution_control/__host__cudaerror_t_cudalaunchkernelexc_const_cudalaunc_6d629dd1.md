# __host__cudaError_t cudaLaunchKernelExC (const cudaLaunchConfig_t *config, const void *func, void **args)

Launches a CUDA function with launch-time configuration.

##### Parameters

**config**

  - Launch configuration
**func**

  - Kernel to launch
**args**

  - Array of pointers to kernel parameters

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration,
cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources,
cudaErrorSharedObjectInitFailed, cudaErrorInvalidPtx, cudaErrorUnsupportedPtxVersion,
cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound,
cudaErrorJitCompilationDisabled

##### Description

Note that the functionally equivalent variadic template cudaLaunchKernelEx is available for C++11
and newer.

Invokes the kernel func on config->gridDim (config->gridDim.x config>gridDim.y config->gridDim.z) grid of blocks. Each block contains config->blockDim
(config->blockDim.x config->blockDim.y config->blockDim.z) threads.

config->dynamicSmemBytes sets the amount of dynamic shared memory that will be available
to each thread block.

config->stream specifies a stream the invocation is associated to.

Configuration beyond grid and block dimensions, dynamic shared memory size, and stream can be
provided with the following two fields of config:

config->attrs is an array of config->numAttrs contiguous cudaLaunchAttribute elements.
The value of this pointer is not considered if config->numAttrs is zero. However, in that case,
it is recommended to set the pointer to NULL. config->numAttrs is the number of attributes
populating the first config->numAttrs positions of the config->attrs array.

If the kernel has N parameters the args should point to array of N pointers. Each pointer, from
args[0] to args[N - 1], point to the region of memory from which the actual parameter will be
copied.


CUDA Runtime API vRelease Version  |  108


Modules



N.B. This function is so named to avoid unintentionally invoking the templated version,
cudaLaunchKernelEx, for kernels taking a single void** or void* parameter.











See also:

cudaLaunchKernelEx(const cudaLaunchConfig_t *config, void (*kernel)(ExpTypes...), ActTypes
&&... args) "cudaLaunchKernelEx (C++ API)", cuLaunchKernelEx