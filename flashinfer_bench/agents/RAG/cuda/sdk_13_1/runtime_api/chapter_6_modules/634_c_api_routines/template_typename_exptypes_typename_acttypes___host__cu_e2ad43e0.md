# template < typename... ExpTypes, typename... ActTypes > __host__cudaError_t cudaLaunchKernelEx (const cudaLaunchConfig_t *config, void(*)(ExpTypes...) kernel, ActTypes &&... args)

Launches a CUDA function with launch-time configuration.

##### Parameters

**config**

  - Launch configuration
**kernel**

  - Kernel to launch


CUDA Runtime API vRelease Version  |  485


Modules


**args**

  - Parameter pack of kernel parameters

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration,
cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources,
cudaErrorSharedObjectInitFailed, cudaErrorInvalidPtx, cudaErrorUnsupportedPtxVersion,
cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound,
cudaErrorJitCompilationDisabled

##### Description

Invokes the kernel kernel on config->gridDim (config->gridDim.x config>gridDim.y config->gridDim.z) grid of blocks. Each block contains config->blockDim
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

The kernel arguments should be passed as arguments to this function via the args parameter pack.

The C API version of this function, cudaLaunchKernelExC, is also available for pre-C++11
compilers and for use cases where the ability to pass kernel parameters via void* array is preferable.













CUDA Runtime API vRelease Version  |  486


Modules


See also:

cudaLaunchKernelEx ( C API), cuLaunchKernelEx