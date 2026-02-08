# __host____device__cudaError_t cudaGetLastError (void)

Returns the last error from a runtime call.

##### Returns

cudaSuccess, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation,
cudaErrorInitializationError, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration,
cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidPitchValue,
cudaErrorInvalidSymbol, cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidDevicePointer,
cudaErrorInvalidTexture, cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor,
cudaErrorInvalidMemcpyDirection, cudaErrorInvalidFilterSetting, cudaErrorInvalidNormSetting,
cudaErrorUnknown, cudaErrorInvalidResourceHandle, cudaErrorInsufficientDriver,
cudaErrorNoDevice, cudaErrorSetOnActiveProcess, cudaErrorStartupFailure,
cudaErrorInvalidPtx, cudaErrorUnsupportedPtxVersion, cudaErrorNoKernelImageForDevice,
cudaErrorJitCompilerNotFound, cudaErrorJitCompilationDisabled

##### Description

Returns the last error that has been produced by any of the runtime calls in the same instance of the
CUDA Runtime library in the host thread and resets it to cudaSuccess.

Note: Multiple instances of the CUDA Runtime library can be present in an application when using a
library that statically links the CUDA Runtime.



See also:


CUDA Runtime API vRelease Version  |  47


Modules


cudaPeekAtLastError, cudaGetErrorName, cudaGetErrorString, cudaError