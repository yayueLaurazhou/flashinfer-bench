# __host__cudaError_t cudaInitDevice (int device, unsigned int deviceFlags, unsigned int flags)

Initialize device to be used for GPU executions.

##### Parameters

**device**

  - Device on which the runtime will initialize itself.
**deviceFlags**

  - Parameters for device operation.
**flags**

  - Flags for controlling the device initialization.

##### Returns

cudaSuccess, cudaErrorInvalidDevice,

##### Description

This function will initialize the CUDA Runtime structures and primary context on device when
called, but the context will not be made current to device.

When cudaInitDeviceFlagsAreValid is set in flags, deviceFlags are applied to the requested device.
The values of deviceFlags match those of the flags parameters in cudaSetDeviceFlags. The effect may
be verified by cudaGetDeviceFlags.

This function will return an error if the device is in cudaComputeModeExclusiveProcess and is
occupied by another process or if the device is in cudaComputeModeProhibited.





CUDA Runtime API vRelease Version  |  34


Modules


See also:

cudaGetDeviceCount, cudaGetDevice, cudaGetDeviceProperties, cudaChooseDevice, cudaSetDevice
cuCtxSetCurrent