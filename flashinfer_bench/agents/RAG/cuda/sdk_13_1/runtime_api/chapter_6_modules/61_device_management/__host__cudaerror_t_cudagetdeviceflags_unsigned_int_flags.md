# __host__cudaError_t cudaGetDeviceFlags (unsigned int *flags)

Gets the flags for the current device.

##### Parameters

**flags**

  - Pointer to store the device flags

##### Returns

cudaSuccess, cudaErrorInvalidDevice

##### Description

Returns in flags the flags for the current device. If there is a current device for the calling thread, the
flags for the device are returned. If there is no current device, the flags for the first device are returned,
which may be the default flags. Compare to the behavior of cudaSetDeviceFlags.

Typically, the flags returned should match the behavior that will be seen if the calling thread uses a
device after this call, without any change to the flags or current device inbetween by this or another
thread. Note that if the device is not initialized, it is possible for another thread to change the flags
for the current device before it is initialized. Additionally, when using exclusive mode, if this thread
has not requested a specific device, it may use a device other than the first device, contrary to the
assumption made by this function.


CUDA Runtime API vRelease Version  |  32


Modules


If a context has been created via the driver API and is current to the calling thread, the flags for that
context are always returned.

Flags returned by this function may specifically include cudaDeviceMapHost even though it is not
accepted by cudaSetDeviceFlags because it is implicit in runtime API flags. The reason for this is that
the current context may have been created via the driver API in which case the flag is not implicit and
may be unset.



See also:

cudaGetDevice, cudaGetDeviceProperties, cudaSetDevice, cudaSetDeviceFlags, cudaInitDevice,
cuCtxGetFlags, cuDevicePrimaryCtxGetState