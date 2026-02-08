# __host____device__cudaError_t cudaDeviceSynchronize (void)

Wait for compute device to finish.

##### Returns

cudaSuccess, cudaErrorStreamCaptureUnsupported

##### Description

Blocks until the device has completed all preceding requested tasks. cudaDeviceSynchronize() returns
an error if one of the preceding tasks has failed. If the cudaDeviceScheduleBlockingSync flag was set
for this device, the host thread will block until the device has finished its work.


CUDA Runtime API vRelease Version  |  29


Modules







See also:

cudaDeviceReset, cuCtxSynchronize