# __host__cudaError_t cudaSetDeviceFlags (unsigned int flags)

Sets flags to be used for device executions.

##### Parameters

**flags**

  - Parameters for device operation

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Records flags as the flags for the current device. If the current device has been set and that device
has already been initialized, the previous flags are overwritten. If the current device has not been
initialized, it is initialized with the provided flags. If no device has been made current to the calling
thread, a default device is selected and initialized with the provided flags.

The three LSBs of the flags parameter can be used to control how the CPU thread interacts with the
OS scheduler when waiting for results from the device.

cudaDeviceScheduleAuto: The default value if the parameter is zero, uses a heuristic based

##### ‣ flags

on the number of active CUDA contexts in the process C and the number of logical processors in
the system P. If C > P, then CUDA will yield to other OS threads when waiting for the device,
otherwise CUDA will not yield while waiting for results and actively spin on the processor.
Additionally, on Tegra devices, cudaDeviceScheduleAuto uses a heuristic based on the power


CUDA Runtime API vRelease Version  |  41


Modules


profile of the platform and may choose cudaDeviceScheduleBlockingSync for low-powered
devices.
cudaDeviceScheduleSpin: Instruct CUDA to actively spin when waiting for results from the

##### **‣**

device. This can decrease latency when waiting for the device, but may lower the performance of
CPU threads if they are performing work in parallel with the CUDA thread.
cudaDeviceScheduleYield: Instruct CUDA to yield its thread when waiting for results from the

##### **‣**

device. This can increase latency when waiting for the device, but can increase the performance of
CPU threads performing work in parallel with the device.
cudaDeviceScheduleBlockingSync: Instruct CUDA to block the CPU thread on a synchronization

##### **‣**

primitive when waiting for the device to finish work.
cudaDeviceBlockingSync: Instruct CUDA to block the CPU thread on a synchronization primitive

##### **‣**

when waiting for the device to finish work.

Deprecated: This flag was deprecated as of CUDA 4.0 and replaced with
cudaDeviceScheduleBlockingSync.
cudaDeviceMapHost: This flag enables allocating pinned host memory that is accessible to the

##### **‣**

device. It is implicit for the runtime but may be absent if a context is created using the driver API.
If this flag is not set, cudaHostGetDevicePointer() will always return a failure code.
cudaDeviceLmemResizeToMax: Instruct CUDA to not reduce local memory after resizing local

##### **‣**

memory for a kernel. This can prevent thrashing by local memory allocations when launching
many kernels with high local memory usage at the cost of potentially increased memory usage.

Deprecated: This flag is deprecated and the behavior enabled by this flag is now the default and
cannot be disabled.
cudaDeviceSyncMemops: Ensures that synchronous memory operations initiated on this context

##### **‣**

will always synchronize. See further documentation in the section titled "API Synchronization
behavior" to learn more about cases when synchronous memory operations can exhibit
asynchronous behavior.



See also:

cudaGetDeviceFlags, cudaGetDeviceCount, cudaGetDevice, cudaGetDeviceProperties,
cudaSetDevice, cudaSetValidDevices, cudaInitDevice, cudaChooseDevice,
cuDevicePrimaryCtxSetFlags


CUDA Runtime API vRelease Version  |  42


Modules