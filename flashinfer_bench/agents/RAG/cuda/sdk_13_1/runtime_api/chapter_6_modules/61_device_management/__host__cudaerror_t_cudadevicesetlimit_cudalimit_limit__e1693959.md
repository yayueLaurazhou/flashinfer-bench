# __host__cudaError_t cudaDeviceSetLimit (cudaLimit limit, size_t value)

Set resource limits.

##### Parameters

**limit**

  - Limit to set
**value**

  - Size of limit

##### Returns

cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Setting limit to value is a request by the application to update the current limit maintained by
the device. The driver is free to modify the requested value to meet h/w requirements (this could be
clamping to minimum or maximum values, rounding up to nearest element size, etc). The application
can use cudaDeviceGetLimit() to find out exactly what the limit has been set to.

Setting each cudaLimit has its own specific restrictions, so each is discussed here.

cudaLimitStackSize controls the stack size in bytes of each GPU thread.

##### **‣**

cudaLimitPrintfFifoSize controls the size in bytes of the shared FIFO used by the printf() device

##### **‣**

system call. Setting cudaLimitPrintfFifoSize must not be performed after launching any kernel that
uses the printf() device system call - in such case cudaErrorInvalidValue will be returned.

cudaLimitMallocHeapSize controls the size in bytes of the heap used by the malloc() and free()

##### **‣**

device system calls. Setting cudaLimitMallocHeapSize must not be performed after launching any
kernel that uses the malloc() or free() device system calls - in such case cudaErrorInvalidValue will
be returned.

cudaLimitDevRuntimeSyncDepth controls the maximum nesting depth of a grid at which a

##### **‣**

thread can safely call cudaDeviceSynchronize(). Setting this limit must be performed before any


CUDA Runtime API vRelease Version  |  27


Modules


launch of a kernel that uses the device runtime and calls cudaDeviceSynchronize() above the
default sync depth, two levels of grids. Calls to cudaDeviceSynchronize() will fail with error
code cudaErrorSyncDepthExceeded if the limitation is violated. This limit can be set smaller than
the default or up the maximum launch depth of 24. When setting this limit, keep in mind that
additional levels of sync depth require the runtime to reserve large amounts of device memory
which can no longer be used for user allocations. If these reservations of device memory fail,
cudaDeviceSetLimit will return cudaErrorMemoryAllocation, and the limit can be reset to a lower
value. This limit is only applicable to devices of compute capability < 9.0. Attempting to set this
limit on devices of other compute capability will results in error cudaErrorUnsupportedLimit being
returned.

cudaLimitDevRuntimePendingLaunchCount controls the maximum number of outstanding device

##### **‣**

runtime launches that can be made from the current device. A grid is outstanding from the point of
launch up until the grid is known to have been completed. Device runtime launches which violate
this limitation fail and return cudaErrorLaunchPendingCountExceeded when cudaGetLastError()
is called after launch. If more pending launches than the default (2048 launches) are needed
for a module using the device runtime, this limit can be increased. Keep in mind that being
able to sustain additional pending launches will require the runtime to reserve larger amounts
of device memory upfront which can no longer be used for allocations. If these reservations
fail, cudaDeviceSetLimit will return cudaErrorMemoryAllocation, and the limit can be reset
to a lower value. This limit is only applicable to devices of compute capability 3.5 and higher.
Attempting to set this limit on devices of compute capability less than 3.5 will result in the error
cudaErrorUnsupportedLimit being returned.

cudaLimitMaxL2FetchGranularity controls the L2 cache fetch granularity. Values can range from

##### **‣**

0B to 128B. This is purely a performance hint and it can be ignored or clamped depending on the
platform.

cudaLimitPersistingL2CacheSize controls size in bytes available for persisting L2 cache. This is

##### **‣**

purely a performance hint and it can be ignored or clamped depending on the platform.



See also:

cudaDeviceGetLimit, cuCtxSetLimit


CUDA Runtime API vRelease Version  |  28


Modules