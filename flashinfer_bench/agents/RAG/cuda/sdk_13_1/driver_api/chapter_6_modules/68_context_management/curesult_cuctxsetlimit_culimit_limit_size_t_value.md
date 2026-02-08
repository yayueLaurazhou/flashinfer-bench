# CUresult cuCtxSetLimit (CUlimit limit, size_t value)

Set resource limits.

###### Parameters

**limit**

  - Limit to set
**value**

  - Size of limit

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNSUPPORTED_LIMIT, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_INVALID_CONTEXT


CUDA Driver API TRM-06703-001 _vRelease Version  |  136


Modules

###### Description

Setting limit to value is a request by the application to update the current limit maintained by
the context. The driver is free to modify the requested value to meet h/w requirements (this could be
clamping to minimum or maximum values, rounding up to nearest element size, etc). The application
can use cuCtxGetLimit() to find out exactly what the limit has been set to.

Setting each CUlimit has its own specific restrictions, so each is discussed here.

CU_LIMIT_STACK_SIZE controls the stack size in bytes of each GPU thread. The driver

###### **‣**

automatically increases the per-thread stack size for each kernel launch as needed. This size isn't
reset back to the original value after each launch. Setting this value will take effect immediately,
and if necessary, the device will block until all preceding requested tasks are complete.

CU_LIMIT_PRINTF_FIFO_SIZE controls the size in bytes of the FIFO used by the printf() device

###### **‣**

system call. Setting CU_LIMIT_PRINTF_FIFO_SIZE must be performed before launching any
kernel that uses the printf() device system call, otherwise CUDA_ERROR_INVALID_VALUE
will be returned.

CU_LIMIT_MALLOC_HEAP_SIZE controls the size in bytes of the heap used by the malloc()

###### **‣**

and free() device system calls. Setting CU_LIMIT_MALLOC_HEAP_SIZE must be performed
before launching any kernel that uses the malloc() or free() device system calls, otherwise
CUDA_ERROR_INVALID_VALUE will be returned.

CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH controls the maximum nesting depth of a grid

###### **‣**

at which a thread can safely call cudaDeviceSynchronize(). Setting this limit must be performed
before any launch of a kernel that uses the device runtime and calls cudaDeviceSynchronize()
above the default sync depth, two levels of grids. Calls to cudaDeviceSynchronize() will fail
with error code cudaErrorSyncDepthExceeded if the limitation is violated. This limit can be
set smaller than the default or up the maximum launch depth of 24. When setting this limit,
keep in mind that additional levels of sync depth require the driver to reserve large amounts of
device memory which can no longer be used for user allocations. If these reservations of device
memory fail, cuCtxSetLimit() will return CUDA_ERROR_OUT_OF_MEMORY, and the limit
can be reset to a lower value. This limit is only applicable to devices of compute capability < 9.0.
Attempting to set this limit on devices of other compute capability versions will result in the error
CUDA_ERROR_UNSUPPORTED_LIMIT being returned.

CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT controls the maximum

###### **‣**

number of outstanding device runtime launches that can be made from the current
context. A grid is outstanding from the point of launch up until the grid is known to have
been completed. Device runtime launches which violate this limitation fail and return
cudaErrorLaunchPendingCountExceeded when cudaGetLastError() is called after launch. If
more pending launches than the default (2048 launches) are needed for a module using the
device runtime, this limit can be increased. Keep in mind that being able to sustain additional
pending launches will require the driver to reserve larger amounts of device memory upfront
which can no longer be used for allocations. If these reservations fail, cuCtxSetLimit() will


CUDA Driver API TRM-06703-001 _vRelease Version  |  137


Modules


return CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
This limit is only applicable to devices of compute capability 3.5 and higher. Attempting
to set this limit on devices of compute capability less than 3.5 will result in the error
CUDA_ERROR_UNSUPPORTED_LIMIT being returned.

CU_LIMIT_MAX_L2_FETCH_GRANULARITY controls the L2 cache fetch granularity. Values

###### **‣**

can range from 0B to 128B. This is purely a performance hint and it can be ignored or clamped
depending on the platform.

CU_LIMIT_PERSISTING_L2_CACHE_SIZE controls size in bytes available for persisting

###### **‣**

L2 cache. This is purely a performance hint and it can be ignored or clamped depending on the
platform.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig,
cuCtxSynchronize, cudaDeviceSetLimit