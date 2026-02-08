# __host__cudaError_t cudaHostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags)

Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by
cudaHostRegister.

##### Parameters

**pDevice**

  - Returned device pointer for mapped memory
**pHost**

  - Requested host pointer mapping
**flags**

  - Flags for extensions (must be 0 for now)

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Passes back the device pointer corresponding to the mapped, pinned host buffer allocated by
cudaHostAlloc() or registered by cudaHostRegister().


CUDA Runtime API vRelease Version  |  129


Modules


cudaHostGetDevicePointer() will fail if the cudaDeviceMapHost flag was not specified before deferred
context creation occurred, or if called on a device that does not support mapped, pinned memory.

For devices that have a non-zero value for the device attribute
cudaDevAttrCanUseHostPointerForRegisteredMem, the memory can also be accessed from the
device using the host pointer pHost. The device pointer returned by cudaHostGetDevicePointer()
may or may not match the original host pointer pHost and depends on the devices visible to the
application. If all devices visible to the application have a non-zero value for the device attribute, the
device pointer returned by cudaHostGetDevicePointer() will match the original pointer pHost. If any
device visible to the application has a zero value for the device attribute, the device pointer returned by
cudaHostGetDevicePointer() will not match the original host pointer pHost, but it will be suitable for
use on all devices provided Unified Virtual Addressing is enabled. In such systems, it is valid to access
the memory using either pointer on devices that have a non-zero value for the device attribute. Note
however that such devices should access the memory using only of the two pointers and not both.

flags provides for future releases. For now, it must be set to 0.



See also:

cudaSetDeviceFlags, cudaHostAlloc, cuMemHostGetDevicePointer