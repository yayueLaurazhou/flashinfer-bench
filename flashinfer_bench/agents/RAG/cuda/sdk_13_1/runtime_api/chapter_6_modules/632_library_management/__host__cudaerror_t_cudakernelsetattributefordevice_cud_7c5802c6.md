# __host__cudaError_t cudaKernelSetAttributeForDevice (cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device)

Sets information about a kernel.

##### Parameters

**kernel**

  - Kernel to set attribute of
**attr**

  - Attribute requested
**value**

  - Value to set
**device**

  - Device to set attribute of

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue

##### Description

This call sets the value of a specified attribute attr on the kernel kernel for the requested device
device to an integer value specified by value. This function returns cudaSuccess if the new
value of the attribute could be successfully set. If the set fails, this call will return an error. Not all
attributes can have values set. Attempting to set a value on a read-only attribute will result in an error
(cudaErrorInvalidValue)


CUDA Runtime API vRelease Version  |  432


Modules


Note that attributes set using cudaFuncSetAttribute() will override the attribute set by this API
irrespective of whether the call to cudaFuncSetAttribute() is made before or after this API call. Because
of this and the stricter locking requirements mentioned below it is suggested that this call be used
during the initialization path and not on each thread accessing kernel such as on kernel launches or
on the critical path.

Valid values for attr are:

cudaFuncAttributeMaxDynamicSharedMemorySize  - The requested maximum

##### **‣**

size in bytes of dynamically-allocated shared memory. The sum of this value
and the function attribute sharedSizeBytes cannot exceed the device attribute
cudaDevAttrMaxSharedMemoryPerBlockOptin. The maximal size of requestable dynamic shared
memory may differ by GPU architecture.
cudaFuncAttributePreferredSharedMemoryCarveout   - On devices where the L1 cache and shared

##### **‣**

memory use the same hardware resources, this sets the shared memory carveout preference, in
percent of the total shared memory. See cudaDevAttrMaxSharedMemoryPerMultiprocessor. This
is only a hint, and the driver can choose a different ratio if required to execute the function.
cudaFuncAttributeRequiredClusterWidth: The required cluster width in blocks. The width, height,

##### **‣**

and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is
checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it
at runtime will return cudaErrorNotPermitted.
cudaFuncAttributeRequiredClusterHeight: The required cluster height in blocks. The width, height,

##### **‣**

and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is
checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it
at runtime will return cudaErrorNotPermitted.
cudaFuncAttributeRequiredClusterDepth: The required cluster depth in blocks. The width, height,

##### **‣**

and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is
checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it
at runtime will return cudaErrorNotPermitted.
cudaFuncAttributeNonPortableClusterSizeAllowed: Indicates whether the function can be

##### **‣**

launched with non-portable cluster size. 1 is allowed, 0 is disallowed.
cudaFuncAttributeClusterSchedulingPolicyPreference: The block scheduling policy of a function.

##### **‣**

The value type is cudaClusterSchedulingPolicy.





See also:


CUDA Runtime API vRelease Version  |  433


Modules


cudaLibraryLoadData, cudaLibraryLoadFromFile, cudaLibraryUnload, cudaLibraryGetKernel,
cudaLaunchKernel, cudaFuncSetAttribute, cuKernelSetAttribute