# __host__cudaError_t cudaFuncSetAttribute (const void *func, cudaFuncAttribute attr, int value)

Set attributes for a given function.

##### Parameters

**func**

  - Function to get attributes of
**attr**

  - Attribute to set
**value**

  - Value to set

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue

##### Description

This function sets the attributes of a function specified via func. The parameter func must be a
pointer to a function that executes on the device. The parameter specified by func must be declared
as a __global__ function. The enumeration defined by attr is set to the value defined by value.
If the specified function does not exist, then it is assumed to be a cudaKernel_t and used as is. If
the specified attribute cannot be written, or if the value is incorrect, then cudaErrorInvalidValue is
returned.

Valid values for attr are:


CUDA Runtime API vRelease Version  |  98


Modules


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









CUDA Runtime API vRelease Version  |  99


Modules


cudaLaunchKernel (C++ API), cudaFuncSetCacheConfig ( C++ API), cudaFuncGetAttributes ( C
API),