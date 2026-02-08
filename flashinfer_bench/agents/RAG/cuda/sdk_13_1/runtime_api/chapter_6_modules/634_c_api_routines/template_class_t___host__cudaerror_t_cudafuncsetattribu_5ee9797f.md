# template < class T > __host__cudaError_t cudaFuncSetAttribute (T *func, cudaFuncAttribute attr, int value)

[C++ API] Set attributes for a given function

##### Parameters

**func**
**attr**

  - Attribute to set
**value**

  - Value to set

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue

##### Description

This function sets the attributes of a function specified via entry. The parameter entry must
be a pointer to a function that executes on the device. The parameter specified by entry must be
declared as a __global__ function. The enumeration defined by attr is set to the value defined
by value. If the specified function does not exist, then cudaErrorInvalidDeviceFunction is returned.
If the specified attribute cannot be written, or if the value is incorrect, then cudaErrorInvalidValue is
returned.

Valid values for attr are:

cudaFuncAttributeMaxDynamicSharedMemorySize  - The requested maximum

##### **‣**

size in bytes of dynamically-allocated shared memory. The sum of this value
and the function attribute sharedSizeBytes cannot exceed the device attribute


CUDA Runtime API vRelease Version  |  466


Modules


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









cudaLaunchKernel ( C++ API), cudaFuncSetCacheConfig ( C++ API), cudaFuncGetAttributes ( C
API), cudaSetDoubleForDevice, cudaSetDoubleForHost


CUDA Runtime API vRelease Version  |  467


Modules