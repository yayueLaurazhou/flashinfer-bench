# CUresult cuFuncSetAttribute (CUfunction hfunc, CUfunction_attribute attrib, int value)

Sets information about a function.

###### Parameters

**hfunc**

  - Function to query attribute of
**attrib**

  - Attribute requested
**value**

  - The value to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE

###### Description

This call sets the value of a specified attribute attrib on the kernel given by hfunc to an
integer value specified by val This function returns CUDA_SUCCESS if the new value of the
attribute could be successfully set. If the set fails, this call will return an error. Not all attributes
can have values set. Attempting to set a value on a read-only attribute will result in an error
(CUDA_ERROR_INVALID_VALUE)

Supported attributes for the cuFuncSetAttribute call are:

CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: This maximum

###### **‣**

size in bytes of dynamically-allocated shared memory. The value should contain the requested
maximum size of dynamically-allocated shared memory. The sum of this value and the function
attribute CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the device attribute
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN. The maximal
size of requestable dynamic shared memory may differ by GPU architecture.
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: On

###### **‣**

devices where the L1 cache and shared memory use the same hardware resources, this


CUDA Driver API TRM-06703-001 _vRelease Version  |  387


Modules


sets the shared memory carveout preference, in percent of the total shared memory. See
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR This is
only a hint, and the driver can choose a different ratio if required to execute the function.
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: The required cluster width in

###### **‣**

blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of
the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot
be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: The required cluster height in

###### **‣**

blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of
the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot
be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: The required cluster depth in

###### **‣**

blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of
the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot
be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: Indicates whether

###### **‣**

the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed.
CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: The block

###### **‣**

scheduling policy of a function. The value type is CUclusterSchedulingPolicy.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuFuncSetCacheConfig, cuLaunchKernel,
cudaFuncGetAttributes, cudaFuncSetAttribute, cuKernelSetAttribute