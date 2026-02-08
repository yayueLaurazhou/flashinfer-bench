# CUresult cuKernelSetAttribute (CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev)

Sets information about a kernel.

###### Parameters

**attrib**

  - Attribute requested
**val**

  - Value to set
**kernel**

  - Kernel to set attribute of
**dev**

  - Device to set attribute of


CUDA Driver API TRM-06703-001 _vRelease Version  |  163


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_OUT_OF_MEMORY

###### Description

This call sets the value of a specified attribute attrib on the kernel kernel for the requested
device dev to an integer value specified by val. This function returns CUDA_SUCCESS if the new
value of the attribute could be successfully set. If the set fails, this call will return an error. Not all
attributes can have values set. Attempting to set a value on a read-only attribute will result in an error
(CUDA_ERROR_INVALID_VALUE)

Note that attributes set using cuFuncSetAttribute() will override the attribute set by this API
irrespective of whether the call to cuFuncSetAttribute() is made before or after this API call. However,
cuKernelGetAttribute() will always return the attribute value set by this API.

Supported attributes are:

CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: This is the maximum

###### **‣**

size in bytes of dynamically-allocated shared memory. The value should contain the requested
maximum size of dynamically-allocated shared memory. The sum of this value and the function
attribute CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the device attribute
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN. The maximal
size of requestable dynamic shared memory may differ by GPU architecture.
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: On

###### **‣**

devices where the L1 cache and shared memory use the same hardware resources, this
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


CUDA Driver API TRM-06703-001 _vRelease Version  |  164


Modules


CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: The block

###### **‣**

scheduling policy of a function. The value type is CUclusterSchedulingPolicy.





See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuKernelGetAttribute,
cuLibraryGetKernel, cuLaunchKernel, cuKernelGetFunction, cuLibraryGetModule,
cuModuleGetFunction, cuFuncSetAttribute