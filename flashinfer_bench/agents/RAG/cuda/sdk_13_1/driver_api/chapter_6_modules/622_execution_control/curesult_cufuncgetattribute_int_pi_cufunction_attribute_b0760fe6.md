# CUresult cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)

Returns information about a function.

###### Parameters

**pi**

  - Returned attribute value
**attrib**

  - Attribute requested
**hfunc**

  - Function to query attribute of

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_FUNCTION_NOT_LOADED

###### Description

Returns in *pi the integer value of the attribute attrib on the kernel given by hfunc. The
supported attributes are:

CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: The maximum number of threads

###### **‣**

per block, beyond which a launch of the function would fail. This number depends on both the
function and the device on which the function is currently loaded.
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: The size in bytes of statically-allocated

###### **‣**

shared memory per block required by this function. This does not include dynamically-allocated
shared memory requested by the user at runtime.
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: The size in bytes of user-allocated constant

###### **‣**

memory required by this function.
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: The size in bytes of local memory used by

###### **‣**

each thread of this function.
CU_FUNC_ATTRIBUTE_NUM_REGS: The number of registers used by each thread of this

###### **‣**

function.
CU_FUNC_ATTRIBUTE_PTX_VERSION: The PTX virtual architecture version for which the

###### **‣**

function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a
PTX version 1.3 function would return the value 13. Note that this may return the undefined value
of 0 for cubins compiled prior to CUDA 3.0.
CU_FUNC_ATTRIBUTE_BINARY_VERSION: The binary architecture version for which the

###### **‣**

function was compiled. This value is the major binary version * 10 + the minor binary version, so


CUDA Driver API TRM-06703-001 _vRelease Version  |  382


Modules


a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for
legacy cubins that do not have a properly-encoded binary architecture version.
CU_FUNC_CACHE_MODE_CA: The attribute to indicate whether the function has been

###### **‣**

compiled with user specified option "-Xptxas --dlcm=ca" set .
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: The maximum size in

###### **‣**

bytes of dynamically-allocated shared memory.
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: Preferred shared

###### **‣**

memory-L1 cache split ratio in percent of total shared memory.
CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: If this attribute is set, the kernel

###### **‣**

must launch with a valid cluster size specified.
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: The required cluster width in

###### **‣**

blocks.
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: The required cluster height in

###### **‣**

blocks.
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: The required cluster depth in

###### **‣**

blocks.
CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: Indicates whether

###### **‣**

the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed. A
non-portable cluster size may only function on the specific SKUs the program is tested on. The
launch might fail if the program is run on a different hardware platform. CUDA API provides
cudaOccupancyMaxActiveClusters to assist with checking whether the desired size can be
launched on the current device. A portable cluster size is guaranteed to be functional on all
compute capabilities higher than the target compute capability. The portable cluster size for sm_90
is 8 blocks per cluster. This value may increase for future compute capabilities. The specific
hardware unit may support higher cluster sizes that’s not guaranteed to be portable.
CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: The block

###### **‣**

scheduling policy of a function. The value type is CUclusterSchedulingPolicy.

With a few execeptions, function attributes may also be queried on unloaded function handles returned
from cuModuleEnumerateFunctions. CUDA_ERROR_FUNCTION_NOT_LOADED is returned if the
attribute requires a fully loaded function but the function is not loaded. The loading state of a function
may be queried using cuFuncIsloaded. cuFuncLoad may be called to explicitly load a function before
querying the following attributes that require the function to be loaded:

CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK

###### **‣**

CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES

###### **‣**

CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  383


Modules


See also:

cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuFuncSetCacheConfig, cuLaunchKernel,
cudaFuncGetAttributes, cudaFuncSetAttribute, cuFuncIsLoaded, cuFuncLoad, cuKernelGetAttribute