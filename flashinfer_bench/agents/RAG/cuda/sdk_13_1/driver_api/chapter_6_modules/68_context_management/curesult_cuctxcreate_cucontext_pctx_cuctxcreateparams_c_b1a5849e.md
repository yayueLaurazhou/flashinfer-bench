# CUresult cuCtxCreate (CUcontext *pctx, CUctxCreateParams *ctxCreateParams, unsigned int flags, CUdevice dev)

Create a CUDA context.

###### Parameters

**pctx**

  - Returned context handle of the new context
**ctxCreateParams**

  - Context creation parameters
**flags**

  - Context creation flags
**dev**

  - Device to create context on

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN

###### Description

Creates a new CUDA context and associates it with the calling thread. The flags parameter is
described below. The context is created with a usage count of 1 and the caller of cuCtxCreate()
must call cuCtxDestroy() when done using the context. If a context is already current to the
thread, it is supplanted by the newly created context and may be restored by a subsequent call to
cuCtxPopCurrent().

CUDA context can be created with execution affinity. The type and the amount of execution
resource the context can use is limited by paramsArray and numExecAffinityParams
in execAffinity. The paramsArray is an array of CUexecAffinityParam
and the numExecAffinityParams describes the size of the paramsArray. If two
CUexecAffinityParam in the array have the same type, the latter execution affinity parameter
overrides the former execution affinity parameter. The supported execution affinity types are:

CU_EXEC_AFFINITY_TYPE_SM_COUNT limits the portion of SMs that the context can use.

###### **‣**

The portion of SMs is specified as the number of SMs via CUexecAffinitySmCount. This
limit will be internally rounded up to the next hardware-supported amount. Hence, it is imperative
to query the actual execution affinity of the context via cuCtxGetExecAffinity after context
creation. Currently, this attribute is only supported under Volta+ MPS.


CUDA Driver API TRM-06703-001 _vRelease Version  |  120


Modules


CUDA context can be created in CIG(CUDA in Graphics) mode by setting cigParams.
Data from graphics client is shared with CUDA via the sharedData in cigParams.
Support for D3D12 graphics client can be determined using cuDeviceGetAttribute()
with CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED. sharedData is a
ID3D12CommandQueue handle. Support for Vulkan graphics client can be determined using
cuDeviceGetAttribute() with CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED.
sharedData is a Nvidia specific data blob populated by calling
vkGetExternalComputeQueueDataNV(). Either execAffinityParams or cigParams can be set
to a non-null value. Setting both to a non-null value will result in an undefined behavior.

The three LSBs of the flags parameter can be used to control how the OS thread, which owns the
CUDA context at the time of an API call, interacts with the OS scheduler when waiting for results from
the GPU. Only one of the scheduling flags can be set when creating a context.

CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for results from the GPU.

###### **‣**

This can decrease latency when waiting for the GPU, but may lower the performance of CPU
threads if they are performing work in parallel with the CUDA thread.

CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for results from the

###### **‣**

GPU. This can increase latency when waiting for the GPU, but can increase the performance of
CPU threads performing work in parallel with the GPU.

CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a

###### **‣**

synchronization primitive when waiting for the GPU to finish work.

CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a synchronization

###### **‣**

primitive when waiting for the GPU to finish work.

Deprecated: This flag was deprecated as of CUDA 4.0 and was replaced with
CU_CTX_SCHED_BLOCKING_SYNC.

CU_CTX_SCHED_AUTO: The default value if the parameter is zero, uses a heuristic

###### ‣ flags

based on the number of active CUDA contexts in the process C and the number of logical
processors in the system P. If C > P, then CUDA will yield to other OS threads when waiting for
the GPU (CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while waiting for results
and actively spin on the processor (CU_CTX_SCHED_SPIN). Additionally, on Tegra devices,
CU_CTX_SCHED_AUTO uses a heuristic based on the power profile of the platform and may
choose CU_CTX_SCHED_BLOCKING_SYNC for low-powered devices.

CU_CTX_MAP_HOST: Instruct CUDA to support mapped pinned allocations. This flag must be

###### **‣**

set in order to allocate pinned host memory that is accessible to the GPU.

CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory after resizing

###### **‣**

local memory for a kernel. This can prevent thrashing by local memory allocations when launching
many kernels with high local memory usage at the cost of potentially increased memory usage.

Deprecated: This flag is deprecated and the behavior enabled by this flag is now the default and
cannot be disabled. Instead, the per-thread stack size can be controlled with cuCtxSetLimit().


CUDA Driver API TRM-06703-001 _vRelease Version  |  121


Modules


CU_CTX_COREDUMP_ENABLE: If GPU coredumps have not been enabled globally with

###### **‣**

cuCoredumpSetAttributeGlobal or environment variables, this flag can be set during context
creation to instruct CUDA to create a coredump if this context raises an exception during
execution. These environment variables are described in the CUDA-GDB user guide under the
"GPU core dump support" section. The initial attributes will be taken from the global attributes at
the time of context creation. The other attributes that control coredump output can be modified by
calling cuCoredumpSetAttribute from the created context after it becomes current. This flag is not
supported when CUDA context is created in CIG(CUDA in Graphics) mode.

CU_CTX_USER_COREDUMP_ENABLE: If user-triggered GPU coredumps have not been

###### **‣**

enabled globally with cuCoredumpSetAttributeGlobal or environment variables, this flag can be
set during context creation to instruct CUDA to create a coredump if data is written to a certain
pipe that is present in the OS space. These environment variables are described in the CUDAGDB user guide under the "GPU core dump support" section. It is important to note that the pipe
name *must* be set with cuCoredumpSetAttributeGlobal before creating the context if this flag is
used. Setting this flag implies that CU_CTX_COREDUMP_ENABLE is set. The initial attributes
will be taken from the global attributes at the time of context creation. The other attributes that
control coredump output can be modified by calling cuCoredumpSetAttribute from the created
context after it becomes current. Setting this flag on any context creation is equivalent to setting
the CU_COREDUMP_ENABLE_USER_TRIGGER attribute to true globally. This flag is not
supported when CUDA context is created in CIG(CUDA in Graphics) mode.

CU_CTX_SYNC_MEMOPS: Ensures that synchronous memory operations initiated on

###### **‣**

this context will always synchronize. See further documentation in the section titled "API
Synchronization behavior" to learn more about cases when synchronous memory operations can
exhibit asynchronous behavior.

Context creation will fail with CUDA_ERROR_UNKNOWN if the compute mode of the device
is CU_COMPUTEMODE_PROHIBITED. The function cuDeviceGetAttribute() can be used with
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode of the device. The
nvidia-smi tool can be used to set the compute mode for * devices. Documentation for nvidia-smi can
be obtained by passing a -h option to it.

Context creation will fail with :: CUDA_ERROR_INVALID_VALUE if invalid parameter was passed
by client to create the CUDA context.

Context creation in CIG mode will fail with CUDA_ERROR_NOT_SUPPORTED if CIG is not
supported by the device or the driver.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  122


Modules


cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice, cuCtxGetFlags,
cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit,
cuCoredumpSetAttributeGlobal, cuCoredumpSetAttribute, cuCtxSynchronize