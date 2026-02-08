# CUresult cuDevicePrimaryCtxSetFlags (CUdevice dev, unsigned int flags)

Set flags for the primary context.

###### Parameters

**dev**

  - Device for which the primary context flags are set
**flags**

  - New flags for the device

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE,

###### Description

Sets the flags for the primary context on the device overwriting perviously set ones.

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


CUDA Driver API TRM-06703-001 _vRelease Version  |  118


Modules


CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory after resizing

###### **‣**

local memory for a kernel. This can prevent thrashing by local memory allocations when launching
many kernels with high local memory usage at the cost of potentially increased memory usage.

Deprecated: This flag is deprecated and the behavior enabled by this flag is now the default and
cannot be disabled.

CU_CTX_COREDUMP_ENABLE: If GPU coredumps have not been enabled globally with

###### **‣**

cuCoredumpSetAttributeGlobal or environment variables, this flag can be set during context
creation to instruct CUDA to create a coredump if this context raises an exception during
execution. These environment variables are described in the CUDA-GDB user guide under the
"GPU core dump support" section. The initial settings will be taken from the global settings at
the time of context creation. The other settings that control coredump output can be modified by
calling cuCoredumpSetAttribute from the created context after it becomes current.

CU_CTX_USER_COREDUMP_ENABLE: If user-triggered GPU coredumps have not been

###### **‣**

enabled globally with cuCoredumpSetAttributeGlobal or environment variables, this flag can be
set during context creation to instruct CUDA to create a coredump if data is written to a certain
pipe that is present in the OS space. These environment variables are described in the CUDA-GDB
user guide under the "GPU core dump support" section. It is important to note that the pipe name
*must* be set with cuCoredumpSetAttributeGlobal before creating the context if this flag is used.
Setting this flag implies that CU_CTX_COREDUMP_ENABLE is set. The initial settings will
be taken from the global settings at the time of context creation. The other settings that control
coredump output can be modified by calling cuCoredumpSetAttribute from the created context
after it becomes current.

CU_CTX_SYNC_MEMOPS: Ensures that synchronous memory operations initiated on

###### **‣**

this context will always synchronize. See further documentation in the section titled "API
Synchronization behavior" to learn more about cases when synchronous memory operations can
exhibit asynchronous behavior.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDevicePrimaryCtxRetain, cuDevicePrimaryCtxGetState, cuCtxCreate, cuCtxGetFlags,
cuCtxSetFlags, cudaSetDeviceFlags