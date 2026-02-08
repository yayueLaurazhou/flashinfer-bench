# CUresult cuCtxGetExecAffinity (CUexecAffinityParam *pExecAffinity, CUexecAffinityType type)

Returns the execution affinity setting for the current context.

###### Parameters

**pExecAffinity**

  - Returned execution affinity


CUDA Driver API TRM-06703-001 _vRelease Version  |  127


Modules


**type**

  - Execution affinity type to query

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY

###### Description

Returns in *pExecAffinity the current value of type. The supported CUexecAffinityType values
are:

CU_EXEC_AFFINITY_TYPE_SM_COUNT: number of SMs the context is limited to use.

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

CUexecAffinityParam