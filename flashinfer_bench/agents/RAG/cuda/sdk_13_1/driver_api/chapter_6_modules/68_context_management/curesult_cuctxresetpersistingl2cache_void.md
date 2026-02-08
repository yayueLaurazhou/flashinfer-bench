# CUresult cuCtxResetPersistingL2Cache (void)

Resets all persisting lines in cache to normal status.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_SUPPORTED


CUDA Driver API TRM-06703-001 _vRelease Version  |  133


Modules

###### Description

cuCtxResetPersistingL2Cache Resets all persisting lines in cache to normal status. Takes effect on
function return.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

CUaccessPolicyWindow