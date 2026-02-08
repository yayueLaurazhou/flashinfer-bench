# __host__cudaError_t cudaCtxResetPersistingL2Cache (void)

Resets all persisting lines in cache to normal status.

##### Returns

cudaSuccess,

##### Description

Resets all persisting lines in cache to normal status. Takes effect on function return.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaAccessPolicyWindow