# __device__cudaError_t cudaGraphKernelNodeUpdatesApply (const cudaGraphKernelNodeUpdate *updates, size_t updateCount)

Batch applies multiple kernel node updates.

##### Parameters

**updates**

  - The updates to apply
**updateCount**

  - The number of updates to apply

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Batch applies one or more kernel node updates based on the information provided in updates.
updateCount specifies the number of updates to apply. Each entry in updates must specify
a node to update, the type of update to apply, and the parameters for that type of update. See the
documentation for cudaGraphKernelNodeUpdate for more detail.

If this function is called for the node's immediate dependent and that dependent is configured for
programmatic dependent launch, then a memory fence must be invoked via __threadfence() before
kickoff of the dependent is triggered via cudaTriggerProgrammaticLaunchCompletion() to ensure that
the update is visible to that dependent node before it is launched.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphKernelNodeSetParam, cudaGraphKernelNodeSetEnabled,
cudaGraphKernelNodeSetGridDim


CUDA Runtime API vRelease Version  |  403


Modules