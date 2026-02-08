# __device__cudaError_t cudaGraphKernelNodeSetEnabled (cudaGraphDeviceNode_t node, bool enable)

Enables or disables the given kernel node.

##### Parameters

**node**

  - The node to update
**enable**

  - Whether to enable or disable the node

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Enables or disables node based upon enable. If enable is true, the node will be enabled; if it is
false, the node will be disabled. Disabled nodes will act as a NOP during execution. node must be
device-updatable, and must reside upon the same device as the calling kernel.

If this function is called for the node's immediate dependent and that dependent is configured for
programmatic dependent launch, then a memory fence must be invoked via __threadfence() before
kickoff of the dependent is triggered via cudaTriggerProgrammaticLaunchCompletion() to ensure that
the update is visible to that dependent node before it is launched.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Runtime API vRelease Version  |  398


Modules


cudaGraphKernelNodeSetParam, cudaGraphKernelNodeSetGridDim,
cudaGraphKernelNodeUpdatesApply