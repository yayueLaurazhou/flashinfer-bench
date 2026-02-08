# template < typename T > __device__cudaError_t cudaGraphKernelNodeSetParam (cudaGraphDeviceNode_t node, size_t offset, const T value)

Updates the kernel parameters of the given kernel node.

##### Parameters

**node**

  - The node to update
**offset**

  - The offset into the params at which to make the update
**value**

  - Parameter value to write

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Updates the kernel parameters of node at offset to value. node must be device-updatable, and
must reside upon the same device as the calling kernel.

If this function is called for the node's immediate dependent and that dependent is configured for
programmatic dependent launch, then a memory fence must be invoked via __threadfence() before
kickoff of the dependent is triggered via cudaTriggerProgrammaticLaunchCompletion() to ensure that
the update is visible to that dependent node before it is launched.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphKernelNodeSetEnabled, cudaGraphKernelNodeSetGridDim,
cudaGraphKernelNodeUpdatesApply


CUDA Runtime API vRelease Version  |  400


Modules