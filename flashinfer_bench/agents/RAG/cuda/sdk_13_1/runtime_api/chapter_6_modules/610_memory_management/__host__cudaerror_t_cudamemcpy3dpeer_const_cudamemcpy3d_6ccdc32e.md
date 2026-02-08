# __host__cudaError_t cudaMemcpy3DPeer (const cudaMemcpy3DPeerParms *p)

Copies memory between devices.

##### Parameters

**p**

  - Parameters for the memory copy

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice, cudaErrorInvalidPitchValue

##### Description

Perform a 3D memory copy according to the parameters specified in p. See the definition of the
cudaMemcpy3DPeerParms structure for documentation of its parameters.

Note that this function is synchronous with respect to the host only if the source or destination of the
transfer is host memory. Note also that this copy is serialized with respect to all pending and future


CUDA Runtime API vRelease Version  |  169


Modules


asynchronous work in to the current device, the copy's source device, and the copy's destination device
(use cudaMemcpy3DPeerAsync to avoid this synchronization).











See also:

cudaMemcpy, cudaMemcpyPeer, cudaMemcpyAsync, cudaMemcpyPeerAsync,
cudaMemcpy3DPeerAsync, cuMemcpy3DPeer