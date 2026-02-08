# __host__cudaError_t cudaMemcpy3DPeerAsync (const cudaMemcpy3DPeerParms *p, cudaStream_t stream)

Copies memory between devices asynchronously.

##### Parameters

**p**

  - Parameters for the memory copy
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice, cudaErrorInvalidPitchValue

##### Description

Perform a 3D memory copy according to the parameters specified in p. See the definition of the
cudaMemcpy3DPeerParms structure for documentation of its parameters.











CUDA Runtime API vRelease Version  |  170


Modules





See also:

cudaMemcpy, cudaMemcpyPeer, cudaMemcpyAsync, cudaMemcpyPeerAsync,
cudaMemcpy3DPeerAsync, cuMemcpy3DPeerAsync