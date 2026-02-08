# __host__cudaError_t cudaGraphUpload (cudaGraphExec_t graphExec, cudaStream_t stream)

Uploads an executable graph in a stream.

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Uploads hGraphExec to the device in hStream without executing it. Uploads of the same
hGraphExec will be serialized. Each upload is ordered behind both any previous work in hStream
and any previous launches of hGraphExec. Uses memory cached by stream to back the allocations
owned by graphExec.





See also:


CUDA Runtime API vRelease Version  |  425


Modules


cudaGraphInstantiate, cudaGraphLaunch, cudaGraphExecDestroy