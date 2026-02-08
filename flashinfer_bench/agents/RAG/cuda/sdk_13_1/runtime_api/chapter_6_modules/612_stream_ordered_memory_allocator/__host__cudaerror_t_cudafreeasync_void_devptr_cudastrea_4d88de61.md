# __host__cudaError_t cudaFreeAsync (void *devPtr, cudaStream_t hStream)

Frees memory with stream ordered semantics.

##### Parameters

**devPtr**
**hStream**

  - The stream establishing the stream ordering promise

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported

##### Description

Inserts a free operation into hStream. The allocation must not be accessed after stream execution
reaches the free. After this API returns, accessing the memory from any subsequent work launched on
the GPU or querying its pointer attributes results in undefined behavior.


Note:


During stream capture, this function results in the creation of a free node and must therefore be passed
the address of a graph allocation.











See also:

cuMemFreeAsync, cudaMallocAsync


CUDA Runtime API vRelease Version  |  211


Modules