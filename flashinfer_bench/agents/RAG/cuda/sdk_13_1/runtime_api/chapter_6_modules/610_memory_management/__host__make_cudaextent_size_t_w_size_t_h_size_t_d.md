# __host__make_cudaExtent (size_t w, size_t h, size_t d)

Returns a cudaExtent based on input parameters.

##### Parameters

**w**

  - Width in elements when referring to array memory, in bytes when referring to linear memory
**h**

  - Height in elements


CUDA Runtime API vRelease Version  |  201


Modules


**d**

  - Depth in elements

##### Returns

cudaExtent specified by w, h, and d

##### Description

Returns a cudaExtent based on the specified input parameters w, h, and d.


See also:

make_cudaPitchedPtr, make_cudaPos