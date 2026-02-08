# __host__make_cudaPitchedPtr (void *d, size_t p, size_t xsz, size_t ysz)

Returns a cudaPitchedPtr based on input parameters.

##### Parameters

**d**

  - Pointer to allocated memory
**p**

  - Pitch of allocated memory in bytes
**xsz**

  - Logical width of allocation in elements
**ysz**

  - Logical height of allocation in elements

##### Returns

cudaPitchedPtr specified by d, p, xsz, and ysz

##### Description

Returns a cudaPitchedPtr based on the specified input parameters d, p, xsz, and ysz.


See also:

make_cudaExtent, make_cudaPos


CUDA Runtime API vRelease Version  |  202


Modules