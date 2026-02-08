# __host__cudaError_t cudaMemset3D (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent)

Initializes or sets device memory to a value.

##### Parameters

**pitchedDevPtr**

  - Pointer to pitched device memory
**value**

  - Value to set for each byte of specified memory
**extent**

  - Size parameters for where to set device memory (width field in bytes)

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Initializes each element of a 3D array to the specified value value. The object to initialize is defined
by pitchedDevPtr. The pitch field of pitchedDevPtr is the width in memory in bytes of


CUDA Runtime API vRelease Version  |  196


Modules


the 3D array pointed to by pitchedDevPtr, including any padding added to the end of each row.
The xsize field specifies the logical width of each row in bytes, while the ysize field specifies the
height of each 2D slice in rows. The pitch field of pitchedDevPtr is ignored when height and
depth are both equal to 1.

The extents of the initialized region are specified as a width in bytes, a height in rows, and a
depth in slices.

Extents with width greater than or equal to the xsize of pitchedDevPtr may perform
significantly faster than extents narrower than the xsize. Secondarily, extents with height equal
to the ysize of pitchedDevPtr will perform faster than when the height is shorter than the
ysize.

This function performs fastest when the pitchedDevPtr has been allocated by cudaMalloc3D().

Note that this function is asynchronous with respect to the host unless pitchedDevPtr refers to
pinned host memory.











See also:

cudaMemset, cudaMemset2D, cudaMemsetAsync, cudaMemset2DAsync, cudaMemset3DAsync,
cudaMalloc3D, make_cudaPitchedPtr, make_cudaExtent