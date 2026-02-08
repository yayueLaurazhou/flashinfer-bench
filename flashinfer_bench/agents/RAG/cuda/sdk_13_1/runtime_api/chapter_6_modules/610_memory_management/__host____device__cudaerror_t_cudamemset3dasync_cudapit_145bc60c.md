# __host____device__cudaError_t cudaMemset3DAsync (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream)

Initializes or sets device memory to a value.

##### Parameters

**pitchedDevPtr**

  - Pointer to pitched device memory
**value**

  - Value to set for each byte of specified memory


CUDA Runtime API vRelease Version  |  197


Modules


**extent**

  - Size parameters for where to set device memory (width field in bytes)
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Initializes each element of a 3D array to the specified value value. The object to initialize is defined
by pitchedDevPtr. The pitch field of pitchedDevPtr is the width in memory in bytes of
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

cudaMemset3DAsync() is asynchronous with respect to the host, so the call may return before the
memset is complete. The operation can optionally be associated to a stream by passing a non-zero
stream argument. If stream is non-zero, the operation may overlap with operations in other
streams.

The device version of this function only handles device to device copies and cannot be given local or
shared pointers.













CUDA Runtime API vRelease Version  |  198


Modules


See also:

cudaMemset, cudaMemset2D, cudaMemset3D, cudaMemsetAsync, cudaMemset2DAsync,
cudaMalloc3D, make_cudaPitchedPtr, make_cudaExtent