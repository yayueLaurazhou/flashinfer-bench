# __host__cudaError_t cudaMemcpy3D (const cudaMemcpy3DParms *p)

Copies data between 3D objects.

##### Parameters

**p**

  - 3D memory copy parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

##### Description

CUDA Runtime API vRelease Version  |  163


Modules


cudaMemcpy3D() copies data betwen two 3D objects. The source and destination objects may be in
either host memory, device memory, or a CUDA array. The source, destination, extent, and kind of
copy performed is specified by the cudaMemcpy3DParms struct which should be initialized to zero
before use:
‎cudaMemcpy3DParms myParms = {0};

The struct passed to cudaMemcpy3D() must specify one of srcArray or srcPtr and one
of dstArray or dstPtr. Passing more than one non-zero source or destination will cause
cudaMemcpy3D() to return an error.

The srcPos and dstPos fields are optional offsets into the source and destination objects and are
defined in units of each object's elements. The element for a host or device pointer is assumed to be
unsigned char.

The extent field defines the dimensions of the transferred area in elements. If a CUDA array is
participating in the copy, the extent is defined in terms of that array's elements. If no CUDA array is
participating in the copy then the extents are defined in elements of unsigned char.

The kind field defines the direction of the copy. It must be one of cudaMemcpyHostToHost,
cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or
cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer
is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that
support unified virtual addressing. For cudaMemcpyHostToHost or cudaMemcpyHostToDevice
or cudaMemcpyDeviceToHost passed as kind and cudaArray type passed as source or destination,
if the kind implies cudaArray type to be present on the host, cudaMemcpy3D() will disregard that
implication and silently correct the kind based on the fact that cudaArray type can only be present on
the device.

If the source and destination are both arrays, cudaMemcpy3D() will return an error if they do not have
the same element size.

The source and destination object may not overlap. If overlapping source and destination objects are
specified, undefined behavior will result.

The source object must entirely contain the region defined by srcPos and extent. The destination
object must entirely contain the region defined by dstPos and extent.

cudaMemcpy3D() returns an error if the pitch of srcPtr or dstPtr exceeds the maximum allowed.
The pitch of a cudaPitchedPtr allocated with cudaMalloc3D() will always be valid.









CUDA Runtime API vRelease Version  |  164


Modules





See also:

cudaMalloc3D, cudaMalloc3DArray, cudaMemset3D, cudaMemcpy3DAsync,
cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyFromSymbol,
cudaMemcpyAsync, cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync,
cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync,
make_cudaExtent, make_cudaPos, cuMemcpy3D