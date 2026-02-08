# __host__cudaError_t cudaD3D10RegisterResource (ID3D10Resource *pResource, unsigned int flags)

Registers a Direct3D 10 resource for access by CUDA.

##### Parameters

**pResource**

  - Resource to register
**flags**

  - Parameters for resource registration

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Registers the Direct3D resource pResource for access by CUDA.

If this call is successful, then the application will be able to map and unmap this resource until it is
unregistered through cudaD3D10UnregisterResource(). Also on success, this call will increase the
internal reference count on pResource. This reference count will be decremented when this resource
is unregistered through cudaD3D10UnregisterResource().

This call potentially has a high-overhead and should not be called every frame in interactive
applications.

The type of pResource must be one of the following:

ID3D10Buffer: Cannot be used with set to .

##### ‣ flags cudaD3D10RegisterFlagsArray

ID3D10Texture1D: No restrictions.

##### **‣**

ID3D10Texture2D: No restrictions.

##### **‣**

CUDA Runtime API vRelease Version  |  266


Modules


ID3D10Texture3D: No restrictions.

##### **‣**

The flags argument specifies the mechanism through which CUDA will access the Direct3D
resource. The following values are allowed.

cudaD3D10RegisterFlagsNone: Specifies that CUDA will access this resource through a

##### **‣**

void*. The pointer, size, and pitch for each subresource of this resource may be queried
through cudaD3D10ResourceGetMappedPointer(), cudaD3D10ResourceGetMappedSize(), and
cudaD3D10ResourceGetMappedPitch() respectively. This option is valid for all resource types.
cudaD3D10RegisterFlagsArray: Specifies that CUDA will access this resource through a

##### **‣**

CUarray queried on a sub-resource basis through cudaD3D10ResourceGetMappedArray().
This option is only valid for resources of type ID3D10Texture1D, ID3D10Texture2D, and
ID3D10Texture3D.

Not all Direct3D resources of the above types may be used for interoperability with CUDA. The
following are some limitations.

The primary rendertarget may not be registered with CUDA.

##### **‣**

Resources allocated as shared may not be registered with CUDA.

##### **‣**

Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or

##### **‣**

floating-point data cannot be shared.
Surfaces of depth or stencil formats cannot be shared.

##### **‣**

If Direct3D interoperability is not initialized on this context then cudaErrorInvalidDevice is returned.
If pResource is of incorrect type or is already registered then cudaErrorInvalidResourceHandle is
returned. If pResource cannot be registered then cudaErrorUnknown is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsD3D10RegisterResource


CUDA Runtime API vRelease Version  |  267


Modules