# __host__cudaError_t cudaD3D9RegisterResource (IDirect3DResource9 *pResource, unsigned int flags)

Registers a Direct3D resource for access by CUDA.

##### Parameters

**pResource**

  - Resource to register
**flags**

  - Parameters for resource registration

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Registers the Direct3D resource pResource for access by CUDA.

If this call is successful, then the application will be able to map and unmap this resource until it is
unregistered through cudaD3D9UnregisterResource(). Also on success, this call will increase the
internal reference count on pResource. This reference count will be decremented when this resource
is unregistered through cudaD3D9UnregisterResource().

This call potentially has a high-overhead and should not be called every frame in interactive
applications.

The type of pResource must be one of the following.

IDirect3DVertexBuffer9: No notes.

##### **‣**

IDirect3DIndexBuffer9: No notes.

##### **‣**

IDirect3DSurface9: Only stand-alone objects of type IDirect3DSurface9 may be explicitly shared.

##### **‣**

In particular, individual mipmap levels and faces of cube maps may not be registered directly. To
access individual surfaces associated with a texture, one must register the base texture object.
IDirect3DBaseTexture9: When a texture is registered, all surfaces associated with all mipmap

##### **‣**

levels of all faces of the texture will be accessible to CUDA.

The flags argument specifies the mechanism through which CUDA will access the Direct3D
resource. The following value is allowed:

cudaD3D9RegisterFlagsNone: Specifies that CUDA will access this resource through a

##### **‣**

void*. The pointer, size, and pitch for each subresource of this resource may be queried
through cudaD3D9ResourceGetMappedPointer(), cudaD3D9ResourceGetMappedSize(), and
cudaD3D9ResourceGetMappedPitch() respectively. This option is valid for all resource types.


CUDA Runtime API vRelease Version  |  250


Modules


Not all Direct3D resources of the above types may be used for interoperability with CUDA. The
following are some limitations:

The primary rendertarget may not be registered with CUDA.

##### **‣**

Resources allocated as shared may not be registered with CUDA.

##### **‣**

Any resources allocated in D3DPOOL_SYSTEMMEM or D3DPOOL_MANAGED may not be

##### **‣**

registered with CUDA.
Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or

##### **‣**

floating-point data cannot be shared.
Surfaces of depth or stencil formats cannot be shared.

##### **‣**

If Direct3D interoperability is not initialized on this context, then cudaErrorInvalidDevice is returned.
If pResource is of incorrect type (e.g, is a non-stand-alone IDirect3DSurface9) or is already
registered, then cudaErrorInvalidResourceHandle is returned. If pResource cannot be registered then
cudaErrorUnknown is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsD3D9RegisterResource