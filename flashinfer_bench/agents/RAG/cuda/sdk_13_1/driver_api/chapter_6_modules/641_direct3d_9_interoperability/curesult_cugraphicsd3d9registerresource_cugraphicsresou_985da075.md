# CUresult cuGraphicsD3D9RegisterResource (CUgraphicsResource *pCudaResource, IDirect3DResource9 *pD3DResource, unsigned int Flags)

Register a Direct3D 9 resource for access by CUDA.

###### Parameters

**pCudaResource**

  - Returned graphics resource handle
**pD3DResource**

  - Direct3D resource to register
**Flags**

  - Parameters for resource registration

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_UNKNOWN

###### Description

Registers the Direct3D 9 resource pD3DResource for access by CUDA and returns a CUDA handle
to pD3Dresource in pCudaResource. The handle returned in pCudaResource may be used
to map and unmap this resource until it is unregistered. On success this call will increase the internal
reference count on pD3DResource. This reference count will be decremented when this resource is
unregistered through cuGraphicsUnregisterResource().

This call is potentially high-overhead and should not be called every frame in interactive applications.

The type of pD3DResource must be one of the following.

IDirect3DVertexBuffer9: may be accessed through a device pointer

###### **‣**

IDirect3DIndexBuffer9: may be accessed through a device pointer

###### **‣**

IDirect3DSurface9: may be accessed through an array. Only stand-alone objects of type

###### **‣**

IDirect3DSurface9 may be explicitly shared. In particular, individual mipmap levels and faces of
cube maps may not be registered directly. To access individual surfaces associated with a texture,
one must register the base texture object.
IDirect3DBaseTexture9: individual surfaces on this texture may be accessed through an array.

###### **‣**

The Flags argument may be used to specify additional parameters at register time. The valid values
for this parameter are


CUDA Driver API TRM-06703-001 _vRelease Version  |  617


Modules


CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be

###### **‣**

used.
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will bind this

###### **‣**

resource to a surface reference.
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform

###### **‣**

texture gather operations on this resource.

Not all Direct3D resources of the above types may be used for interoperability with CUDA. The
following are some limitations.

The primary rendertarget may not be registered with CUDA.

###### **‣**

Resources allocated as shared may not be registered with CUDA.

###### **‣**

Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or

###### **‣**

floating-point data cannot be shared.
Surfaces of depth or stencil formats cannot be shared.

###### **‣**

A complete list of supported formats is as follows:

D3DFMT_L8

###### **‣**

D3DFMT_L16

###### **‣**

D3DFMT_A8R8G8B8

###### **‣**

D3DFMT_X8R8G8B8

###### **‣**

D3DFMT_G16R16

###### **‣**

D3DFMT_A8B8G8R8

###### **‣**

D3DFMT_A8

###### **‣**

D3DFMT_A8L8

###### **‣**

D3DFMT_Q8W8V8U8

###### **‣**

D3DFMT_V16U16

###### **‣**

D3DFMT_A16B16G16R16F

###### **‣**

D3DFMT_A16B16G16R16

###### **‣**

D3DFMT_R32F

###### **‣**

D3DFMT_G16R16F

###### **‣**

D3DFMT_A32B32G32R32F

###### **‣**

D3DFMT_G32R32F

###### **‣**

D3DFMT_R16F

###### **‣**

If Direct3D interoperability is not initialized for this context using cuD3D9CtxCreate then
CUDA_ERROR_INVALID_CONTEXT is returned. If pD3DResource is of incorrect type or
is already registered then CUDA_ERROR_INVALID_HANDLE is returned. If pD3DResource
cannot be registered then CUDA_ERROR_UNKNOWN is returned. If Flags is not one of the above
specified value then CUDA_ERROR_INVALID_VALUE is returned.


CUDA Driver API TRM-06703-001 _vRelease Version  |  618


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuD3D9CtxCreate, cuGraphicsUnregisterResource, cuGraphicsMapResources,
cuGraphicsSubResourceGetMappedArray, cuGraphicsResourceGetMappedPointer,
cudaGraphicsD3D9RegisterResource