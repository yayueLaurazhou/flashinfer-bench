# CUresult cuGraphicsD3D11RegisterResource (CUgraphicsResource *pCudaResource, ID3D11Resource *pD3DResource, unsigned int Flags)

Register a Direct3D 11 resource for access by CUDA.

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

Registers the Direct3D 11 resource pD3DResource for access by CUDA and returns a CUDA handle
to pD3Dresource in pCudaResource. The handle returned in pCudaResource may be used
to map and unmap this resource until it is unregistered. On success this call will increase the internal
reference count on pD3DResource. This reference count will be decremented when this resource is
unregistered through cuGraphicsUnregisterResource().

This call is potentially high-overhead and should not be called every frame in interactive applications.

The type of pD3DResource must be one of the following.

ID3D11Buffer: may be accessed through a device pointer.

###### **‣**

ID3D11Texture1D: individual subresources of the texture may be accessed via arrays

###### **‣**

ID3D11Texture2D: individual subresources of the texture may be accessed via arrays

###### **‣**

ID3D11Texture3D: individual subresources of the texture may be accessed via arrays

###### **‣**

The Flags argument may be used to specify additional parameters at register time. The valid values
for this parameter are

CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be

###### **‣**

used.
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will bind this

###### **‣**

resource to a surface reference.


CUDA Driver API TRM-06703-001 _vRelease Version  |  651


Modules


CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform

###### **‣**

texture gather operations on this resource.

Not all Direct3D resources of the above types may be used for interoperability with CUDA. The
following are some limitations.

The primary rendertarget may not be registered with CUDA.

###### **‣**

Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or

###### **‣**

floating-point data cannot be shared.
Surfaces of depth or stencil formats cannot be shared.

###### **‣**

A complete list of supported DXGI formats is as follows. For compactness the notation A_{B,C,D}
represents A_B, A_C, and A_D.

DXGI_FORMAT_A8_UNORM

###### **‣**

DXGI_FORMAT_B8G8R8A8_UNORM

###### **‣**

DXGI_FORMAT_B8G8R8X8_UNORM

###### **‣**

DXGI_FORMAT_R16_FLOAT

###### **‣**

DXGI_FORMAT_R16G16B16A16_{FLOAT,SINT,SNORM,UINT,UNORM}

###### **‣**

DXGI_FORMAT_R16G16_{FLOAT,SINT,SNORM,UINT,UNORM}

###### **‣**

DXGI_FORMAT_R16_{SINT,SNORM,UINT,UNORM}

###### **‣**

DXGI_FORMAT_R32_FLOAT

###### **‣**

DXGI_FORMAT_R32G32B32A32_{FLOAT,SINT,UINT}

###### **‣**

DXGI_FORMAT_R32G32_{FLOAT,SINT,UINT}

###### **‣**

DXGI_FORMAT_R32_{SINT,UINT}

###### **‣**

DXGI_FORMAT_R8G8B8A8_{SINT,SNORM,UINT,UNORM,UNORM_SRGB}

###### **‣**

DXGI_FORMAT_R8G8_{SINT,SNORM,UINT,UNORM}

###### **‣**

DXGI_FORMAT_R8_{SINT,SNORM,UINT,UNORM}

###### **‣**

If pD3DResource is of incorrect type or is already registered then
CUDA_ERROR_INVALID_HANDLE is returned. If pD3DResource cannot be registered then
CUDA_ERROR_UNKNOWN is returned. If Flags is not one of the above specified value then
CUDA_ERROR_INVALID_VALUE is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGraphicsUnregisterResource, cuGraphicsMapResources, cuGraphicsSubResourceGetMappedArray,
cuGraphicsResourceGetMappedPointer, cudaGraphicsD3D11RegisterResource


CUDA Driver API TRM-06703-001 _vRelease Version  |  652


Modules