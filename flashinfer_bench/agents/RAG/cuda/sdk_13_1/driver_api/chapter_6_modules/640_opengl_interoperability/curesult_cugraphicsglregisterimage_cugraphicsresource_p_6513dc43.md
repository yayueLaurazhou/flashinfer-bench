# CUresult cuGraphicsGLRegisterImage (CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags)

Register an OpenGL texture or renderbuffer object.

###### Parameters

**pCudaResource**

  - Pointer to the returned object handle
**image**

  - name of texture or renderbuffer object to be registered
**target**

  - Identifies the type of object specified by image
**Flags**

  - Register flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_OPERATING_SYSTEM

###### Description

Registers the texture or renderbuffer object specified by image for access by CUDA. A handle to the
registered object is returned as pCudaResource.

target must match the type of the object, and must be one of GL_TEXTURE_2D,
GL_TEXTURE_RECTANGLE, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D,
GL_TEXTURE_2D_ARRAY, or GL_RENDERBUFFER.

The register flags Flags specify the intended usage, as follows:

CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be

###### **‣**

used. It is therefore assumed that this resource will be read from and written to by CUDA. This is
the default value.


CUDA Driver API TRM-06703-001 _vRelease Version  |  602


Modules


CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that CUDA will not write to this

###### **‣**

resource.
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that CUDA will not read

###### **‣**

from this resource and will write over the entire contents of the resource, so none of the data
previously stored in the resource will be preserved.
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will bind this

###### **‣**

resource to a surface reference.
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform

###### **‣**

texture gather operations on this resource.

The following image formats are supported. For brevity's sake, the list is abbreviated. For ex.,
{GL_R, GL_RG} X {8, 16} would expand to the following 4 formats {GL_R8, GL_R16, GL_RG8,
GL_RG16} :

GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA,

###### **‣**

GL_INTENSITY
{GL_R, GL_RG, GL_RGBA} X {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}

###### **‣**

{GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} X {8, 16,

###### **‣**

16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT, 32I_EXT}

The following image classes are currently disallowed:

Textures with borders

###### **‣**

Multisampled renderbuffers

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGraphicsUnregisterResource, cuGraphicsMapResources, cuGraphicsSubResourceGetMappedArray,
cudaGraphicsGLRegisterImage