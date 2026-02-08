# __host__cudaError_t cudaGraphicsGLRegisterImage (cudaGraphicsResource **resource, GLuint image, GLenum target, unsigned int flags)

Register an OpenGL texture or renderbuffer object.

##### Parameters

**resource**

  - Pointer to the returned object handle
**image**

  - name of texture or renderbuffer object to be registered
**target**

  - Identifies the type of object specified by image
**flags**

  - Register flags

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
cudaErrorOperatingSystem, cudaErrorUnknown

##### Description

Registers the texture or renderbuffer object specified by image for access by CUDA. A handle to the
registered object is returned as resource.

target must match the type of the object, and must be one of GL_TEXTURE_2D,
GL_TEXTURE_RECTANGLE, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D,
GL_TEXTURE_2D_ARRAY, or GL_RENDERBUFFER.

The register flags flags specify the intended usage, as follows:

cudaGraphicsRegisterFlagsNone: Specifies no hints about how this resource will be used. It is

##### **‣**

therefore assumed that this resource will be read from and written to by CUDA. This is the default
value.
cudaGraphicsRegisterFlagsReadOnly: Specifies that CUDA will not write to this resource.

##### **‣**

cudaGraphicsRegisterFlagsWriteDiscard: Specifies that CUDA will not read from this resource

##### **‣**

and will write over the entire contents of the resource, so none of the data previously stored in the
resource will be preserved.
cudaGraphicsRegisterFlagsSurfaceLoadStore: Specifies that CUDA will bind this resource to a

##### **‣**

surface reference.
cudaGraphicsRegisterFlagsTextureGather: Specifies that CUDA will perform texture gather

##### **‣**

operations on this resource.


CUDA Runtime API vRelease Version  |  234


Modules


The following image formats are supported. For brevity's sake, the list is abbreviated. For ex.,
{GL_R, GL_RG} X {8, 16} would expand to the following 4 formats {GL_R8, GL_R16, GL_RG8,
GL_RG16} :

GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA,

##### **‣**

GL_INTENSITY
{GL_R, GL_RG, GL_RGBA} X {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}

##### **‣**

{GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} X {8, 16,

##### **‣**

16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT, 32I_EXT}

The following image classes are currently disallowed:

Textures with borders

##### **‣**

Multisampled renderbuffers

##### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource, cudaGraphicsMapResources,
cudaGraphicsSubResourceGetMappedArray, cuGraphicsGLRegisterImage