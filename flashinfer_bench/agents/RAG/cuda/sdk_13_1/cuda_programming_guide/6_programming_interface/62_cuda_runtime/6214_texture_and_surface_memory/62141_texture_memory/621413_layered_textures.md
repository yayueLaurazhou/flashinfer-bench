# 6.2.14.1.3. Layered Textures

##### 6.2.14.1.3. Layered Textures[](#layered-textures "Permalink to this headline")

A one-dimensional or two-dimensional layered texture (also known as *texture array* in Direct3D and *array texture* in OpenGL) is a texture made up of a sequence of layers, all of which are regular textures of same dimensionality, size, and data type.

A one-dimensional layered texture is addressed using an integer index and a floating-point texture coordinate; the index denotes a layer within the sequence and the coordinate addresses a texel within that layer. A two-dimensional layered texture is addressed using an integer index and two floating-point texture coordinates; the index denotes a layer within the sequence and the coordinates address a texel within that layer.

A layered texture can only be a CUDA array by calling `cudaMalloc3DArray()` with the `cudaArrayLayered` flag (and a height of zero for one-dimensional layered texture).

Layered textures are fetched using the device functions described in [tex1DLayered()](#tex1dlayered-object) and [tex2DLayered()](#tex2dlayered-object). Texture filtering (see [Texture Fetching](#texture-fetching)) is done only within a layer, not across layers.

Layered textures are only supported on devices of compute capability 2.0 and higher.