# 6.2.14.1.5. Cubemap Layered Textures

##### 6.2.14.1.5. Cubemap Layered Textures[](#cubemap-layered-textures "Permalink to this headline")

A *cubemap layered* texture is a layered texture whose layers are cubemaps of same dimension.

A cubemap layered texture is addressed using an integer index and three floating-point texture coordinates; the index denotes a cubemap within the sequence and the coordinates address a texel within that cubemap.

A cubemap layered texture can only be a CUDA array by calling `cudaMalloc3DArray()` with the `cudaArrayLayered` and `cudaArrayCubemap` flags.

Cubemap layered textures are fetched using the device function described in [texCubemapLayered()](#texcubemaplayered-object). Texture filtering (see [Texture Fetching](#texture-fetching)) is done only within a layer, not across layers.

Cubemap layered textures are only supported on devices of compute capability 2.0 and higher.