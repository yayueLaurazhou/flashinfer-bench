# 6.2.14.2.2. Cubemap Surfaces

##### 6.2.14.2.2. Cubemap Surfaces[](#cubemap-surfaces "Permalink to this headline")

Cubemap surfaces are accessed using`surfCubemapread()` and `surfCubemapwrite()` ([surfCubemapread()](#surfcubemapread-object) and [surfCubemapwrite()](#surfcubemapwrite-object)) as a two-dimensional layered surface, i.e., using an integer index denoting a face and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in [Table 6](#cubemap-textures-cubemap-fetch).