# 6.2.14.2.3. Cubemap Layered Surfaces

##### 6.2.14.2.3. Cubemap Layered Surfaces[](#cubemap-layered-surfaces "Permalink to this headline")

Cubemap layered surfaces are accessed using `surfCubemapLayeredread()` and `surfCubemapLayeredwrite()` ([surfCubemapLayeredread()](#surfcubemaplayeredread-object) and [surfCubemapLayeredwrite()](#surfcubemaplayeredwrite-object)) as a two-dimensional layered surface, i.e., using an integer index denoting a face of one of the cubemaps and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in [Table 6](#cubemap-textures-cubemap-fetch), so index ((2 \* 6) + 3), for example, accesses the fourth face of the third cubemap.