# 19. Texture Fetching

# 19. Texture Fetching[](#texture-fetching "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

This section gives the formula used to compute the value returned by the texture functions of [Texture Functions](#texture-functions) depending on the various attributes of the texture object (see [Texture and Surface Memory](#texture-and-surface-memory)).

The texture bound to the texture object is represented as an array *T* of

* *N* texels for a one-dimensional texture,
* *N x M* texels for a two-dimensional texture,
* *N x M x L* texels for a three-dimensional texture.

It is fetched using non-normalized texture coordinates *x*, *y*, and *z*, or the normalized texture coordinates *x/N*, *y/M*, and *z/L* as described in [Texture Memory](#texture-memory). In this section, the coordinates are assumed to be in the valid range. [Texture Memory](#texture-memory) explained how out-of-range coordinates are remapped to the valid range based on the addressing mode.