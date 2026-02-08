# 6.2.14.3. CUDA Arrays

#### 6.2.14.3. CUDA Arrays[](#cuda-arrays "Permalink to this headline")

CUDA arrays are opaque memory layouts optimized for texture fetching. They are one dimensional, two dimensional, or three-dimensional and composed of elements, each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats. CUDA arrays are only accessible by kernels through texture fetching as described in [Texture Memory](#texture-memory) or surface reading and writing as described in [Surface Memory](#surface-memory).