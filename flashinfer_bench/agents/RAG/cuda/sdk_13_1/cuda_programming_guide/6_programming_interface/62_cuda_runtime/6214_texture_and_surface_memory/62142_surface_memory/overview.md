# 6.2.14.2. Surface Memory

#### 6.2.14.2. Surface Memory[](#surface-memory "Permalink to this headline")

For devices of compute capability 2.0 and higher, a CUDA array (described in [Cubemap Surfaces](#cubemap-surfaces)), created with the `cudaArraySurfaceLoadStore` flag, can be read and written via a *surface object* using the functions described in [Surface Functions](#surface-functions).

[Table 27](#features-and-technical-specifications-technical-specifications-per-compute-capability) lists the maximum surface width, height, and depth depending on the compute capability of the device.