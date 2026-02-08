# enum cudaGLMapFlags

CUDA GL Map Flags

##### Values

**cudaGLMapFlagsNone = 0**
Default; Assume resource can be read/written
**cudaGLMapFlagsReadOnly = 1**
CUDA kernels will not write to this resource
**cudaGLMapFlagsWriteDiscard = 2**
CUDA kernels will only write to and will not read from this resource