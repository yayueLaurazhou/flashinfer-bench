# enum cudaD3D9MapFlags

CUDA D3D9 Map Flags

##### Values

**cudaD3D9MapFlagsNone = 0**
Default; Assume resource can be read/written
**cudaD3D9MapFlagsReadOnly = 1**
CUDA kernels will not write to this resource
**cudaD3D9MapFlagsWriteDiscard = 2**
CUDA kernels will only write to and will not read from this resource