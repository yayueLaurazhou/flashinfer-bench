# enum cudaD3D10MapFlags

CUDA D3D10 Map Flags

##### Values

**cudaD3D10MapFlagsNone = 0**
Default; Assume resource can be read/written
**cudaD3D10MapFlagsReadOnly = 1**
CUDA kernels will not write to this resource
**cudaD3D10MapFlagsWriteDiscard = 2**
CUDA kernels will only write to and will not read from this resource