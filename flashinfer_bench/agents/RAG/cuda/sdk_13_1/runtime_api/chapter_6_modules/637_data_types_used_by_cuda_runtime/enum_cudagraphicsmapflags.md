# enum cudaGraphicsMapFlags

CUDA graphics interop map flags

##### Values

**cudaGraphicsMapFlagsNone = 0**
Default; Assume resource can be read/written
**cudaGraphicsMapFlagsReadOnly = 1**
CUDA will not write to this resource
**cudaGraphicsMapFlagsWriteDiscard = 2**
CUDA will only write to and will not read from this resource