# enum cudaGraphicsRegisterFlags

CUDA graphics interop register flags

##### Values

**cudaGraphicsRegisterFlagsNone = 0**
Default
**cudaGraphicsRegisterFlagsReadOnly = 1**
CUDA will not write to this resource
**cudaGraphicsRegisterFlagsWriteDiscard = 2**
CUDA will only write to and will not read from this resource
**cudaGraphicsRegisterFlagsSurfaceLoadStore = 4**


CUDA Runtime API vRelease Version  |  560


Modules


CUDA will bind this resource to a surface reference
**cudaGraphicsRegisterFlagsTextureGather = 8**
CUDA will perform texture gather operations on this resource