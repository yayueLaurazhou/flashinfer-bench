# enum cudaGraphDebugDotFlags

CUDA Graph debug write options

##### Values

**cudaGraphDebugDotFlagsVerbose = 1<<0**
Output all debug data as if every debug flag is enabled
**cudaGraphDebugDotFlagsKernelNodeParams = 1<<2**
Adds cudaKernelNodeParams to output
**cudaGraphDebugDotFlagsMemcpyNodeParams = 1<<3**
Adds cudaMemcpy3DParms to output
**cudaGraphDebugDotFlagsMemsetNodeParams = 1<<4**
Adds cudaMemsetParams to output
**cudaGraphDebugDotFlagsHostNodeParams = 1<<5**
Adds cudaHostNodeParams to output
**cudaGraphDebugDotFlagsEventNodeParams = 1<<6**
Adds cudaEvent_t handle from record and wait nodes to output
**cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7**
Adds cudaExternalSemaphoreSignalNodeParams values to output
**cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 1<<8**
Adds cudaExternalSemaphoreWaitNodeParams to output
**cudaGraphDebugDotFlagsKernelNodeAttributes = 1<<9**
Adds cudaKernelNodeAttrID values to output


CUDA Runtime API vRelease Version  |  558


Modules


**cudaGraphDebugDotFlagsHandles = 1<<10**
Adds node handles and every kernel function handle to output
**cudaGraphDebugDotFlagsConditionalNodeParams = 1<<15**
Adds cudaConditionalNodeParams to output