# enum cudaGraphExecUpdateResult

CUDA Graph Update error types

##### Values

**cudaGraphExecUpdateSuccess = 0x0**
The update succeeded
**cudaGraphExecUpdateError = 0x1**
The update failed for an unexpected reason which is described in the return value of the function
**cudaGraphExecUpdateErrorTopologyChanged = 0x2**
The update failed because the topology changed
**cudaGraphExecUpdateErrorNodeTypeChanged = 0x3**
The update failed because a node type changed
**cudaGraphExecUpdateErrorFunctionChanged = 0x4**
The update failed because the function of a kernel node changed (CUDA driver < 11.2)
**cudaGraphExecUpdateErrorParametersChanged = 0x5**
The update failed because the parameters changed in a way that is not supported
**cudaGraphExecUpdateErrorNotSupported = 0x6**
The update failed because something about the node is not supported
**cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7**
The update failed because the function of a kernel node changed in an unsupported way
**cudaGraphExecUpdateErrorAttributesChanged = 0x8**
The update failed because the node attributes changed in a way that is not supported


CUDA Runtime API vRelease Version  |  559


Modules