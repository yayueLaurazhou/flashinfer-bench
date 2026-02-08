# enum CUgraphExecUpdateResult

CUDA Graph Update error types

###### Values

**CU_GRAPH_EXEC_UPDATE_SUCCESS = 0x0**
The update succeeded
**CU_GRAPH_EXEC_UPDATE_ERROR = 0x1**
The update failed for an unexpected reason which is described in the return value of the function
**CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 0x2**
The update failed because the topology changed
**CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 0x3**
The update failed because a node type changed
**CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 0x4**
The update failed because the function of a kernel node changed (CUDA driver < 11.2)
**CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 0x5**
The update failed because the parameters changed in a way that is not supported
**CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 0x6**
The update failed because something about the node is not supported
**CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 0x7**
The update failed because the function of a kernel node changed in an unsupported way
**CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 0x8**
The update failed because the node attributes changed in a way that is not supported


CUDA Driver API TRM-06703-001 _vRelease Version  |  48


Modules