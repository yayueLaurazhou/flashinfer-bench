# enum CUgraphInstantiateResult

Graph instantiation results


CUDA Driver API TRM-06703-001 _vRelease Version  |  49


Modules

###### Values

**CUDA_GRAPH_INSTANTIATE_SUCCESS = 0**
Instantiation succeeded
**CUDA_GRAPH_INSTANTIATE_ERROR = 1**
Instantiation failed for an unexpected reason which is described in the return value of the function
**CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2**
Instantiation failed due to invalid structure, such as cycles
**CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3**
Instantiation for device launch failed because the graph contained an unsupported operation
**CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4**
Instantiation for device launch failed due to the nodes belonging to different contexts
**CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED = 5**
One or more conditional handles are not associated with conditional nodes