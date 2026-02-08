# enum cudaGraphInstantiateResult

Graph instantiation results

##### Values

**cudaGraphInstantiateSuccess = 0**
Instantiation succeeded
**cudaGraphInstantiateError = 1**
Instantiation failed for an unexpected reason which is described in the return value of the function
**cudaGraphInstantiateInvalidStructure = 2**
Instantiation failed due to invalid structure, such as cycles
**cudaGraphInstantiateNodeOperationNotSupported = 3**
Instantiation for device launch failed because the graph contained an unsupported operation
**cudaGraphInstantiateMultipleDevicesNotSupported = 4**
Instantiation for device launch failed due to the nodes belonging to different contexts
**cudaGraphInstantiateConditionalHandleUnused = 5**
One or more conditional handles are not associated with conditional nodes


CUDA Runtime API vRelease Version  |  561


Modules