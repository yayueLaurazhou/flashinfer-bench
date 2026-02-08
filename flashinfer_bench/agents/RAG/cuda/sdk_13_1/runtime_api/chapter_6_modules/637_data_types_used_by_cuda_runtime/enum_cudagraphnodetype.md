# enum cudaGraphNodeType

CUDA Graph node types

##### Values

**cudaGraphNodeTypeKernel = 0x00**
GPU kernel node
**cudaGraphNodeTypeMemcpy = 0x01**
Memcpy node
**cudaGraphNodeTypeMemset = 0x02**
Memset node


CUDA Runtime API vRelease Version  |  562


Modules


**cudaGraphNodeTypeHost = 0x03**
Host (executable) node
**cudaGraphNodeTypeGraph = 0x04**
Node which executes an embedded graph
**cudaGraphNodeTypeEmpty = 0x05**
Empty (no-op) node
**cudaGraphNodeTypeWaitEvent = 0x06**
External event wait node
**cudaGraphNodeTypeEventRecord = 0x07**
External event record node
**cudaGraphNodeTypeExtSemaphoreSignal = 0x08**
External semaphore signal node
**cudaGraphNodeTypeExtSemaphoreWait = 0x09**
External semaphore wait node
**cudaGraphNodeTypeMemAlloc = 0x0a**
Memory allocation node
**cudaGraphNodeTypeMemFree = 0x0b**
Memory free node
**cudaGraphNodeTypeConditional = 0x0d**
Conditional nodeMay be used to implement a conditional execution path or loop inside of a graph.
The graph(s) contained within the body of the conditional node can be selectively executed or
iterated upon based on the value of a conditional variable.Handles must be created in advance
of creating the node using cudaGraphConditionalHandleCreate.The following restrictions apply
to graphs which contain conditional nodes: The graph cannot be used in a child node. Only one
instantiation of the graph may exist at any point in time. The graph cannot be cloned.To set the
control value, supply a default value when creating the handle and/or call cudaGraphSetConditional
from device code.
**cudaGraphNodeTypeCount**