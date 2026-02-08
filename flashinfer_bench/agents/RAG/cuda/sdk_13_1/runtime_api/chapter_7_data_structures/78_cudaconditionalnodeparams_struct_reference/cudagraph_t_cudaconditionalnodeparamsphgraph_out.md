# cudaGraph_t *cudaConditionalNodeParams::phGraph_out

CUDA-owned array populated with conditional node child graphs during creation of the node. Valid
for the lifetime of the conditional node. The contents of the graph(s) are subject to the following
constraints:

Allowed node types are kernel nodes, empty nodes, child graphs, memsets, memcopies, and

##### **‣**

conditionals. This applies recursively to child graphs and conditional bodies.
All kernels, including kernels in nested conditionals or child graphs at any level, must belong to the

##### **‣**

same CUDA context.

These graphs may be populated using graph node creation APIs or cudaStreamBeginCaptureToGraph.
cudaGraphCondTypeIf: phGraph_out[0] is executed when the condition is non-zero. If size
== 2, phGraph_out[1] will be executed when the condition is zero. cudaGraphCondTypeWhile:
phGraph_out[0] is executed as long as the condition is non-zero. cudaGraphCondTypeSwitch:
phGraph_out[n] is executed when the condition is equal to n. If the condition >= size, no body graph
is executed.