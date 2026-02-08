# 6.2.8.6.3. Use in CUDA Graphs

##### 6.2.8.6.3. Use in CUDA Graphs[](#use-in-cuda-graphs "Permalink to this headline")

Programmatic Dependent Launch can be used in [CUDA Graphs](#cuda-graphs) via [stream capture](#creating-a-graph-using-stream-capture) or
directly via [edge data](#edge-data). To program this feature in a CUDA Graph with edge data, use a `cudaGraphDependencyType`
value of `cudaGraphDependencyTypeProgrammatic` on an edge connecting two kernel nodes. This edge type makes the upstream kernel
visible to a `cudaGridDependencySynchronize()` in the downstream kernel. This type must be used with an outgoing port of
either `cudaGraphKernelNodePortLaunchCompletion` or `cudaGraphKernelNodePortProgrammatic`.

The resulting graph equivalents for stream capture are as follows:

| Stream code (abbreviated) | Resulting graph edge |
| --- | --- |
| ``` cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization; attribute.val.programmaticStreamSerializationAllowed = 1; ``` | ``` cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; ``` |
| ``` cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 0; ``` | ``` cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; ``` |
| ``` cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 1; ``` | ``` cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion; ``` |