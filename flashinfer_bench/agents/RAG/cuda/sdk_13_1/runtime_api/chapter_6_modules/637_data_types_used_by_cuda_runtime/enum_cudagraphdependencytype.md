# enum cudaGraphDependencyType

Type annotations that can be applied to graph edges as part of cudaGraphEdgeData.

##### Values

**cudaGraphDependencyTypeDefault = 0**
This is an ordinary dependency.
**cudaGraphDependencyTypeProgrammatic = 1**
This dependency type allows the downstream node to use
cudaGridDependencySynchronize() . It may only be used between kernel
nodes, and must be used with either the cudaGraphKernelNodePortProgrammatic or
cudaGraphKernelNodePortLaunchCompletion outgoing port.