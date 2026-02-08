# 13.6.1.1.1. Parent and Child Grids (CDP1)

##### 13.6.1.1.1. Parent and Child Grids (CDP1)[](#parent-and-child-grids-cdp1 "Permalink to this headline")

See [Parent and Child Grids](#parent-and-child-grids-cdp2), above, for CDP2 version of document.

A device thread that configures and launches a new grid belongs to the parent grid, and the grid created by the invocation is a child grid.

The invocation and completion of child grids is properly nested, meaning that the parent grid is not considered complete until all child grids created by its threads have completed. Even if the invoking threads do not explicitly synchronize on the child grids launched, the runtime guarantees an implicit synchronization between the parent and child.

Warning

Explicit synchronization with child kernels from a parent block (i.e. using `cudaDeviceSynchronize()` in device code) is deprecated in CUDA 11.6, removed for compute\_90+ compilation, and is slated for full removal in a future CUDA release.

[![The GPU Devotes More Transistors to Data Processing](_images/parent-child-launch-nesting.png)](_images/parent-child-launch-nesting.png)


Figure 31 Parent-Child Launch Nesting[](#parent-child-launch-nesting "Permalink to this image")