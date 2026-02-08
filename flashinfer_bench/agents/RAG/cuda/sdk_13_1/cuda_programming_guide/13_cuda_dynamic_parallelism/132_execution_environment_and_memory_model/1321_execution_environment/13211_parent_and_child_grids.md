# 13.2.1.1. Parent and Child Grids

#### 13.2.1.1. Parent and Child Grids[](#parent-and-child-grids "Permalink to this headline")

A device thread that configures and launches a new grid belongs to the parent grid, and the grid created by the invocation is a child grid.

The invocation and completion of child grids is properly nested, meaning that the parent grid is not considered complete until all child grids created by its threads have completed, and the runtime guarantees an implicit synchronization between the parent and child.

![Parent-Child Launch Nesting](_images/parent-child-launch-nesting.png)


Figure 30 Parent-Child Launch Nesting[](#parent-child-launch-nesting-figure "Permalink to this image")