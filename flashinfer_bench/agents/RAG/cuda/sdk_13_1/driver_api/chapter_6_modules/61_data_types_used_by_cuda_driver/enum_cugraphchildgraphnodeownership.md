# enum CUgraphChildGraphNodeOwnership

Child graph node ownership

###### Values

**CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE = 0**
Default behavior for a child graph node. Child graph is cloned into the parent and memory
allocation/free nodes can't be present in the child graph.
**CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE = 1**
The child graph is moved to the parent. The handle to the child graph is owned by the parent and
will be destroyed when the parent is destroyed.The following restrictions apply to child graphs after
they have been moved: Cannot be independently instantiated or destroyed; Cannot be added as a
child graph of a separate parent graph; Cannot be used as an argument to cuGraphExecUpdate;
Cannot have additional memory allocation or free nodes added.