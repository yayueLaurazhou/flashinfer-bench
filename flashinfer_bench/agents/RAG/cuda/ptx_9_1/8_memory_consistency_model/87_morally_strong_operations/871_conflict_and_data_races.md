# 8.7.1. Conflict and Data-races

### 8.7.1. [Conflict and Data-races](https://docs.nvidia.com/cuda/parallel-thread-execution/#conflict-and-data-races)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#conflict-and-data-races "Permalink to this headline")

Two *overlapping* memory operations are said to *conflict* when at least one of them is a *write*.

Two *conflicting* memory operations are said to be in a *data-race* if they are not related in
*causality order* and they are not *morally strong*.