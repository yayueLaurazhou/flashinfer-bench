# 6.2.8.7.1. Graph Structure

##### 6.2.8.7.1. Graph Structure[](#graph-structure "Permalink to this headline")

An operation forms a node in a graph. The dependencies between the operations are the edges. These dependencies constrain the execution sequence of the operations.

An operation may be scheduled at any time once the nodes on which it depends are complete. Scheduling is left up to the CUDA system.