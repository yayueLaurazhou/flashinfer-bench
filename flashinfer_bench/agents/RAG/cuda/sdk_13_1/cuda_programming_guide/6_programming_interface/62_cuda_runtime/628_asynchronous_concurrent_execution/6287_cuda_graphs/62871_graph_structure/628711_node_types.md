# 6.2.8.7.1.1. Node Types

###### 6.2.8.7.1.1. Node Types[](#node-types "Permalink to this headline")

A graph node can be one of:

* kernel
* CPU function call
* memory copy
* memset
* empty node
* waiting on an [event](#events)
* recording an [event](#events)
* signalling an [external semaphore](#external-resource-interoperability)
* waiting on an [external semaphore](#external-resource-interoperability)
* [conditional node](#conditional-graph-nodes)
* [Graph Memory Nodes](#graph-memory-nodes)
* child graph: To execute a separate nested graph, as shown in the following figure.

[![Child Graph Example](_images/child-graph.png)](_images/child-graph.png)


Figure 13 Child Graph Example[](#node-types-fig-child-graph "Permalink to this image")