# 16.4.1. Address Reuse within a Graph

### 16.4.1. Address Reuse within a Graph[](#address-reuse-within-a-graph "Permalink to this headline")

CUDA may reuse memory within a graph by assigning the same virtual address ranges to different allocations whose lifetimes do not overlap. Since virtual addresses may be reused, pointers to different allocations with disjoint lifetimes are not guaranteed to be unique.

The following figure shows adding a new allocation node (2) that can reuse the address freed by a dependent node (1).

![Adding New Alloc Node 2](_images/new-alloc-node.png)


Figure 33 Adding New Alloc Node 2[](#id476 "Permalink to this image")

The following figure shows adding a new alloc node (4). The new alloc node is not dependent on the free node (2) so cannot reuse the address from the associated alloc node (2). If the alloc node (2) used the address freed by free node (1), the new alloc node 3 would need a new address.


![Adding New Alloc Node 3](_images/adding-new-alloc-nodes.png)


Figure 34 Adding New Alloc Node 3[](#id477 "Permalink to this image")