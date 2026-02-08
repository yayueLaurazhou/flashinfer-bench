# 6.35. Green Contexts

This section describes the APIs for creation and manipulation of green contexts in the CUDA driver.
Green contexts are a lightweight alternative to traditional contexts, that can be used to select a subset of
device resources. This allows the developer to, for example, select SMs from distinct spatial partitions
of the GPU and target them via CUDA stream operations, kernel launches, etc.

Here are the broad initial steps to follow to get started:

(1) Start with an initial set of resources. For SM resources, they can be fetched via