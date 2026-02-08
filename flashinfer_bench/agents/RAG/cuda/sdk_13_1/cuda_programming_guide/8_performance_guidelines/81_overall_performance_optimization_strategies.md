# 8.1. Overall Performance Optimization Strategies

## 8.1. Overall Performance Optimization Strategies[](#overall-performance-optimization-strategies "Permalink to this headline")

Performance optimization revolves around four basic strategies:

* Maximize parallel execution to achieve maximum utilization;
* Optimize memory usage to achieve maximum memory throughput;
* Optimize instruction usage to achieve maximum instruction throughput;
* Minimize memory thrashing.

Which strategies will yield the best performance gain for a particular portion of an application depends on the performance limiters for that portion; optimizing instruction usage of a kernel that is mostly limited by memory accesses will not yield any significant performance gain, for example. Optimization efforts should therefore be constantly directed by measuring and monitoring the performance limiters, for example using the CUDA profiler. Also, comparing the floating-point operation throughput or memory throughput—whichever makes more sense—of a particular kernel to the corresponding peak theoretical throughput of the device indicates how much room for improvement there is for the kernel.