# 2.2. Thread Hierarchy

## 2.2. [Thread Hierarchy](https://docs.nvidia.com/cuda/parallel-thread-execution/#thread-hierarchy)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#thread-hierarchy "Permalink to this headline")

The batch of threads that executes a kernel is organized as a grid. A grid consists of either
cooperative thread arrays or clusters of cooperative thread arrays as described in this section and
illustrated in [Figure 1](https://docs.nvidia.com/cuda/parallel-thread-execution/#grid-of-clusters-grid-with-ctas) and
[Figure 2](https://docs.nvidia.com/cuda/parallel-thread-execution/#grid-of-clusters-grid-with-clusters). *Cooperative thread arrays (CTAs)* implement CUDA
thread blocks and clusters implement CUDA thread block clusters.