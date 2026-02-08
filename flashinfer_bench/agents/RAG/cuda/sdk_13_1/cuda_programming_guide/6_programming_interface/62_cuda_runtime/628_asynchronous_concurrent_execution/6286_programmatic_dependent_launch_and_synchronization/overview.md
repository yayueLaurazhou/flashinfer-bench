# 6.2.8.6. Programmatic Dependent Launch and Synchronization

#### 6.2.8.6. Programmatic Dependent Launch and Synchronization[](#programmatic-dependent-launch-and-synchronization "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

The *Programmatic Dependent Launch* mechanism allows for a dependent *secondary* kernel
to launch before the *primary* kernel it depends on in the same CUDA stream has finished executing.
Available starting with devices of compute capability 9.0, this technique can provide performance
benefits when the *secondary* kernel can complete significant work that does not depend on the results of the *primary* kernel.