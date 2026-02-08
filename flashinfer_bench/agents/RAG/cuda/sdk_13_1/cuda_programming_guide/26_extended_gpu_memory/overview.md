# 26. Extended GPU Memory

# 26. Extended GPU Memory[](#extended-gpu-memory "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

The Extended GPU Memory (EGM) feature, utilizing the high-bandwidth
NVLink-C2C, facilitates efficient access to all system memory by GPUs,
in a single-node system.
EGM applies to integrated CPU-GPU NVIDIA systems by allowing physical memory
allocation that can be accessed from any GPU
thread within the setup. EGM ensures that all GPUs can access
its resources at the speed of either GPU-GPU NVLink or NVLink-C2C.

[![EGM](_images/egm-c2c-intro.png)](_images/egm-c2c-intro.png)

In this setup, memory accesses occur via the local high-bandwidth
NVLink-C2C. For remote memory accesses,
GPU NVLink and, in some cases, NVLink-C2C are used. With EGM, GPU
threads gain the capability to access all available memory resources,
including CPU attached memory and HBM3, over the NVSwitch fabric.