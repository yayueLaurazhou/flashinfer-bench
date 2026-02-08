# 24.3.1. Unified memory on devices with only CUDA Managed Memory support

### 24.3.1. Unified memory on devices with only CUDA Managed Memory support[](#unified-memory-on-devices-with-only-cuda-managed-memory-support "Permalink to this headline")

For devices with compute capability 6.x or higher but without [pageable memory access](#um-requirements),
CUDA Managed Memory is fully supported and coherent.
The programming model and performance tuning of unified memory is largely similar
to the model as described in
[Unified memory on devices with full CUDA Unified Memory support](#um-pageable-systems),
with the notable exception that system allocators cannot be used to allocate memory.
Thus, the following list of sub-sections do not apply:

* [System-Allocated Memory: in-depth examples](#um-system-allocator)
* [Hardware/Software Coherency](#um-hw-coherency)