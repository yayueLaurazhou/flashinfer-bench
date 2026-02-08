# 14.8. Fabric Memory

## 14.8. Fabric Memory[](#fabric-memory "Permalink to this headline")

CUDA 12.4 introduced a new VMM allocation handle type `CU_MEM_HANDLE_TYPE_FABRIC`. On supported platforms and provided the NVIDIA IMEX daemon
is running this allocation handle type enables sharing allocations not only intra node with any communication mechanism, e.g. MPI, but also inter
node. This allows GPUs in a Multi Node NVLINK System to map the memory of all other GPUs part of the same NVLINK fabric even if they are in
different nodes greatly increasing the scale of multi-GPU Programming with NVLINK.