# 24.1.1. System Requirements for Unified Memory

### 24.1.1. System Requirements for Unified Memory[](#system-requirements-for-unified-memory "Permalink to this headline")

The following table shows the different levels of support for CUDA Unified Memory,
the device properties required to detect these levels of support
and links to the documentation specific to each level of support:

Table 31 Overview of levels of unified memory support[](#table-unified-memory-levels "Permalink to this table")





| Unified Memory Support Level | System device properties | Further documentation |
| --- | --- | --- |
| Full CUDA Unified Memory: all memory has full support. This includes System-Allocated and CUDA Managed Memory. | Set to 1: `pageableMemoryAccess`  [Systems with hardware acceleration](#um-system-allocator) also have the following properties set to 1:  `hostNativeAtomicSupported`, `pageableMemoryAccessUsesHostPageTables`, `directManagedMemAccessFromHost` | [Unified Memory on devices with full CUDA Unified Memory support](#um-pageable-systems) |
| Only CUDA Managed Memory has full support. | Set to 1: `concurrentManagedAccess`  Set to 0: `pageableMemoryAccess` | [Unified Memory on devices with only CUDA Managed Memory support](#um-cc60) |
| CUDA Managed Memory without full support: unified addressing but no concurrent access. | Set to 1: `managedMemory`  Set to 0: `concurrentManagedAccess` | [Unified Memory on Windows or devices with compute capability 5.x](#um-legacy-devices)  [CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)  [Unified Memory on Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#effective-usage-of-unified-memory-on-tegra) |
| No Unified Memory support. | Set to 0: `managedMemory` | [CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management) |

The behavior of an application that attempts to use Unified Memory on a system that does not support it is undefined.
The following properties enable CUDA applications to check the level of system support for Unified Memory, and
to be portable between systems with different levels of support:

* `pageableMemoryAccess`: This property is set to 1 on systems with CUDA Unified Memory support where
  all threads may access System-Allocated Memory and CUDA Managed Memory.
  These systems include NVIDIA Grace Hopper, IBM Power9 + Volta, and modern Linux systems with HMM enabled (see next bullet), among others.

  + Linux HMM requires Linux kernel version 6.1.24+, 6.2.11+ or 6.3+,
    devices with compute capability 7.5 or higher and
    a CUDA driver version 535+ installed with
    [Open Kernel Modules](http://download.nvidia.com/XFree86/Linux-x86_64/515.43.04/README/kernel_open.html).
* `concurrentManagedAccess`: This property is set to 1 on systems with full CUDA Managed Memory support.
  When this property is set to 0, there is only partial support for Unified Memory in CUDA Managed Memory.
  For Tegra support of Unified Memory, see
  [CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management).

A program may query the level of GPU support for CUDA Unified Memory, by querying the attributes in
[Table 31](#table-unified-memory-levels) using `cudaGetDeviceProperties()`.