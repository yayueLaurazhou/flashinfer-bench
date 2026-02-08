# #define CU_MEMHOSTREGISTER_IOMEMORY 0x04

If set, the passed memory pointer is treated as pointing to some memory-mapped I/O space,
e.g. belonging to a third-party PCIe device. On Windows the flag is a no-op. On Linux that
memory is marked as non cache-coherent for the GPU and is expected to be physically
contiguous. It may return CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions. On all other platforms, it is not
supported and CUDA_ERROR_NOT_SUPPORTED is returned. Flag for cuMemHostRegister()