# 6.2.6.3. Mapped Memory

#### 6.2.6.3. Mapped Memory[](#mapped-memory "Permalink to this headline")

A block of page-locked host memory can also be mapped into the address space of the device by passing flag `cudaHostAllocMapped` to `cudaHostAlloc()` or by passing flag `cudaHostRegisterMapped` to `cudaHostRegister()`. Such a block has therefore in general two addresses: one in host memory that is returned by `cudaHostAlloc()` or `malloc()`, and one in device memory that can be retrieved using `cudaHostGetDevicePointer()` and then used to access the block from within a kernel. The only exception is for pointers allocated with `cudaHostAlloc()` and when a unified address space is used for the host and the device as mentioned in [Unified Virtual Address Space](#unified-virtual-address-space).

Accessing host memory directly from within a kernel does not provide the same bandwidth as device memory, but does have some advantages:

* There is no need to allocate a block in device memory and copy data between this block and the block in host memory; data transfers are implicitly performed as needed by the kernel;
* There is no need to use streams (see [Concurrent Data Transfers](#concurrent-data-transfers)) to overlap data transfers with kernel execution; the kernel-originated data transfers automatically overlap with kernel execution.

Since mapped page-locked memory is shared between host and device however, the application must synchronize memory accesses using streams or events (see [Asynchronous Concurrent Execution](#asynchronous-concurrent-execution)) to avoid any potential read-after-write, write-after-read, or write-after-write hazards.

To be able to retrieve the device pointer to any mapped page-locked memory, page-locked memory mapping must be enabled by calling `cudaSetDeviceFlags()` with the `cudaDeviceMapHost` flag before any other CUDA call is performed. Otherwise, `cudaHostGetDevicePointer()` will return an error.

`cudaHostGetDevicePointer()` also returns an error if the device does not support mapped page-locked host memory. Applications may query this capability by checking the `canMapHostMemory` device property (see [Device Enumeration](#device-enumeration)), which is equal to 1 for devices that support mapped page-locked host memory.

Note that atomic functions (see [Atomic Functions](#atomic-functions)) operating on mapped page-locked memory are not atomic from the point of view of the host or other devices.

Also note that CUDA runtime requires that 1-byte, 2-byte, 4-byte, 8-byte, and 16-byte naturally aligned
loads and stores to host memory initiated from the device are preserved as single accesses
from the point of view of the host and other devices.
On some platforms, atomics to memory may be broken by the hardware into separate
load and store operations.
These component load and store operations have the same requirements on preservation of naturally aligned accesses.
The CUDA runtime does not support a PCI Express bus topology where a PCI Express bridge splits
8-byte naturally aligned operations and NVIDIA is not aware of any topology that splits
16-byte naturally aligned operations.