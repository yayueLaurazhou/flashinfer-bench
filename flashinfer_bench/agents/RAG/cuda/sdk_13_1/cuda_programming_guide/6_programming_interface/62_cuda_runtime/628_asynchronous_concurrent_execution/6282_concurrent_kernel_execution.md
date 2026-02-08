# 6.2.8.2. Concurrent Kernel Execution

#### 6.2.8.2. Concurrent Kernel Execution[](#concurrent-kernel-execution "Permalink to this headline")

Some devices of compute capability 2.x and higher can execute multiple kernels concurrently. Applications may query this capability by checking the `concurrentKernels` device property (see [Device Enumeration](#device-enumeration)), which is equal to 1 for devices that support it.

The maximum number of kernel launches that a device can execute concurrently depends on its compute capability and is listed in [Table 27](#features-and-technical-specifications-technical-specifications-per-compute-capability).

A kernel from one CUDA context cannot execute concurrently with a kernel from another CUDA context. The GPU may time slice to provide forward progress to each context. If a user wants to run kernels from multiple process simultaneously on the SM, one must enable MPS.

Kernels that use many textures or a large amount of local memory are less likely to execute concurrently with other kernels.