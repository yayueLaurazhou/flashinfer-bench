# 8.3. Maximize Memory Throughput

## 8.3. Maximize Memory Throughput[](#maximize-memory-throughput "Permalink to this headline")

The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth.

That means minimizing data transfers between the host and the device, as detailed in [Data Transfer between Host and Device](#data-transfer-between-host-and-device), since these have much lower bandwidth than data transfers between global memory and the device.

That also means minimizing data transfers between global memory and the device by maximizing use of on-chip memory: shared memory and caches (i.e., L1 cache and L2 cache available on devices of compute capability 2.x and higher, texture cache and constant cache available on all devices).

Shared memory is equivalent to a user-managed cache: The application explicitly allocates and accesses it. As illustrated in [CUDA Runtime](#cuda-c-runtime), a typical programming pattern is to stage data coming from device memory into shared memory; in other words, to have each thread of a block:

* Load data from device memory to shared memory,
* Synchronize with all the other threads of the block so that each thread can safely read shared memory locations that were populated by different threads,
* Process the data in shared memory,
* Synchronize again if necessary to make sure that shared memory has been updated with the results,
* Write the results back to device memory.

For some applications (for example, for which global memory access patterns are data-dependent), a traditional hardware-managed cache is more appropriate to exploit data locality. As mentioned in [Compute Capability 7.x](#compute-capability-7-x), [Compute Capability 8.x](#compute-capability-8-x) and [Compute Capability 9.0](#compute-capability-9-0), for devices of compute capability 7.x, 8.x and 9.0, the same on-chip memory is used for both L1 and shared memory, and how much of it is dedicated to L1 versus shared memory is configurable for each kernel call.

The throughput of memory accesses by a kernel can vary by an order of magnitude depending on access pattern for each type of memory. The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns described in [Device Memory Accesses](#device-memory-accesses). This optimization is especially important for global memory accesses as global memory bandwidth is low compared to available on-chip bandwidths and arithmetic instruction throughput, so non-optimal global memory accesses generally have a high impact on performance.