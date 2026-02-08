# 10.29. Asynchronous Data Copies using the Tensor Memory Accelerator (TMA)

## 10.29. Asynchronous Data Copies using the Tensor Memory Accelerator (TMA)[](#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma "Permalink to this headline")

Many applications require movement of large amounts of data from and to global
memory. Often, the data is laid out in global memory as a multi-dimensional
array with non-sequential data acess patterns. To reduce global memory usage,
sub-tiles of such arrays are copied to shared memory before use in computations.
The loading and storing involves address-calculations that can be error-prone
and repetitive. To offload these computations, Compute Capability 9.0 introduces the
Tensor Memory Accelerator (TMA). The primary goal of TMA is to provide an efficient
data transfer mechanism from global memory to shared memory for
multi-dimensional arrays.

**Naming**. Tensor memory accelerator (TMA) is a broad term used to refer to the
features described in this section. For the purpose of forward-compatibility and
to reduce discrepancies with the PTX ISA, the text in this section refers to TMA
operations as either bulk-asynchronous copies or bulk tensor asynchronous
copies, depending on the specific type of copy used. The term “bulk” is used to
contrast these operations with the asynchronous memory operations described in
the previous sections.

**Dimensions**. TMA supports copying both one-dimensional and multi-dimensional
arrays (up to 5-dimensional). The programming model for **bulk-asynchronous
copies** of one-dimensional contiguous arrays is different from the programming
model for **bulk tensor asynchronous copies** of multi-dimensional arrays. To
perform a bulk tensor asynchronous copy of a multi-dimensional array, the
hardware requires a [tensor map](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap).
This object describes the layout of the multi-dimensional array in global and
shared memory. A tensor map is typically created on the host using the [cuTensorMapEncode
API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)
and then transferred from host to device as a `const` kernel parameter annotated
with `__grid_constant__`.
The tensor map is transferred from host to device as a `const` kernel
parameter annotated with `__grid_constant__`, and can be used on the device to
copy a tile of data between shared and global memory. In contrast, performing a
bulk-asynchronous copy of a contiguous one-dimensional array does not require a
tensor map: it can be performed on-device with a pointer and size parameter.

**Source and destination**. The source and destination addresses of bulk-asynchronous copy
operations can be in shared or global memory. The operations can read data from global to
shared memory, write data from shared to global memory, and also copy from
shared memory to [Distributed Shared Memory](#distributed-shared-memory) of another block in the same cluster.
In addition, when in a cluster, a bulk-asynchronous operation can be specified as being
multicast. In this case, data can be transferred from global memory to the
shared memory of multiple blocks within the cluster. The multicast feature is
optimized for target architecture `sm_90a` and may have [significantly reduced performance](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) on
other targets. Hence, it is advised to be used with [compute architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)
`sm_90a`.

**Asynchronous**. Data transfers using TMA are [asynchronous](#asynchronous-simt-programming-model). This allows the initiating
thread to continue computing while the hardware asynchronously copies the data.
**Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future**.
There are several [completion mechanisms](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms)
that bulk-asynchronous operations can use to signal that they have completed.
When the operation reads from global to shared memory, any
thread in the block can wait for the data to be readable in shared memory by
waiting on a [Shared Memory Barrier](#aw-barrier). When the bulk-asynchronous
operation writes data from shared memory to global or distributed shared memory,
only the initiating thread can wait for the operation to have completed.
This is accomplished using a *bulk async-group* based completion mechanism. A
table describing the completion mechanisms can be found below and in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk).

Table 8 Asynchronous copies with possible source and destinations memory spaces and completion mechanisms. An empty cell indicates that a source-destination pair is not supported.[](#table-tma-source-dest-state-spaces "Permalink to this table")

| Direction | | Completion mechanism | |
| --- | --- | --- | --- |
| Destination | Source | Asynchronous copy | Bulk-asynchronous copy (TMA) |
| Global | Global |  |  |
| Global | Shared::cta |  | Bulk async-group |
| Shared::cta | Global | Async-group, mbarrier | Mbarrier |
| Shared::cluster | Global |  | Mbarrier (multicast) |
| Shared::cta | Shared::cluster |  | Mbarrier |
| Shared::cta | Shared::cta |  |  |