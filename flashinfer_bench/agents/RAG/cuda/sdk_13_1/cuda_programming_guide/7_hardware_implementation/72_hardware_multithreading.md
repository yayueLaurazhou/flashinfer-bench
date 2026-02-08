# 7.2. Hardware Multithreading

## 7.2. Hardware Multithreading[](#hardware-multithreading "Permalink to this headline")

The execution context (program counters, registers, and so on) for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp. Therefore, switching from one execution context to another has no cost, and at every instruction issue time, a warp scheduler selects a warp that has threads ready to execute its next instruction (the [active threads](#simt-architecture-notes) of the warp) and issues the instruction to those threads.

In particular, each multiprocessor has a set of 32-bit registers that are partitioned among the warps, and a *parallel data cache* or *shared memory* that is partitioned among the thread blocks.

The number of blocks and warps that can reside and be processed together on the multiprocessor for a given kernel depends on the amount of registers and shared memory used by the kernel and the amount of registers and shared memory available on the multiprocessor. There are also a maximum number of resident blocks and a maximum number of resident warps per multiprocessor. These limits as well the amount of registers and shared memory available on the multiprocessor are a function of the compute capability of the device and are given in [Compute Capabilities](#compute-capabilities). If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.

The total number of warps in a block is as follows:

\(\text{ceil}\left( \frac{T}{W\_{size}},1 \right)\)

* *T* is the number of threads per block,
* *Wsize* is the warp size, which is equal to 32,
* ceil(x, y) is equal to x rounded up to the nearest multiple of y.

The total number of registers and total amount of shared memory allocated for a block are documented in the CUDA Occupancy Calculator provided in the CUDA Toolkit.

[2](#id126)
:   The term *warp-synchronous* refers to code that implicitly assumes threads in the same warp are synchronized at every instruction.