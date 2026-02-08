# 6.2.8.1. Concurrent Execution between Host and Device

#### 6.2.8.1. Concurrent Execution between Host and Device[](#concurrent-execution-between-host-and-device "Permalink to this headline")

Concurrent host execution is facilitated through asynchronous library functions that return control to the host thread before the device completes the requested task. Using asynchronous calls, many device operations can be queued up together to be executed by the CUDA driver when appropriate device resources are available. This relieves the host thread of much of the responsibility to manage the device, leaving it free for other tasks. The following device operations are asynchronous with respect to the host:

* Kernel launches;
* Memory copies within a single device’s memory;
* Memory copies from host to device of a memory block of 64 KB or less;
* Memory copies performed by functions that are suffixed with `Async`;
* Memory set function calls.

Programmers can globally disable asynchronicity of kernel launches for all CUDA applications running on a system by setting the `CUDA_LAUNCH_BLOCKING` environment variable to 1. This feature is provided for debugging purposes only and should not be used as a way to make production software run reliably.

Kernel launches are synchronous if hardware counters are collected via a profiler (Nsight Compute) unless concurrent kernel profiling is enabled. `Async` memory copies might also be synchronous if they involve host memory that is not page-locked.