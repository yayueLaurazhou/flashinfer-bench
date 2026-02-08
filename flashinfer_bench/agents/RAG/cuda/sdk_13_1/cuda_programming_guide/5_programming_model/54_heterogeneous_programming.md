# 5.4. Heterogeneous Programming

## 5.4. Heterogeneous Programming[](#heterogeneous-programming "Permalink to this headline")

As illustrated by [Figure 7](#heterogeneous-programming-heterogeneous-programming), the CUDA programming model assumes that the CUDA threads execute on a physically separate *device* that operates as a coprocessor to the *host* running the C++ program. This is the case, for example, when the kernels execute on a GPU and the rest of the C++ program executes on a CPU.

The CUDA programming model also assumes that both the host and the device maintain their own separate memory spaces in DRAM, referred to as *host memory* and *device memory*, respectively. Therefore, a program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime (described in [Programming Interface](#programming-interface)). This includes device memory allocation and deallocation as well as data transfer between host and device memory.

Unified Memory provides *managed memory* to bridge the host and device memory spaces. Managed memory is accessible from all CPUs and GPUs in the system as a single, coherent memory image with a common address space. This capability enables oversubscription of device memory and can greatly simplify the task of porting applications by eliminating the need to explicitly mirror data on host and device. See [Unified Memory Programming](#um-unified-memory-programming-hd) for an introduction to Unified Memory.

![Heterogeneous Programming](_images/heterogeneous-programming.png)


Figure 7 Heterogeneous Programming[](#heterogeneous-programming-heterogeneous-programming "Permalink to this image")

Note

Serial code executes on the host while parallel code executes on the device.