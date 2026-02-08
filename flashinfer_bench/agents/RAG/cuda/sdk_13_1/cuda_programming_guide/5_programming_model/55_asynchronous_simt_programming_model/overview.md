# 5.5. Asynchronous SIMT Programming Model

## 5.5. Asynchronous SIMT Programming Model[](#asynchronous-simt-programming-model "Permalink to this headline")

In the CUDA programming model a thread is the lowest level of abstraction for doing a computation or a memory operation. Starting with devices based on the **NVIDIA Ampere GPU Architecture**, the CUDA programming model provides acceleration to memory operations via the asynchronous programming model. The asynchronous programming model defines the behavior of asynchronous operations with respect to CUDA threads.

The asynchronous programming model defines the behavior of [Asynchronous Barrier](#aw-barrier) for synchronization between CUDA threads. The model also explains and defines how [cuda::memcpy\_async](#asynchronous-data-copies) can be used to move data asynchronously from global memory while computing in the GPU.