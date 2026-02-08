# 9.7.16.1.1. Tensor Memory Addressing

##### 9.7.16.1.1. [Tensor Memory Addressing](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing "Permalink to this headline")

Tensor Memory addresses are 32-bit wide and specify two components.

1. Lane index
2. Column index

The layout is as follows:

> |  |  |
> | --- | --- |
> | 31 16 | 15 0 |
> | Lane index | Column index |

[Figure 182](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-layout) shows the view of the Tensor Memory Layout within CTA.

![_images/tensor-memory-layout.png](./ptx_files/tensor-memory-layout.png)


Figure 182 Tensor Memory Layout and Addressing[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-layout "Permalink to this image")