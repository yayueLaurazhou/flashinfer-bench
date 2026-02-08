# 9.7.16.9.1.1. Decompression of 4-bit floating point to 8-bit type

###### 9.7.16.9.1.1. [Decompression of 4-bit floating point to 8-bit type](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-optional-decompression-4bit-8bit)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-optional-decompression-4bit-8bit "Permalink to this headline")

A contiguous set of 16 elements of 4-bits each followed by 8 bytes of padding can be converted
into 16 elements of 8-bits each as shown in [Figure 194](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-4b8b).

![_images/tcgen05-decompression-4b8b.png](./ptx_files/tcgen05-decompression-4b8b.png)


Figure 194 Decompression from 4-bit to 8-bit[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-4b8b "Permalink to this image")

The individual 4-bit to 8-bit decompression would look like as shown in [Figure 195](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-4b8b-individual).

![_images/tcgen05-decompression-4b8b-individual.png](./ptx_files/tcgen05-decompression-4b8b-individual.png)


Figure 195 Individual decompression from 4-bit to 8-bit[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-4b8b-individual "Permalink to this image")