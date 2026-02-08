# 9.7.16.9.1.2. Decompression of 6-bit floating point to 8-bit type

###### 9.7.16.9.1.2. [Decompression of 6-bit floating point to 8-bit type](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-optional-decompression-6bit-8bit)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-optional-decompression-6bit-8bit "Permalink to this headline")

A contiguous set of 16 elements of 6-bits each followed by 4 bytes of padding is
decompressed into 16 elements of 8-bits each as shown in [Figure 196](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-6b8b).

![_images/tcgen05-decompression-6b8b.png](./ptx_files/tcgen05-decompression-6b8b.png)


Figure 196 Decompression from 6-bit to 8-bit[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-6b8b "Permalink to this image")

The individual 6-bit to 8-bit decompression for types `E3M2` and `E2M3` is shown in
[Figure 197](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-6b8b-individual1) and [Figure 198](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-6b8b-individual2)
respectively.

![_images/tcgen05-decompression-6b8b-individual1.png](./ptx_files/tcgen05-decompression-6b8b-individual1.png)


Figure 197 Individual decompression from 6-bit to 8-bit for E3M2 type[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-6b8b-individual1 "Permalink to this image")


![_images/tcgen05-decompression-6b8b-individual2.png](./ptx_files/tcgen05-decompression-6b8b-individual2.png)


Figure 198 Individual decompression from 6-bit to 8-bit for E2M3 type[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-decompression-6b8b-individual2 "Permalink to this image")