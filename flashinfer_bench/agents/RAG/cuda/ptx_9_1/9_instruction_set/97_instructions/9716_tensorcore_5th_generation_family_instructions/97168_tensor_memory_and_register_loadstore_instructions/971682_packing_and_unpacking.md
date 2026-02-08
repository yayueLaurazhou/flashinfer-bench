# 9.7.16.8.2. Packing and Unpacking

##### 9.7.16.8.2. [Packing and Unpacking](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st-packing-unpacking)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st-packing-unpacking "Permalink to this headline")

Optionally, the following pack and unpack operations can be performed during the load and store:

1. Packing: two 16-bit chunks can be packed into a single 32-bit chunk in the register in `tcgen05.ld`
2. Unpacking: a single 32-bit chunk in the register can be unpacked into two 16-bit chunks in `tcgen05.st`

as shown in the [Figure 193](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-ld-st-pack-unpack).

![_images/tcgen05-ld-st-pack-unpack.png](./ptx_files/tcgen05-ld-st-pack-unpack.png)


Figure 193 Pack/Unpack operations for tcgen05 ld/st[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-ld-st-pack-unpack "Permalink to this image")