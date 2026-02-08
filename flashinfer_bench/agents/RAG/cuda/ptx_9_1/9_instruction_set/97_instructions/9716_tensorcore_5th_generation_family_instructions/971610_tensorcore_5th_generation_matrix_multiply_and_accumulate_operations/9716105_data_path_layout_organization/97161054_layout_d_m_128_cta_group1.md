# 9.7.16.10.5.4. Layout D (M = 128 + cta-group::1)

###### 9.7.16.10.5.4. [Layout D (M = 128 + cta-group::1)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d "Permalink to this headline")

Layout organization for M = 128 + .cta\_group::1 is shown in
[Figure 211](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d1).

![_images/tcgen05-data-path-layout-d1.png](./ptx_files/tcgen05-data-path-layout-d1.png)


Figure 211 Layout organization for M = 128 + .cta\_group::1[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 212](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d2)

![_images/tcgen05-data-path-layout-d2.png](./ptx_files/tcgen05-data-path-layout-d2.png)


Figure 212 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d2 "Permalink to this image")