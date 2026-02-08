# 9.7.16.10.5.2. Layout B (M = 128 + cta-group::2 + Dense A matrix)

###### 9.7.16.10.5.2. [Layout B (M = 128 + cta-group::2 + Dense A matrix)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b "Permalink to this headline")

Layout organization for M = 128 + .cta\_group::2 + Dense A matrix is shown in
[Figure 207](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b1).

![_images/tcgen05-data-path-layout-b1.png](./ptx_files/tcgen05-data-path-layout-b1.png)


Figure 207 Layout organization for M = 128 + .cta\_group::2 + Dense A matrix[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 208](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b2)

![_images/tcgen05-data-path-layout-b2.png](./ptx_files/tcgen05-data-path-layout-b2.png)


Figure 208 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b2 "Permalink to this image")