# 9.7.16.10.5.3. Layout C (M = 128 + cta-group::2 + Sparse A matrix)

###### 9.7.16.10.5.3. [Layout C (M = 128 + cta-group::2 + Sparse A matrix)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c "Permalink to this headline")

Layout organization for M = 128 + .cta\_group::2 + Sparse A matrix is shown in
[Figure 209](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c1).

![_images/tcgen05-data-path-layout-c1.png](./ptx_files/tcgen05-data-path-layout-c1.png)


Figure 209 Layout organization for M = 128 + .cta\_group::2 + Sparse A matrix[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 210](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c2)

![_images/tcgen05-data-path-layout-c2.png](./ptx_files/tcgen05-data-path-layout-c2.png)


Figure 210 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c2 "Permalink to this image")