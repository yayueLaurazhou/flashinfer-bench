# 9.7.16.10.5.5. Layout E (M = 64 + .ws mode)

###### 9.7.16.10.5.5. [Layout E (M = 64 + .ws mode)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e "Permalink to this headline")

Layout organization for M = 64 + .ws mode is shown in
[Figure 213](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e1).

![_images/tcgen05-data-path-layout-e1.png](./ptx_files/tcgen05-data-path-layout-e1.png)


Figure 213 Layout organization for M = 64 + .ws mode[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 214](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e2)

![_images/tcgen05-data-path-layout-e2.png](./ptx_files/tcgen05-data-path-layout-e2.png)


Figure 214 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e2 "Permalink to this image")