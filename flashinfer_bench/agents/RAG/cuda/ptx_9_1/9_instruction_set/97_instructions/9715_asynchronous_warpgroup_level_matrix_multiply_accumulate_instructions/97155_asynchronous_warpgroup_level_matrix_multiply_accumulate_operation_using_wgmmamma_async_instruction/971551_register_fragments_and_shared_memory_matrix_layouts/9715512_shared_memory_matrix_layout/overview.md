# 9.7.15.5.1.2. Shared Memory Matrix Layout

###### 9.7.15.5.1.2. [Shared Memory Matrix Layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout "Permalink to this headline")

If the argument `imm-trans-a` / `imm-trans-b` of the instruction `wgmma.mma_async{.sp}`
is 0, then *K-major* is used for matrix `A` / `B` respectively. If the value of argument
`imm-trans-a` is 1 then *M-major* is used for matrix `A`. If the value of the argument
`imm-trans-b` is 1, then *N-major* is used for matrix `B`.

In a column-major default BLAS library such as cuBLAS, the matrices `A` and `B` with and
without transpose can be classified as either *K-Major* or *M-or-N-Major* as shown in the
following table:

|  | Non-Transposed | Transposed |
| --- | --- | --- |
| A | K-major | M-major |
| B | K-major | N-major |

To avoid confusion with `A`, `B`, `row-major`, `col-major`, `transpose`, and
`non-transpose`, we will use *MN-Major* and *K-Major* throughout this section.

The matrices in the shared memory are made up of one or more “swizzle layout atom”.
The exact layout of these swizzle atoms depends on the swizzling mode, swizzle-atomicity,
and the leading dimension. The layout of the swizzle are shown in
[Table 38](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-swizzle-lead-dim).

Table 38 Various combinations of swizzling mode, leading dimension and swizzle-atom layout[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-swizzle-lead-dim "Permalink to this table")





| Swizzling mode | Leading Dimension / Major-ness | Swizzle atom layout (128b element) |
| --- | --- | --- |
| 128B Swizzling Mode | M/N | 8x8 |
| K | 8x8 |
| 64B Swizzling Mode | M/N | 4x8 |
| K | 8x4 |
| 32B Swizzling Mode | M/N | 2x8 |
| K | 8x2 |
| None | M/N | 1x8 |
| K | 8x1 |

The above shapes are for elements of size 128 bits. For smaller elements sizes, the same
shapes would get multiplied along the leading dimension by a factor of `128/sizeof_bits(Element)`.
For example, 128B MN major swizzle atom would have a shape of `(8*(128/32))x8 = 32x8` for
`tf32` tensor core inputs.

Examples

The following are some example layouts of *MxK* or *KxN* matrices with various swizzling modes,
and are in units of 128b elements as shown
by each colored cell as shown in
[Figure 156](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-mn),
[Figure 157](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-k),
[Figure 158](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-64b-mn),
[Figure 159](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-64b-k),
[Figure 160](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-32b-mn),
[Figure 161](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-32b-k),
[Figure 162](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-mn-interleaved),
[Figure 163](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-k-interleaved).

![_images/async-warpgroup-smem-layout-128B-mn.png](./ptx_files/async-warpgroup-smem-layout-128B-mn.png)


Figure 156 MN major 128B swizzling[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-mn "Permalink to this image")


![_images/async-warpgroup-smem-layout-128B-k.png](./ptx_files/async-warpgroup-smem-layout-128B-k.png)


Figure 157 K major 128B swizzling[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-k "Permalink to this image")


![_images/async-warpgroup-smem-layout-64B-mn.png](./ptx_files/async-warpgroup-smem-layout-64B-mn.png)


Figure 158 MN major 64B swizzling[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-64b-mn "Permalink to this image")


![_images/async-warpgroup-smem-layout-64B-k.png](./ptx_files/async-warpgroup-smem-layout-64B-k.png)


Figure 159 K major 64B swizzling[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-64b-k "Permalink to this image")


![_images/async-warpgroup-smem-layout-32B-mn.png](./ptx_files/async-warpgroup-smem-layout-32B-mn.png)


Figure 160 MN major 32B swizzling[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-32b-mn "Permalink to this image")


![_images/async-warpgroup-smem-layout-32B-k.png](./ptx_files/async-warpgroup-smem-layout-32B-k.png)


Figure 161 K major 32B swizzling[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-32b-k "Permalink to this image")


![_images/async-warpgroup-smem-layout-mn-interleaved.png](./ptx_files/async-warpgroup-smem-layout-mn-interleaved.png)


Figure 162 MN major interleaved[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-mn-interleaved "Permalink to this image")


![_images/async-warpgroup-smem-layout-k-interleaved.png](./ptx_files/async-warpgroup-smem-layout-k-interleaved.png)


Figure 163 K major interleaved[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-k-interleaved "Permalink to this image")

Following are some of the examples of the 128B swizzling layout for `tf32` element type.

* K-Major: [Figure 164](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-k-tf32)

  > ![_images/async-warpgroup-smem-layout-128B-k-tf32.png](./ptx_files/async-warpgroup-smem-layout-128B-k-tf32.png)
  >
  >
  > Figure 164 K major[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-k-tf32 "Permalink to this image")
* MN-Major: [Figure 165](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-mn-tf32)

  > ![_images/async-warpgroup-smem-layout-128B-mn-tf32.png](./ptx_files/async-warpgroup-smem-layout-128B-mn-tf32.png)
  >
  >
  > Figure 165 MN major[](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-smem-layout-128b-mn-tf32 "Permalink to this image")