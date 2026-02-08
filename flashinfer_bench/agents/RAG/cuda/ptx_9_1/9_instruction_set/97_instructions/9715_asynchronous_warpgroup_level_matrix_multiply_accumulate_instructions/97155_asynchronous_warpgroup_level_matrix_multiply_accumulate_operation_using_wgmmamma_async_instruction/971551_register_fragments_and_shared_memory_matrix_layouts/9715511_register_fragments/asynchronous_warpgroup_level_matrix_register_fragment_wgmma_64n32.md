# asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n32

###### 9.7.15.5.1.1.3. [Matrix Fragments for `wgmma.mma_async.m64nNk32`](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n32)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n32 "Permalink to this headline")

A warpgroup executing `wgmma.mma_async.m64nNk32` will compute an MMA operation of shape
`.m64nNk32` where N is a valid `n` dimension as listed in
[Matrix Shape](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shape).

Elements of the matrix are distributed across the threads in a warpgroup so each thread of the
warpgroup holds a fragment of the matrix.

* Multiplicand A in registers:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s8`/`.u8` | A vector expression containing four `.b32` registers, with each register containing four `.u8`/ `.s8` elements from matrix A. | a0, a1, a2, a3, … , a14, a15 |
  | `.e4m3`/ `.e5m2` | A vector expression containing four `.b32` registers, with each register containing four `.e4m3`/ `.e5m2` elements from matrix A. |

  The layout of the fragments held by different threads is shown in [Figure 152](https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n32-a).

  ![_images/wgmma-64N32-A.png](./ptx_files/wgmma-64N32-A.png)


  Figure 152 WGMMA .m64nNk32 register fragment layout for matrix A.[](https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n32-a "Permalink to this image")
* Accumulator D:

  | .dtype | Fragment | Elements (low to high) | Miscellaneous Information |
  | --- | --- | --- | --- |
  | `.s32` | A vector expression containing N/2 number of `.s32` registers. | d0, d1, d2, d3, …, dX, dY, dZ, dW  where `X = N/2  -  4`  `Y = N/2  -  3`  `Z = N/2  -  2`  `W = N/2  -  1`  `N` depends on .dtype, as described in the next column. | `N = 8*i where i = {1, 2, 3, 4}`  `= 16*i where i = {3, 4, ..., 15, 16}` |
  | `.f32` | A vector expression containing N/2 number of `.f32` registers. | `N = 8*i where i = {1, 2, ... , 32}` |
  | `.f16` | A vector expression containing N/4 number of `.f16x2` registers, with each register containing two `.f16` elements from matrix D. |

  The layout of the fragments held by different threads is shown in [Figure 153](https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n32-d).

  ![_images/wgmma-64N32-D.png](./ptx_files/wgmma-64N32-D.png)


  Figure 153 WGMMA .m64nNk32 register fragment layout for accumulator matrix D.[](https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n32-d "Permalink to this image")