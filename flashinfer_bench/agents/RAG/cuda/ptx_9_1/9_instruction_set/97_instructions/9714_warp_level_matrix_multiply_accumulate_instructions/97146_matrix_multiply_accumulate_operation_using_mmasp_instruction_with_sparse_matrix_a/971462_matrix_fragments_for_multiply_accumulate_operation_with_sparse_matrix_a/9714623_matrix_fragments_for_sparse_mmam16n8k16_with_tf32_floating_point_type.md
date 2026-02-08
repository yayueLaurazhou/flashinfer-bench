# 9.7.14.6.2.3. Matrix Fragments for sparse mma.m16n8k16 with .tf32 floating point type

###### 9.7.14.6.2.3. [Matrix Fragments for sparse `mma.m16n8k16` with `.tf32` floating point type](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-sparse-mma-16816-tf32)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-sparse-mma-16816-tf32 "Permalink to this headline")

A warp executing sparse `mma.m16n8k16` with `.tf32` floating point type will compute an MMA
operation of shape `.m16n8k16`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.tf32` | A vector expression containing four `.b32` registers, with each register containing one non-zero `.tf32` element out of 2 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-sparse-matrix-storage). |

  The layout of the fragments held by different threads is shown in [Figure 123](https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-mma-16816-tf32-a).

  ![_images/sparse-mma-16816-tf32-A.png](./ptx_files/sparse-mma-16816-tf32-A.png)


  Figure 123 Sparse MMA .m16n8k16 fragment layout for matrix A with `.tf32` type.[](https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-mma-16816-tf32-a "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2
  threadID_in_group = %laneid % 4

  row =      groupID            for a0 and a2
             groupID + 8        for a1 and a3

  col = [firstcol ... lastcol]  // As per the mapping of non-zero elements
                                // as described in Sparse matrix storage

  Where
  firstcol = threadID_in_group * 2          for a0 and a1
             (threadID_in_group * 2) + 8    for a2 and a3
  lastcol  = firstcol + 1
  ```

  Copy to clipboard
* Multiplicand B:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.tf32` | A vector expression containing four `.b32` registers, each containing four `.tf32` elements from matrix B. | b0, b1, b2, b3 |

  The layout of the fragments held by different threads is shown in [Figure 124](https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-mma-16816-tf32-b).

  ![_images/sparse-mma-16816-tf32-B.png](./ptx_files/sparse-mma-16816-tf32-B.png)


  Figure 124 Sparse MMA .m16n8k16 fragment layout for matrix B with `.tf32` type.[](https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-mma-16816-tf32-b "Permalink to this image")
* Matrix fragments for accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k16 with floating point type](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float).
* Metadata: A `.b32` register containing 8 4-bit vectors each storing the index of a non-zero
  element of a 2-wide chunk of matrix A as shown in [Figure 125](https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-mma-metadata-16816-tf32).

  > ![_images/sparse-mma-metadata-16816-tf32.png](./ptx_files/sparse-mma-metadata-16816-tf32.png)
  >
  >
  > Figure 125 Sparse MMA .m16n8k16 metadata layout for `.tf32` type.[](https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-mma-metadata-16816-tf32 "Permalink to this image")