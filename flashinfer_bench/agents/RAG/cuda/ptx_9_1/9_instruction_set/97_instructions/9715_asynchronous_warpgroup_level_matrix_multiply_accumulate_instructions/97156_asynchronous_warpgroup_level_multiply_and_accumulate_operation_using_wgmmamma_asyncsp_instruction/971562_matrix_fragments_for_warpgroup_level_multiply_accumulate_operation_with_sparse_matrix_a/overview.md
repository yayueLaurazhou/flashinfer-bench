# 9.7.15.6.2. Matrix fragments for warpgroup-level multiply-accumulate operation with sparse matrix A

##### 9.7.15.6.2. [Matrix fragments for warpgroup-level multiply-accumulate operation with sparse matrix A](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-fragments-for-sparse-wgmma)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-fragments-for-sparse-wgmma "Permalink to this headline")

In this section we describe how the contents of thread registers are associated with fragments of A
matrix and the sparsity metadata.

Each warp in the warpgroup provides sparsity information for 16 rows of matrix A. The following
table shows the assignment of warps to rows of matrix A:

| Warp | Sparsity information for rows of matrix A |
| --- | --- |
| `%warpid` % 4 = 3 | 48-63 |
| `%warpid` % 4 = 2 | 32-47 |
| `%warpid` % 4 = 1 | 16-31 |
| `%warpid` % 4 = 0 | 0-15 |

The following conventions are used throughout this section:

* For matrix A, only the layout of a fragment is described in terms of register vector sizes and
  their association with the matrix data.
* For matrix D, since the matrix dimension - data type combination is the same for all supported
  shapes, and is already covered in
  [Asynchronous Warpgroup Level Matrix Multiply-Accumulate Operation using wgmma.mma\_async instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async), the pictorial
  representations of matrix fragments are not included in this section.
* For the metadata operand, pictorial representations of the association between indices of the
  elements of matrix A and the contents of the metadata operand are included. `Tk: [m..n]` present
  in cell `[x][y..z]` indicates that bits `m` through `n` (with `m` being higher) in the
  metadata operand of thread with `%laneid=k` contains the indices of the non-zero elements from
  the chunk `[x][y]..[x][z]` of matrix A.