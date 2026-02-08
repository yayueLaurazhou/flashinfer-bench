# 9.7.14.6.2. Matrix fragments for multiply-accumulate operation with sparse matrix A

##### 9.7.14.6.2. [Matrix fragments for multiply-accumulate operation with sparse matrix A](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragments-for-sparse-mma)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragments-for-sparse-mma "Permalink to this headline")

In this section we describe how the contents of thread registers are associated with fragments of
various matrices and the sparsity metadata. The following conventions are used throughout this
section:

* For matrix A, only the layout of a fragment is described in terms of register vector sizes and
  their association with the matrix data.
* For matrix B, when the combination of matrix dimension and the supported data type is not already
  covered in [Matrix multiply-accumulate operation using mma instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma), a pictorial representation of matrix
  fragments is provided.
* For matrices C and D, since the matrix dimension - data type combination is the same for all
  supported shapes, and is already covered in
  [Matrix multiply-accumulate operation using mma instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma), the pictorial representations
  of matrix fragments are not included in this section.
* For the metadata operand, pictorial representations of the association between indices of the
  elements of matrix A and the contents of the metadata operand are included. `Tk: [m..n]` present
  in cell `[x][y..z]` indicates that bits `m` through `n` (with `m` being higher) in the
  metadata operand of thread with `%laneid=k` contains the indices of the non-zero elements from
  the chunk `[x][y]..[x][z]` of matrix A.