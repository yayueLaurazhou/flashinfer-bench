# 9.7.14.6. Matrix multiply-accumulate operation using mma.sp instruction with sparse matrix A

#### 9.7.14.6. [Matrix multiply-accumulate operation using `mma.sp` instruction with sparse matrix A](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-sparse-mma)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-sparse-mma "Permalink to this headline")

This section describes warp-level `mma.sp{::ordered_metadata}` instruction with sparse matrix A.
This variant of the `mma` operation can be used when A is a structured sparse matrix with 50%
zeros in each row distributed in a shape-specific granularity. For an `MxNxK` sparse
`mma.sp{::ordered_metadata}` operation, the `MxK` matrix A is packed into `MxK/2` elements.
For each K-wide row of matrix A, 50% elements are zeros and the remaining K/2 non-zero elements
are packed in the operand representing matrix A. The mapping of these K/2 elements to the
corresponding K-wide row is provided explicitly as metadata.