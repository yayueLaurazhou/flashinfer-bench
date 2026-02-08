# 9.7.15.6. Asynchronous Warpgroup Level Multiply-and-Accumulate Operation using wgmma.mma_async.sp instruction

#### 9.7.15.6. [Asynchronous Warpgroup Level Multiply-and-Accumulate Operation using `wgmma.mma_async.sp` instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-for-sparse-wgmma)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-for-sparse-wgmma "Permalink to this headline")

This section describes warp-level `wgmma.mma_async.sp` instruction with sparse matrix A. This
variant of the `wgmma.mma_async` operation can be used when A is a structured sparse matrix with
50% zeros in each row distributed in a shape-specific granularity. For an `MxNxK` sparse
`wgmma.mma_async.sp` operation, the `MxK` matrix A is packed into `MxK/2` elements. For each
K-wide row of matrix A, 50% elements are zeros and the remaining `K/2` non-zero elements are
packed in the operand representing matrix A. The mapping of these `K/2` elements to the
corresponding K-wide row is provided explicitly as metadata.