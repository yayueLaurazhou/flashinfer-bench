# 9.7.16.10.8. Sparse Matrices

##### 9.7.16.10.8. [Sparse Matrices](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices "Permalink to this headline")

This instruction `tcgen05.mma.sp` can be used when the matrix `A` is a structured
sparse matrix with 50% zeros in each row distributed as per its sparse granularity.

In a *MxNxK* sparse `tcgen05.mma.sp` operation, the matrix `A` of shape *MxK* is
stored in a packed form as *Mx(K/2)* in memory. For each *K-wide* row of matrix `A`,
50% of elements are zeros and the remaining *K/2* non-zero elements are stored in
memory. The metadata specifies the mapping of the *K/2* non-zero elements to the *K*
elements before performing the MMA operation.

Granularity of sparse matrix `A` is defined as the ratio of the number of non-zero
elements in a sub-chunk of the matrix row to the total number of elements in that
sub-chunk where the size of the sub-chunk is shape-specific. The following table lists
the granularity of different `tcgen05.mma.sp` variants:

| .kind of tcgen05.mma | Sparse Granularity |
| --- | --- |
| `.kind::tf32` | 1:2 |
| `.kind::f16` | 2:4 |
| `.kind::f8f6f4` |
| `.kind::mxf8f6f4` |
| `.kind::i8` |
| `.kind::mxf4` | 4:8 (in pairs) |