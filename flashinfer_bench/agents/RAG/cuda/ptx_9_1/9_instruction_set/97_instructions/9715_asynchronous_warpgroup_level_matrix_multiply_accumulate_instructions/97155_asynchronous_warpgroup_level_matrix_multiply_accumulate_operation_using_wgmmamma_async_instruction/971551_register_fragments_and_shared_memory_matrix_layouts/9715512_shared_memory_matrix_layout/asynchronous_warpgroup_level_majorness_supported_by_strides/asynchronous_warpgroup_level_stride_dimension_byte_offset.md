# asynchronous-warpgroup-level-stride-dimension-byte-offset

###### 9.7.15.5.1.2.1.2. [Stride Dimension Byte Offset](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-stride-dimension-byte-offset)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-stride-dimension-byte-offset "Permalink to this headline")

The stride dimension byte offset is defined differently for transposed and non-transposed
matrices. The stride dimension byte offset is defined as follows for matrices whose element
types are normalized to 128-bits:

| Major-ness | Definition |
| --- | --- |
| K-Major | The offset from the first 8 rows to the next 8 rows. |
| MN-Major | * Interleave: offset from the first row to the next row. * Swizzled layout: offset from the first 8 columns to the next 8   columns |