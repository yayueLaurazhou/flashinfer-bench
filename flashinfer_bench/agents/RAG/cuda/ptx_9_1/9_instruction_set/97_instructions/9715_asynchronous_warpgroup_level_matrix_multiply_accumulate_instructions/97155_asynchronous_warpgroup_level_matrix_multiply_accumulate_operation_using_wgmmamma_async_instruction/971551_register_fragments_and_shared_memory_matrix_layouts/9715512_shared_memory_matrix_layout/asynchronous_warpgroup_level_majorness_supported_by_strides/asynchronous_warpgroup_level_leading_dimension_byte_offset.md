# asynchronous-warpgroup-level-leading-dimension-byte-offset

###### 9.7.15.5.1.2.1.1. [Leading Dimension Byte Offset](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset "Permalink to this headline")

The leading dimension byte offset is defined differently for transposed and non-transposed
matrices. The leading byte offset is defined as follows for matrices whose element types are
normalized to 128-bits:

| Major-ness | Definition |
| --- | --- |
| K-Major | * No-Swizzling: the offset from the first column to the second columns   of the 8x2 tile in the 128-bit element type normalized matrix. * Swizzled layouts: not used, assumed to be 1. |
| MN-Major | * Interleave: offset from the first 8 columns to the next 8 columns. * Swizzled layouts: offset from the first (swizzle-byte-size/16) rows   to the next (swizzle-byte-size/16) rows. |