# tcgen05-leading-dimension-byte-offset-absolute-address-restriction

###### 9.7.16.3.1.2.1. [Restrictions on the Leading Dimension Absolute Address Stride](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-leading-dimension-byte-offset-absolute-address-restriction)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-leading-dimension-byte-offset-absolute-address-restriction "Permalink to this headline")

Following are the restrictions on the absolute address stride mode:

1. Only 128B swizzle (with 16B atomicity) is supported.
2. Only K-Major mode is supported. That is, the transpose bits(bits #15 and #16) in
   [Instruction descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor) must be 0.
3. The matrix base offset must be 0.