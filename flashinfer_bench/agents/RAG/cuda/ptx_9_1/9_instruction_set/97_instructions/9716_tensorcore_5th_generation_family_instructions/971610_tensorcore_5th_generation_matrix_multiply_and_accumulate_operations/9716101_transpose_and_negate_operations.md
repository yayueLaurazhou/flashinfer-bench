# 9.7.16.10.1. Transpose and Negate operations

##### 9.7.16.10.1. [Transpose and Negate operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-transpose-and-negate-operations)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-transpose-and-negate-operations "Permalink to this headline")

The matrices `A` and `B` can be transposed by specifying the Tranpose `A` Matrix
and Transpose `B` Matrix bits in the instruction descriptor respectively.

The elements of the matrices `A` and `B` can be negated by specifying the Negate
`A` Matrix and Negate `B` Matrix bits in the instruction descriptor respectively.

The support for Transpose and Negate operation for various MMA-Kind are shown in
[Table 51](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-transpose-negate-mma-kind).

Table 51 Transpose and Negate operation for various MMA-Kind[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-transpose-negate-mma-kind "Permalink to this table")





| MMA-Kind | Is Transpose A/B supported | Is Negate A/B supported |
| --- | --- | --- |
| `.kind::tf32` | Yes | Yes |
| `.kind::f16` | Yes | Yes |
| `.kind::f8f6f4` | Yes | Yes |
| `.kind::mxf8f6f4` | Yes | Yes |
| `.kind::i8` | Yes | No |
| `.kind::mxf4` | No | Yes |
| `.kind::mxf4nvf4` | No | Yes |

For `.kind::tf32`, the transpose operations on matrices `A` and `B` are supported
only with 128B swizzling mode with 32B swizzle-atomicity.

For all other MMA-Kinds, the transpose operations on matrices `A` and `B` are not supported
on 128B swizzling mode with 32B swizzle-atomicity.

[Table 52](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes-8b-transpose-b) shows the valid combinations of N shape with
`.cta_group` qualifier for 8bit transpose B.

Table 52 Various combinations of N shape with .cta\_group qualifier for 8bit transpose B[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes-8b-transpose-b "Permalink to this table")




| .cta\_group | N shape |
| --- | --- |
| 1 | 16 <= N <= 256, step 16 |
| 2 | 32 <= N <= 256, step 32 |