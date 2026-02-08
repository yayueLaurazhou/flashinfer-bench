# 10.24.6. Element Types and Matrix Sizes

### 10.24.6. Element Types and Matrix Sizes[](#element-types-and-matrix-sizes "Permalink to this headline")

Tensor Cores support a variety of element types and matrix sizes. The following table presents the various combinations of `matrix_a`, `matrix_b` and `accumulator` matrix supported:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| \_\_half | \_\_half | float | 16x16x16 |
| \_\_half | \_\_half | float | 32x8x16 |
| \_\_half | \_\_half | float | 8x32x16 |
| \_\_half | \_\_half | \_\_half | 16x16x16 |
| \_\_half | \_\_half | \_\_half | 32x8x16 |
| \_\_half | \_\_half | \_\_half | 8x32x16 |
| unsigned char | unsigned char | int | 16x16x16 |
| unsigned char | unsigned char | int | 32x8x16 |
| unsigned char | unsigned char | int | 8x32x16 |
| signed char | signed char | int | 16x16x16 |
| signed char | signed char | int | 32x8x16 |
| signed char | signed char | int | 8x32x16 |

Alternate Floating Point support:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| \_\_nv\_bfloat16 | \_\_nv\_bfloat16 | float | 16x16x16 |
| \_\_nv\_bfloat16 | \_\_nv\_bfloat16 | float | 32x8x16 |
| \_\_nv\_bfloat16 | \_\_nv\_bfloat16 | float | 8x32x16 |
| precision::tf32 | precision::tf32 | float | 16x16x8 |

Double Precision Support:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| double | double | double | 8x8x4 |

Experimental support for sub-byte operations:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| precision::u4 | precision::u4 | int | 8x8x32 |
| precision::s4 | precision::s4 | int | 8x8x32 |
| precision::b1 | precision::b1 | int | 8x8x128 |