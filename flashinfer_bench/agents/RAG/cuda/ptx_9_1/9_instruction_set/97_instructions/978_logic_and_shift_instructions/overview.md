# 9.7.8. Logic and Shift Instructions

### 9.7.8. [Logic and Shift Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions "Permalink to this headline")

The logic and shift instructions are fundamentally untyped, performing bit-wise operations on
operands of any type, provided the operands are of the same size. This permits bit-wise operations
on floating point values without having to define a union to access the bits. Instructions `and`,
`or`, `xor`, and `not` also operate on predicates.

The logical shift instructions are:

* `and`
* `or`
* `xor`
* `not`
* `cnot`
* `lop3`
* `shf`
* `shl`
* `shr`