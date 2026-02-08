# 4.5. Constants

## 4.5. [Constants](https://docs.nvidia.com/cuda/parallel-thread-execution/#constants)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#constants "Permalink to this headline")

PTX supports integer and floating-point constants and constant expressions. These constants may be
used in data initialization and as operands to instructions. Type checking rules remain the same for
integer, floating-point, and bit-size types. For predicate-type data and instructions, integer
constants are allowed and are interpreted as in C, i.e., zero values are `False` and non-zero
values are `True`.