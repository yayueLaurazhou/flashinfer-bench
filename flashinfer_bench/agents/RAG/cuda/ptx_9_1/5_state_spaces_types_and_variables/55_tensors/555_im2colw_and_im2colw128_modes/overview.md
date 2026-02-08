# 5.5.5. im2col::w and im2col::w::128 modes

### 5.5.5. [`im2col::w` and `im2col::w::128` modes](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes "Permalink to this headline")

These modes are similar to the im2col mode with the restriction that elements are accessed across
the `W` dimension only while keeping the `H` and `D` dimension constant.

All the constraints and rules of the im2col mode apply to these modes as well.

The number of elements accessed in the `im2col::w::128` mode is fixed and is equal to 128.
The number of elements accessed in the `im2col::w` mode depends on the field Pixels-per-Column
field in the TensorMap.