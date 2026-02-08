# 5.5.5.2. Traversal Stride

#### 5.5.5.2. [Traversal Stride](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes-traversal-stride)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes-traversal-stride "Permalink to this headline")

This is similar to im2col mode with the exception of that the number of elements traversed
along only the `W` dimension is strided by the traversal stride as specified in the TensorMap.