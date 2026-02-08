# 4.5.3. Predicate Constants

### 4.5.3. [Predicate Constants](https://docs.nvidia.com/cuda/parallel-thread-execution/#predicate-constants)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#predicate-constants "Permalink to this headline")

In PTX, integer constants may be used as predicates. For predicate-type data initializers and
instruction operands, integer constants are interpreted as in C, i.e., zero values are `False` and
non-zero values are `True`.