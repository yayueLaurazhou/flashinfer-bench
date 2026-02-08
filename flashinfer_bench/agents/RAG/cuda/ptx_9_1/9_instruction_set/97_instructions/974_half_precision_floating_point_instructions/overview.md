# 9.7.4. Half Precision Floating-Point Instructions

### 9.7.4. [Half Precision Floating-Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions "Permalink to this headline")

Half precision floating-point instructions operate on `.f16` and `.f16x2` register operands. The
half precision floating-point instructions are:

* `add`
* `sub`
* `mul`
* `fma`
* `neg`
* `abs`
* `min`
* `max`
* `tanh`
* `ex2`

Half-precision `add`, `sub`, `mul`, and `fma` support saturation of results to the range
[0.0, 1.0], with `NaN`s being flushed to positive zero. Half-precision instructions return an
unspecified `NaN`.