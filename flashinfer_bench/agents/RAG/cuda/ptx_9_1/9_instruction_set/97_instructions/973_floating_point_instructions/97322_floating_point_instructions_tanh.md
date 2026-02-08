# 9.7.3.22. Floating Point Instructions: tanh

#### 9.7.3.22. [Floating Point Instructions: `tanh`](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh "Permalink to this headline")

`tanh`

Find the hyperbolic tangent of a value (in radians)

Syntax

```
tanh.approx.f32 d, a;
```

Copy to clipboard

Description

Take hyperbolic tangent value of `a`.

The operands `d` and `a` are of type `.f32`.

Semantics

```
d = tanh(a);
```

Copy to clipboard

Notes

`tanh.approx.f32` implements a fast approximation to FP32 hyperbolic-tangent.

Results of `tanh` for various corner-case inputs are as follows:

| Input | Result |
| --- | --- |
| -Inf | -1.0 |
| -0.0 | -0.0 |
| +0.0 | +0.0 |
| +Inf | 1.0 |
| NaN | NaN |

The maximum relative error over the entire floating point
range is 2-11.
The subnormal numbers are supported.

Note

The subnormal inputs gets passed through to the output since the value of `tanh(x)` for small
values of `x` is approximately the same as `x`.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Target ISA Notes

Requires `sm_75` or higher.

Examples

```
tanh.approx.f32 ta, a;
```

Copy to clipboard