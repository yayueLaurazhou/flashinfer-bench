# 9.7.3.17. Floating Point Instructions: rsqrt.approx.ftz.f64

#### 9.7.3.17. [Floating Point Instructions: `rsqrt.approx.ftz.f64`](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt-approx-ftz-f64)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt-approx-ftz-f64 "Permalink to this headline")

`rsqrt.approx.ftz.f64`

Compute an approximation of the square root reciprocal of a value.

Syntax

```
rsqrt.approx.ftz.f64 d, a;
```

Copy to clipboard

Description

Compute a double-precision (`.f64`) approximation of the square root reciprocal of a value. The
least significant 32 bits of the double-precision (`.f64`) destination `d` are all zeros.

Semantics

```
tmp = a[63:32]; // upper word of a, 1.11.20 format
d[63:32] = 1.0 / sqrt(tmp);
d[31:0] = 0x00000000;
```

Copy to clipboard

Notes

`rsqrt.approx.ftz.f64` implements a fast approximation of the square root reciprocal of a value.

| Input | Result |
| --- | --- |
| -Inf | NaN |
| -subnormal | -Inf |
| -0.0 | -Inf |
| +0.0 | +Inf |
| +subnormal | +Inf |
| +Inf | +0.0 |
| NaN | NaN |

Input `NaN`s map to a canonical `NaN` with encoding `0x7fffffff00000000`.

Subnormal inputs and results are flushed to sign-preserving zero.

PTX ISA Notes

`rsqrt.approx.ftz.f64` introduced in PTX ISA version 4.0.

Target ISA Notes

`rsqrt.approx.ftz.f64` requires `sm_20` or higher.

Examples

```
rsqrt.approx.ftz.f64 xi,x;
```

Copy to clipboard