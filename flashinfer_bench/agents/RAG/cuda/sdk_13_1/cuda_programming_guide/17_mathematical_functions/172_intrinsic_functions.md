# 17.2. Intrinsic Functions

## 17.2. Intrinsic Functions[](#intrinsic-functions "Permalink to this headline")

The functions from this section can only be used in device code.

Among these functions are the less accurate, but faster versions of some of the functions of [Standard Functions](#mathematical-functions-appendix-standard-functions).
They have the same name prefixed with `__` (such as `__sinf(x)`).
They are faster as they map to fewer native instructions.
The compiler has an option (`-use_fast_math`) that forces each function in [Table 20](#intrinsic-functions-functions-affected-use-fast-math)
to compile to its intrinsic counterpart. In addition to reducing the accuracy of the affected functions,
it may also cause some differences in special case handling. A more robust approach is to selectively replace
mathematical function calls by calls to intrinsic functions only where it is merited by the performance gains
and where changed properties such as reduced accuracy and different special case handling can be tolerated.

Table 20 Functions Affected by -use\_fast\_math[](#intrinsic-functions-functions-affected-use-fast-math "Permalink to this table")




| Operator/Function | Device Function |
| --- | --- |
| `x/y` | `__fdividef(x,y)` |
| `sinf(x)` | `__sinf(x)` |
| `cosf(x)` | `__cosf(x)` |
| `tanf(x)` | `__tanf(x)` |
| `sincosf(x,sptr,cptr)` | `__sincosf(x,sptr,cptr)` |
| `logf(x)` | `__logf(x)` |
| `log2f(x)` | `__log2f(x)` |
| `log10f(x)` | `__log10f(x)` |
| `expf(x)` | `__expf(x)` |
| `exp10f(x)` | `__exp10f(x)` |
| `powf(x,y)` | `__powf(x,y)` |
| `tanhf(x)` | `__tanhf(x)` |

**Single-Precision Floating-Point Functions**

`__fadd_[rn,rz,ru,rd]()` and `__fmul_[rn,rz,ru,rd]()` map to addition and multiplication operations that the compiler never merges into FMADs. By contrast, additions and multiplications generated from the ‘\*’ and ‘+’ operators will frequently be combined into FMADs.

Functions suffixed with `_rn` operate using the round to nearest even rounding mode.

Functions suffixed with `_rz` operate using the round towards zero rounding mode.

Functions suffixed with `_ru` operate using the round up (to positive infinity) rounding mode.

Functions suffixed with `_rd` operate using the round down (to negative infinity) rounding mode.

The accuracy of floating-point division varies depending on whether the code is compiled with `-prec-div=false`
or `-prec-div=true`. When the code is compiled with `-prec-div=false`, both the regular division `/`
operator and `__fdividef(x,y)` have the same accuracy, but for 2126 < `|y|` < 2128,
`__fdividef(x,y)` delivers a result of zero, whereas the `/` operator delivers the correct result to
within the accuracy stated in [Table 21](#intrinsic-functions-single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library).
Also, for 2126 < `|y|` < 2128, if `x` is infinity, `__fdividef(x,y)` delivers
a `NaN` (as a result of multiplying infinity by zero), while the `/` operator returns infinity.
On the other hand, the `/` operator is IEEE-compliant when the code is compiled with `-prec-div=true`
or without any `-prec-div` option at all since its default value is true.

Table 21 Single-Precision Floating-Point Intrinsic Functions. (Supported by the CUDA Runtime Library with Respective Error Bounds)[](#intrinsic-functions-single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library "Permalink to this table")




| Function | Error bounds |
| --- | --- |
| `__fadd_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fsub_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fmul_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fmaf_[rn,rz,ru,rd](x,y,z)` | IEEE-compliant. |
| `__frcp_[rn,rz,ru,rd](x)` | IEEE-compliant. |
| `__fsqrt_[rn,rz,ru,rd](x)` | IEEE-compliant. |
| `__frsqrt_rn(x)` | IEEE-compliant. |
| `__fdiv_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fdividef(x,y)` | For `|y|` in [\(2^{-126}, 2^{126}\)], the maximum ulp error is 2. |
| `__expf(x)` | The maximum ulp error is `2 + floor(abs(1.173 * x))`. |
| `__exp10f(x)` | The maximum ulp error is `2 + floor(abs(2.97 * x))`. |
| `__logf(x)` | For `x` in [0.5, 2], the maximum absolute error is \(2^{-21.41}\), otherwise, the maximum ulp error is 3. |
| `__log2f(x)` | For `x` in [0.5, 2], the maximum absolute error is \(2^{-22}\), otherwise, the maximum ulp error is 2. |
| `__log10f(x)` | For `x` in [0.5, 2], the maximum absolute error is \(2^{-24}\), otherwise, the maximum ulp error is 3. |
| `__sinf(x)` | For `x` in [\(-\pi, \pi\)], the maximum absolute error is \(2^{-21.41}\), and larger otherwise. |
| `__cosf(x)` | For `x` in [\(-\pi, \pi\)], the maximum absolute error is \(2^{-21.19}\), and larger otherwise. |
| `__sincosf(x,sptr,cptr)` | Same as `__sinf(x)` and `__cosf(x)`. |
| `__tanf(x)` | Derived from its implementation as `__sinf(x) * (1/__cosf(x))`. |
| `__powf(x, y)` | Derived from its implementation as `exp2f(y * __log2f(x))`. |
| `__tanhf(x)` | The maximum relative error of the current implementation is \(2^{-11}\). Subnormal results of this fast intrinsic are not flushed to zero even under `-ftz=true` compiler setting. Available for devices with compute capability of at least 7.5; and defaults to regular `tanhf()` function behavior on other devices. |

**Double-Precision Floating-Point Functions**

`__dadd_rn()` and `__dmul_rn()` map to addition and multiplication operations that the compiler never merges into FMADs. By contrast, additions and multiplications generated from the ‘\*’ and ‘+’ operators will frequently be combined into FMADs.

Table 22 Double-Precision Floating-Point Intrinsic Functions. (Supported by the CUDA Runtime Library with Respective Error Bounds)[](#id482 "Permalink to this table")




| Function | Error bounds |
| --- | --- |
| `__dadd_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__dsub_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__dmul_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fma_[rn,rz,ru,rd](x,y,z)` | IEEE-compliant. |
| `__ddiv_[rn,rz,ru,rd](x,y)(x,y)` | IEEE-compliant.  Requires compute capability *>* 2. |
| `__drcp_[rn,rz,ru,rd](x)` | IEEE-compliant.  Requires compute capability *>* 2. |
| `__dsqrt_[rn,rz,ru,rd](x)` | IEEE-compliant.  Requires compute capability *>* 2. |