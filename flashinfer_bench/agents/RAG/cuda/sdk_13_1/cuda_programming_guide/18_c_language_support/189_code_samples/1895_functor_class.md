# 18.9.5. Functor Class

### 18.9.5. Functor Class[](#functor-class "Permalink to this headline")

```
class Add {
public:
    __device__  float operator() (float a, float b) const
    {
        return a + b;
    }
};

class Sub {
public:
    __device__  float operator() (float a, float b) const
    {
        return a - b;
    }
};

// Device code
template<class O> __global__
void VectorOperation(const float * A, const float * B, float * C,
                     unsigned int N, O op)
{
    unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
    if (iElement < N)
        C[iElement] = op(A[iElement], B[iElement]);
}

// Host code
int main()
{
    ...
    VectorOperation<<<blocks, threads>>>(v1, v2, v3, N, Add());
    ...
}
```

9
:   e.g., the `<<<...>>>` syntax for launching kernels.

10
:   This does not apply to entities that may be defined in more than one translation unit, such as compiler generated template instantiations.

[11](#id339)
:   The intent is to allow variable memory space specifiers for static variables in a `__host__ __device__` function during device compilation, but disallow it during host compilation

[12](#id349)
:   One way to debug suspected layout mismatch of a type `C` is to use `printf` to output the values of `sizeof(C)` and `offsetof(C, field)` in host and device code.

[13](#id355)
:   Note that this may negatively impact compile time due to presence of extra declarations.

[14](#id356)
:   At present, the `-std=c++11` flag is supported only for the following host compilers : gcc version >= 4.7, clang, icc >= 15, and xlc >= 13.1

[15](#id358)
:   including `operator()`

[16](#id360)
:   The restrictions are the same as with a non-constexpr callee function.

[17](#id361)
:   Note that the behavior of experimental flags may change in future compiler releases.

[18](#id363)
:   C++ Standard Section `[basic.types]`

[19](#id364)
:   C++ Standard Section `[expr.const]`

[20](#id368)
:   At present, the `-std=c++14` flag is supported only for the following host compilers : gcc version >= 5.1, clang version >= 3.7 and icc version >= 17

[21](#id370)
:   At present, the `-std=c++17` flag is supported only for the following host compilers : gcc version >= 7.0, clang version >= 8.0, Visual Studio version >= 2017, pgi compiler version >= 19.0, icc compiler version >= 19.0

[22](#id373)
:   At present, the `-std=c++20` flag is supported only for the following host compilers : gcc version >= 10.0, clang version >= 10.0, Visual Studio Version >= 2022 and nvc++ version >= 20.7.

[23](#id375)
:   When using the icc host compiler, this flag is only supported for icc >= 1800.

24([1](#id376),[2](#id379))
:   The traits will always return false if extended lambda mode is not active.

[25](#id378)
:   In contrast, the C++ standard specifies that the captured variable is used to direct-initialize the field of the closure type.