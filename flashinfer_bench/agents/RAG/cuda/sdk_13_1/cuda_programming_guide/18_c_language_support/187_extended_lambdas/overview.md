# 18.7. Extended Lambdas

## 18.7. Extended Lambdas[](#extended-lambdas "Permalink to this headline")

The nvcc flag `'--extended-lambda'` allows explicit execution space annotations in a lambda expression [23](#fn30). The execution space annotations should be present after the ‘lambda-introducer’ and before the optional ‘lambda-declarator’. nvcc will define the macro `__CUDACC_EXTENDED_LAMBDA__` when the `'--extended-lambda'` flag has been specified.

An ‘extended `__device__` lambda’ is a lambda expression that is annotated explicitly with ‘`__device__`’, and is defined within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

An ‘extended `__host__ __device__` lambda’ is a lambda expression that is annotated explicitly with both ‘`__host__`’ and ‘`__device__`’, and is defined within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

An ‘extended lambda’ denotes either an extended `__device__` lambda or an extended `__host__ __device__` lambda. Extended lambdas can be used in the type arguments of [\_\_global\_\_ function template instantiation](#cpp11-global).

If the execution space annotations are not explicitly specified, they are computed based on the scopes enclosing the closure class associated with the lambda, as described in the section on C++11 support. The execution space annotations are applied to all methods of the closure class associated with the lambda.

Example:

```
void foo_host(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };

  // extended __device__ lambda
  auto lam2 = [] __device__ { };

  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };

  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__host__ __device__ void foo_host_device(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };

  // extended __device__ lambda
  auto lam2 = [] __device__ { };

  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };

  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__device__ void foo_device(void) {
  // none of the lambdas within this function are extended lambdas,
  // because the enclosing function is not a __host__ or __host__ __device__
  // function.
  auto lam1 = [] { };
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };
  auto lam4 = [] __host__ { };
}

// lam1 and lam2 are not extended lambdas because they are not defined
// within a __host__ or __host__ __device__ function.
auto lam1 = [] { };
auto lam2 = [] __host__ __device__ { };
```