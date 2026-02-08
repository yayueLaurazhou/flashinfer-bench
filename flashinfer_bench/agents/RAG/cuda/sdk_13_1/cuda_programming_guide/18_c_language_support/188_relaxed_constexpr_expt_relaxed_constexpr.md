# 18.8. Relaxed Constexpr (-expt-relaxed-constexpr)

## 18.8. Relaxed Constexpr (-expt-relaxed-constexpr)[](#relaxed-constexpr-expt-relaxed-constexpr "Permalink to this headline")

By default, the following cross-execution space calls are not supported:

1. Calling a `__device__`-only `constexpr` function from a `__host__` function during host code generation phase (i.e, when `__CUDA_ARCH__` macro is undefined). Example:

   > ```
   > constexpr __device__ int D() { return 0; }
   > int main() {
   >     int x = D();  //ERROR: calling a __device__-only constexpr function from host code
   > }
   > ```
2. Calling a `__host__`-only `constexpr` function from a `__device__` or `__global__` function, during device code generation phase (i.e. when `__CUDA_ARCH__` macro is defined). Example:

   > ```
   > constexpr  int H() { return 0; }
   > __device__ void dmain()
   > {
   >     int x = H();  //ERROR: calling a __host__-only constexpr function from device code
   > }
   > ```

The experimental flag `-expt-relaxed-constexpr` can be used to relax this constraint. When this flag is specified, the compiler will support cross execution space calls described above, as follows:

1. A cross-execution space call to a constexpr function is supported if it occurs in a context that requires constant evaluation, e.g., in the initializer of a constexpr variable. Example:

   > ```
   > constexpr __host__ int H(int x) { return x+1; };
   > __global__ void doit() {
   > constexpr int val = H(1); // OK: call is in a context that
   >                           // requires constant evaluation.
   > }
   >
   > constexpr __device__ int D(int x) { return x+1; }
   > int main() {
   > constexpr int val = D(1); // OK: call is in a context that
   >                           // requires constant evaluation.
   > }
   > ```
2. Otherwise:

   > 1. During device code generation, device code is generated for the body of a `__host__`-only constexpr function `H`, unless `H` is not used or is only called in a constexpr context. Example:
   >
   >    > ```
   >    > // NOTE: "H" is emitted in generated device code because it is
   >    > // called from device code in a non-constexpr context
   >    > constexpr __host__ int H(int x) { return x+1; }
   >    >
   >    > __device__ int doit(int in) {
   >    >   in = H(in);  // OK, even though argument is not a constant expression
   >    >   return in;
   >    > }
   >    > ```
   > 2. **All code restrictions applicable to a ``\_\_device\_\_`` function are also applicable to the ``constexpr host``-only function ``H`` that is called from device code. However, compiler may not emit any build time diagnostics for ``H`` for these restrictions** [8](#frelaxedconstexpr1) .
   >
   >    > For example, the following code patterns are unsupported in the body of `H` (as with any `__device__` function), but no compiler diagnostic may be generated:
   >    >
   >    > > * ODR-use of a host variable or `__host__`-only non-constexpr function. Example:
   >    > >
   >    > >   > ```
   >    > >   > int qqq, www;
   >    > >   >
   >    > >   > constexpr __host__ int* H(bool b) { return b ? &qqq : &www; };
   >    > >   >
   >    > >   > __device__ int doit(bool flag) {
   >    > >   >   int *ptr;
   >    > >   >   ptr = H(flag); // ERROR: H() attempts to refer to host variables 'qqq' and 'www'.
   >    > >   >                  // code will compile, but will NOT execute correctly.
   >    > >   >   return *ptr;
   >    > >   > }
   >    > >   > ```
   >    > > * Use of exceptions (`throw/catch`) and RTTI (`typeid, dynamic_cast`). Example:
   >    > >
   >    > >   > ```
   >    > >   > struct Base { };
   >    > >   > struct Derived : public Base { };
   >    > >   >
   >    > >   > // NOTE: "H" is emitted in generated device code
   >    > >   > constexpr int H(bool b, Base *ptr) {
   >    > >   >   if (b) {
   >    > >   >     return 1;
   >    > >   >   } else if (typeid(ptr) == typeid(Derived)) { // ERROR: use of typeid in code executing on the GPU
   >    > >   >     return 2;
   >    > >   >   } else {
   >    > >   >     throw int{4}; // ERROR: use of throw in code executing on the GPU
   >    > >   >   }
   >    > >   > }
   >    > >   > __device__ void doit(bool flag) {
   >    > >   >   int val;
   >    > >   >   Derived d;
   >    > >   >   val = H(flag, &d); //ERROR: H() attempts use typeid and throw(), which are not allowed in code that executes on the GPU
   >    > >   > }
   >    > >   > ```
   > 3. During host code generation, the body of a `__device__`-only constexpr function `D` is preserved in the code sent to the host compiler. If the body of `D` attempts to ODR-use a namespace scope device variable or a `__device__`-only non-constexpr function, then the call to `D` from host code is not supported (code may build without compiler diagnostics, but may behave incorrectly at run time). Example:
   >
   >    > ```
   >    > __device__ int qqq, www;
   >    > constexpr __device__ int* D(bool b) { return b ? &qqq : &www; };
   >    >
   >    > int doit(bool flag) {
   >    >   int *ptr;
   >    >   ptr = D(flag); // ERROR: D() attempts to refer to device variables 'qqq' and 'www'
   >    >                  // code will compile, but will NOT execute correctly.
   >    >   return *ptr;
   >    > }
   >    > ```
   > 4. **Note: Given above restrictions and lack of compiler diagnostics for incorrect usage, be careful when calling a constexpr \_\_host\_\_ function in the standard C++ headers from device code**, since the implementation of the function will vary depending on the host platform, e.g., based on the `libstdc++` version for gcc host compiler. Such code may break silently when being ported to a different platform or host compiler version (if the target C++ library implementation odr-uses a host code variable or function, as described earlier).
   >
   >    > Example:
   >    >
   >    > ```
   >    > __device__ int get(int in) {
   >    >  int val = std::foo(in); // "std::foo" is constexpr function defined in the host compiler's standard library header
   >    >                          // WARNING: if std::foo implementation ODR-uses host variables or functions,
   >    >                          // code will not work correctly
   >    > }
   >    > ```

[8](#id380)
:   Diagnostics are usually generated during parsing, but the host-only function `H` may already have been parsed before the call to `H` from device code is encountered later in the translation unit.