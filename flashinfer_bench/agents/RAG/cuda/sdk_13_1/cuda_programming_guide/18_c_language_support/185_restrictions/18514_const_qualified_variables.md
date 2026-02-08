# 18.5.14. Const-qualified variables

### 18.5.14. Const-qualified variables[](#const-qualified-variables "Permalink to this headline")

Let ‘V’ denote a namespace scope variable or a class static member variable that has const qualified type and does not have execution space annotations (for example, `__device__, __constant__, __shared__`). V is considered to be a host code variable.

The value of V may be directly used in device code, if

* V has been initialized with a constant expression before the point of use,
* the type of V is not volatile-qualified, and
* it has one of the following types:

  + built-in floating point type except when the Microsoft compiler is used as the host compiler,
  + built-in integral type.

Device source code cannot contain a reference to V or take the address of V.

Example:

```
const int xxx = 10;
struct S1_t {  static const int yyy = 20; };

extern const int zzz;
const float www = 5.0;
__device__ void foo(void) {
  int local1[xxx];          // OK
  int local2[S1_t::yyy];    // OK

  int val1 = xxx;           // OK

  int val2 = S1_t::yyy;     // OK

  int val3 = zzz;           // error: zzz not initialized with constant
                            // expression at the point of use.

  const int &val3 = xxx;    // error: reference to host variable
  const int *val4 = &xxx;   // error: address of host variable
  const float val5 = www;   // OK except when the Microsoft compiler is used as
                            // the host compiler.
}
const int zzz = 20;
```